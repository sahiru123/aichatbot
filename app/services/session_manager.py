# app/services/session_manager.py
from datetime import datetime, timedelta
import jwt
from app.models.database import Session, User
from sqlalchemy.orm import Session as DBSession
import uuid
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, db_session: DBSession, secret_key: str):
        self.db = db_session
        self.secret_key = secret_key
        self.session_duration = timedelta(hours=24)

    def create_session(self, user: User) -> str:
        try:
            # Create session record
            session_id = str(uuid.uuid4())
            expires_at = datetime.utcnow() + self.session_duration
            
            session = Session(
                id=session_id,
                user_id=user.id,
                token=str(uuid.uuid4()),
                expires_at=expires_at
            )
            
            # Clean up any existing expired sessions for this user
            self.db.query(Session).filter(
                Session.user_id == user.id,
                Session.expires_at <= datetime.utcnow()
            ).delete()
            
            # Add new session
            self.db.add(session)
            self.db.commit()

            # Create JWT token
            token_payload = {
                'session_id': session.id,
                'user_id': str(user.id),
                'exp': int(expires_at.timestamp())
            }
            
            token = jwt.encode(token_payload, self.secret_key, algorithm='HS256')
            
            logger.info(f"Created new session for user {user.id}")
            return token

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating session: {str(e)}")
            raise

    def validate_session(self, token: str) -> User:
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Get session from database
            session = self.db.query(Session).filter(
                Session.id == payload['session_id']
            ).first()
            
            if not session:
                logger.error(f"Session not found: {payload['session_id']}")
                raise ValueError("Session not found")
                
            if session.expires_at <= datetime.utcnow():
                logger.error(f"Session expired: {session.id}")
                # Clean up expired session
                self.db.delete(session)
                self.db.commit()
                raise ValueError("Session expired")
                
            # Get associated user
            user = self.db.query(User).filter(User.id == session.user_id).first()
            if not user:
                logger.error(f"User not found for session: {session.id}")
                raise ValueError("User not found")
                
            return user

        except jwt.ExpiredSignatureError:
            logger.error("JWT token expired")
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid JWT token: {str(e)}")
            raise ValueError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Session validation error: {str(e)}")
            raise ValueError(str(e))

    def end_session(self, token: str) -> None:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            session = self.db.query(Session).filter(
                Session.id == payload['session_id']
            ).first()
            
            if session:
                self.db.delete(session)
                self.db.commit()
                logger.info(f"Ended session: {session.id}")
        except Exception as e:
            logger.error(f"Error ending session: {str(e)}")
            # Don't raise the error as this is a cleanup operation
            pass