from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
security = HTTPBearer()

async def auth_middleware(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    try:
        session_manager = request.app.state.session_manager
        token = credentials.credentials
        
        # Log token details (careful not to log the full token)
        logger.info(f"Processing authentication for token starting with: {token[:10]}...")
        
        try:
            # First try to decode the token
            payload = jwt.decode(token, session_manager.secret_key, algorithms=['HS256'])
            logger.info(f"Token decoded successfully. Session ID: {payload.get('session_id')}")
            
            # Check session in database
            session = session_manager.db.query(Session).filter(
                Session.id == payload['session_id']
            ).first()
            
            if not session:
                logger.error(f"No session found for session_id: {payload.get('session_id')}")
                raise ValueError("Session not found")
            
            # Check expiration
            if session.expires_at <= datetime.utcnow():
                logger.error(f"Session expired at {session.expires_at}")
                raise ValueError("Session expired")
            
            # Log successful authentication
            logger.info(f"Authentication successful for user_id: {session.user_id}")
            
            request.state.user = session.user
            return credentials
            
        except jwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise ValueError("Token expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token format: {str(e)}")
            raise ValueError(f"Invalid token format: {str(e)}")
            
    except ValueError as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )