# app/services/conversation.py
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy.orm import Session as DBSession
from app.models.database import Conversation, Message, User
import logging

logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, db_session: DBSession, redis_client=None):
        self.db = db_session
        self.redis = redis_client
        self.max_context_messages = 10
        self.cache_ttl = 3600  # 1 hour
        self._test_redis_connection()

    def _test_redis_connection(self):
        """Test Redis connection and disable if not available"""
        if self.redis:
            try:
                self.redis.ping()
            except Exception as e:
                logger.warning(f"Redis not available, continuing without caching: {str(e)}")
                self.redis = None

    def create_conversation(self, user: User) -> str:
        conversation = Conversation(
            user_id=user.id,
            meta_data={}
        )
        self.db.add(conversation)
        self.db.commit()
        return conversation.id

    def add_message(self, conversation_id: str, role: str, content: str, user: User) -> None:
        conversation = self._get_conversation(conversation_id, user)
        if not conversation:
            raise ValueError("Conversation not found")

        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )
        self.db.add(message)
        self.db.commit()

        if self.redis:
            try:
                self.redis.delete(f"conv:{conversation_id}:context")
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis cache: {str(e)}")

    def get_conversation_context(self, conversation_id: str, user: User) -> List[Dict]:
        # Try cache first if Redis is available
        if self.redis:
            try:
                cached = self.redis.get(f"conv:{conversation_id}:context")
                if cached:
                    return cached
            except Exception as e:
                logger.warning(f"Failed to get from Redis cache: {str(e)}")

        conversation = self._get_conversation(conversation_id, user)
        if not conversation:
            return []

        # Get recent messages
        recent_messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp.desc())
            .limit(self.max_context_messages)
            .all()
        )

        context = [
            {"role": msg.role, "content": msg.content}
            for msg in reversed(recent_messages)
        ]

        # Try to cache the context, ignore if Redis is not available
        if self.redis:
            try:
                self.redis.setex(
                    f"conv:{conversation_id}:context",
                    self.cache_ttl,
                    context
                )
            except Exception as e:
                logger.warning(f"Failed to set Redis cache: {str(e)}")

        return context

    def _get_conversation(self, conversation_id: str, user: User) -> Optional[Conversation]:
        return (
            self.db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id
            )
            .first()
        )