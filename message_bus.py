# message_bus.py
# ==============================================================================
# Phase 3: Inter-Agent Message Bus
# Enables agents to communicate, coordinate, and share intelligence
# Redis-backed with in-memory fallback
# ==============================================================================

import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable
from collections import defaultdict

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


class Message:
    """A message passed between agents."""

    def __init__(self, sender: str, recipient: str, msg_type: str,
                 content: dict, priority: str = 'normal'):
        self.sender = sender
        self.recipient = recipient  # '*' = broadcast
        self.msg_type = msg_type
        self.content = content
        self.priority = priority
        self.timestamp = datetime.now().isoformat()
        self.id = f"{sender}_{int(time.time() * 1000)}"

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'sender': self.sender,
            'recipient': self.recipient,
            'type': self.msg_type,
            'content': self.content,
            'priority': self.priority,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        msg = cls(
            sender=data['sender'],
            recipient=data['recipient'],
            msg_type=data['type'],
            content=data['content'],
            priority=data.get('priority', 'normal')
        )
        msg.id = data.get('id', msg.id)
        msg.timestamp = data.get('timestamp', msg.timestamp)
        return msg

    def __repr__(self):
        return f"Message({self.sender}→{self.recipient}: {self.msg_type})"


class MessageBus:
    """Inter-agent communication bus.
    
    Message Types:
    - 'inventory_alert': Warehouse warns of low stock
    - 'disruption_alert': Any agent reports a disruption
    - 'demand_forecast': Demand agent shares predictions
    - 'shipment_update': Logistics reports shipment status
    - 'order_request': Warehouse requests supply from supplier
    - 'order_confirmation': Supplier confirms order
    - 'recovery_notice': Agent reports recovery from disruption
    - 'coordination': General coordination messages
    """

    _instance = None  # Singleton

    def __init__(self, redis_url: str = None):
        self.redis_client = None
        self.message_log = []
        self.subscribers = defaultdict(list)  # msg_type -> [callbacks]
        self.inbox = defaultdict(list)  # agent_name -> [messages]
        self._total_messages = 0

        if redis_url and HAS_REDIS:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                print(f"Message Bus connected to Redis: {redis_url}")
            except Exception as e:
                print(f"Redis unavailable: {e}. Using in-memory message bus.")
                self.redis_client = None
        else:
            if not HAS_REDIS:
                pass  # Silent — in-memory is fine for Phase 3

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None

    @property
    def using_redis(self):
        return self.redis_client is not None

    def publish(self, message: Message):
        """Publish a message to the bus.
        
        Messages are delivered to:
        1. The specific recipient's inbox
        2. All subscribers of the message type
        3. Broadcast ('*') goes to all inboxes
        """
        self._total_messages += 1
        self.message_log.append(message.to_dict())

        if self.using_redis:
            try:
                channel = f"supply_chain:{message.msg_type}"
                self.redis_client.publish(channel, json.dumps(message.to_dict()))
                # Also store in list for retrieval
                self.redis_client.lpush(
                    f"inbox:{message.recipient}",
                    json.dumps(message.to_dict())
                )
                self.redis_client.ltrim(f"inbox:{message.recipient}", 0, 99)
            except Exception:
                pass

        # In-memory delivery
        if message.recipient == '*':
            # Broadcast — deliver to all known inboxes
            for agent_name in list(self.inbox.keys()):
                if agent_name != message.sender:
                    self.inbox[agent_name].append(message)
        else:
            self.inbox[message.recipient].append(message)

        # Trigger subscribers
        for callback in self.subscribers.get(message.msg_type, []):
            try:
                callback(message)
            except Exception:
                pass

    def subscribe(self, msg_type: str, callback: Callable):
        """Subscribe to a message type."""
        self.subscribers[msg_type].append(callback)

    def get_messages(self, agent_name: str, msg_type: str = None,
                     limit: int = 10) -> List[Message]:
        """Get messages from an agent's inbox.
        
        Args:
            agent_name: agent to get messages for
            msg_type: optional filter by message type
            limit: max messages to return
        """
        messages = self.inbox.get(agent_name, [])

        if msg_type:
            messages = [m for m in messages if m.msg_type == msg_type]

        return messages[-limit:]

    def get_latest(self, agent_name: str, msg_type: str) -> Optional[Message]:
        """Get the most recent message of a specific type for an agent."""
        messages = self.get_messages(agent_name, msg_type, limit=1)
        return messages[-1] if messages else None

    def clear_inbox(self, agent_name: str):
        """Clear an agent's inbox."""
        self.inbox[agent_name] = []

    def broadcast_alert(self, sender: str, alert_type: str, details: dict):
        """Convenience: broadcast an alert to all agents."""
        msg = Message(
            sender=sender,
            recipient='*',
            msg_type=alert_type,
            content=details,
            priority='high'
        )
        self.publish(msg)

    def send_direct(self, sender: str, recipient: str,
                    msg_type: str, content: dict, priority: str = 'normal'):
        """Convenience: send a direct message."""
        msg = Message(sender, recipient, msg_type, content, priority)
        self.publish(msg)

    def get_message_log(self, n: int = 20) -> List[dict]:
        """Get recent message log."""
        return self.message_log[-n:]

    def get_stats(self) -> dict:
        """Get message bus statistics."""
        type_counts = defaultdict(int)
        for msg in self.message_log:
            type_counts[msg['type']] += 1

        return {
            'total_messages': self._total_messages,
            'using_redis': self.using_redis,
            'active_inboxes': len(self.inbox),
            'message_types': dict(type_counts),
            'subscribers': {t: len(cbs) for t, cbs in self.subscribers.items()}
        }

    def format_for_prompt(self, agent_name: str, limit: int = 5) -> str:
        """Format recent messages for LLM prompt injection.
        
        Agents can see relevant messages from other agents to
        coordinate their decisions.
        """
        messages = self.get_messages(agent_name, limit=limit)
        if not messages:
            return "No recent messages from other agents."

        lines = ["Recent inter-agent communications:"]
        for msg in messages[-limit:]:
            lines.append(
                f"- [{msg.priority.upper()}] {msg.sender} → {msg.msg_type}: "
                f"{json.dumps(msg.content)}"
            )
        return "\n".join(lines)
