"""一个说话者的历史、话题、澄清和记忆；切换访客时整体替换。"""
from collections import deque
from dataclasses import dataclass, field

from friston3.config import NPC_HISTORY_TURNS
from friston3.npc.memory import SessionMemory


@dataclass
class ConversationState:
    memory: object = field(default_factory=SessionMemory)
    history: deque = field(default_factory=lambda: deque(maxlen=2 * NPC_HISTORY_TURNS))
    history_topics: deque = field(default_factory=lambda: deque(maxlen=NPC_HISTORY_TURNS))
    topic: dict = field(default_factory=dict)
    pending_reference: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    turn_number: int = 0
    discourse_topic: dict = field(default_factory=dict)
    discourse_turn: int = 0
