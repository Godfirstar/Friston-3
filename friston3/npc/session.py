"""一个说话者的历史、话题、澄清和记忆；切换访客时整体替换。"""
from collections import deque
from dataclasses import dataclass, field

from friston3.config import NPC_HISTORY_TURNS
from friston3.npc.memory import SessionMemory


# 由 skills/npc.py 的构造器、set_interlocutor 和 clear_history 创建。
# dataclass 是上面从标准库导入的装饰器：根据字段生成构造方法等样板代码。
# 例如 ConversationState(memory=x) 会把 x 保存为新对象的 .memory 属性。
@dataclass
class ConversationState:
    # “字段名: 类型”是类型提示；“= field(...)”进一步说明这个字段怎样取默认值。
    # field 和 dataclass 都来自 dataclasses；SessionMemory 则是本项目 memory.py 的类。
    # 默认是访客临时记忆；NpcTalkSkill.set_interlocutor 可传入主人 MemoryStore。
    # 这里传 SessionMemory（类本身）而不是 SessionMemory()：每创建会话时才新建记忆。
    memory: object = field(default_factory=SessionMemory)
    # 一轮有 user/assistant 两条，因此消息数是轮数的两倍；满后 deque 自动丢最旧项。
    # deque 是 collections 的双端队列；lambda 是一个无参数小函数。
    # 调用这个 lambda 才得到新的队列，所以两个访客不会意外共享同一个 history。
    history: deque = field(default_factory=lambda: deque(maxlen=2 * NPC_HISTORY_TURNS))
    # history_topics 与消息历史分开：保存各轮主体/话题的标识集合，供提示词组织时匹配历史话题。
    history_topics: deque = field(default_factory=lambda: deque(maxlen=NPC_HISTORY_TURNS))
    # dict 是 Python 内置字典类型；default_factory=dict 每次产生新的 {}。
    topic: dict = field(default_factory=dict)  # 当前主体、问题及实际使用的 sources。
    # pending_reference 保存尚待用户澄清的指代；trace 保存当前一轮的诊断字典。
    pending_reference: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    # int/0 是不可变整数，直接用 0 作默认值即可；execute 每开始一轮把它加 1。
    turn_number: int = 0
    # 当本轮没有新的证据话题时，NpcTalkSkill._recent_topic 可在有限轮次内回看它。
    discourse_topic: dict = field(default_factory=dict)
    discourse_turn: int = 0
