"""Explicit local memories with provenance; no LLM-based fact extraction.

Persisted dialogue is an audit log only. Assistant messages never enter recall
evidence. NpcTalkSkill separately carries bounded live-session dialogue for
discourse continuity; persisted model replies are not reloaded as facts.
User reports and actual skill results live in separate tables and remain labelled.
"""
# 这些导入都来自 Python 标准库：closing 管关闭、deque 管定长队列、datetime 管时间、
# Path 管路径、re 管正则、sqlite3 管本地数据库、uuid4 生成随机标识；这里没有深度学习模型。
from contextlib import closing
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

# PROJECT_ROOT 是 config.py 的项目根目录 Path；默认记忆库位于它下面的 data/runtime。
from friston3.config import PROJECT_ROOT


class MemoryStore:
    # NpcTalkSkill.__init__ 创建主人长期记忆对象；构造会创建目录/打开数据库，并非纯数据赋值。
    # 本类与下方 SessionMemory 提供同名方法，让 NPC 按当前说话者选择不同的存储对象。
    def __init__(self, path=None):
        # self 是 MemoryStore 实例；path 为可选路径，None 时用默认位置。
        # Path(path) 创建路径对象；路径上的 / 运算符是拼接路径，不是数值除法。
        self.path = Path(path) if path else PROJECT_ROOT / "data" / "runtime" / "npc_memory.db"
        # .parent 得到上一级目录；mkdir 为 pathlib.Path 方法，允许建父目录和已存在目录。
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # uuid4() 返回 UUID 对象，.hex 读取其十六进制字符串，用来标记本次对象记录的对话。
        self.session_id = uuid4().hex
        # sqlite3.connect 返回数据库连接；str(self.path) 把 Path 转为连接函数接受的路径文本。
        # with closing(...) as db 保证离开时关闭连接；后面的 , db 再进入连接的事务上下文。
        # 连接上下文在正常结束时提交事务、异常时回滚；它自身不会关闭连接，故需要 closing。
        with closing(sqlite3.connect(str(self.path))) as db, db:
            # executescript 执行多条 SQL；三引号内容是 SQL 字符串，不是 Python 代码。
            # 三张表分别保存对话审计、用户明确自述、技能实际结果，避免把模型回复召回成事实。
            # IF NOT EXISTS 不覆盖已有表；INTEGER PRIMARY KEY 提供记录ID，UNIQUE 防止自述重复。
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY, statement TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL DEFAULT 'user_report', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_events (
                    id INTEGER PRIMARY KEY, skill TEXT NOT NULL, success INTEGER NOT NULL,
                    message TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'skill_result',
                    created_at TEXT NOT NULL
                );
            """)

    @staticmethod
    def _now():
        # 本类写入函数和 SessionMemory.record_execution 调用；无参数，返回 UTC 时间 str。
        # @staticmethod 是内置装饰器，不自动传 self；datetime.now 来自标准库 datetime 类。
        # timezone.utc 明确时区，.isoformat 生成可存入 SQLite 的文本，timespec 精确到秒。
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def log_turn(self, user_text, reply):
        # 调用方：NpcTalkSkill._remember；输入用户原文/助手回复 str，写审计记录，隐式返回 None。
        # An audit write must not hold a visitor's reply behind another writer.
        # timeout=0.1 是等待数据库锁的秒数；并非整轮模型回答的超时。
        with closing(sqlite3.connect(str(self.path), timeout=0.1)) as db, db:
            # executemany 对每组参数执行同一 INSERT；? 是 SQL 参数占位符，值作为数据绑定。
            # 不把用户文本拼进 SQL，因此引号等特殊字符不被解释为 SQL 指令。
            # 列表推导式循环两组 (role,content)，拼出 user/assistant 两条记录的参数 tuple。
            db.executemany(
                "INSERT INTO conversations(session_id,role,content,created_at) VALUES(?,?,?,?)",
                [(self.session_id, role, content, self._now())
                 for role, content in (("user", user_text), ("assistant", reply))],
            )
            # Bounded audit history. Explicit memories are never silently evicted.
            # 子查询按 id 倒序保留最近2000条审计，其余删除；这里没有删 user_memories 表。
            db.execute("DELETE FROM conversations WHERE id NOT IN "
                       "(SELECT id FROM conversations ORDER BY id DESC LIMIT 2000)")

    def remember(self, statement):
        # 调用方：NpcTalkSkill._memory_command 的“记住：...”分支，权限已由调用方检查。
        # statement 是用户明确要求保存的 str；写入后隐式返回 None，不调用 LLM 提取新事实。
        with closing(sqlite3.connect(str(self.path))) as db, db:
            # 同一句 statement 触发 UNIQUE 冲突时只更新时间；excluded 指本次拟插入记录。
            # SQL 与参数元组分开传入，user_report 标记“用户自述”，不升级为外部事实证明。
            db.execute("INSERT INTO user_memories(statement,source,created_at) VALUES(?,?,?) "
                       "ON CONFLICT(statement) DO UPDATE SET created_at=excluded.created_at",
                       (statement, "user_report", self._now()))

    def forget(self, query):
        """Delete literal substring matches, including their audit-log copies."""
        # 调用方：NPC 的“忘记：...”分支；query 是子串 str，返回删除长期自述的条数 int。
        with closing(sqlite3.connect(str(self.path))) as db, db:
            # SQLite instr(文本,子串)>0 表示包含匹配；(query,) 是单元素 tuple，逗号不能省。
            # execute 返回 Cursor；.rowcount 读取这条 DELETE 影响行数，保存到局部变量 count。
            count = db.execute("DELETE FROM user_memories WHERE instr(statement,?)>0", (query,)).rowcount
            # 同时删掉审计表中的匹配内容，但返回 count 不包括这一张表删除的条数。
            db.execute("DELETE FROM conversations WHERE instr(content,?)>0", (query,))
            return count

    def record_execution(self, skill_name, result):
        # main 对拍照结果调用 NpcTalkSkill.record_execution，再转发到当前 memory 的本方法。
        # skill_name 为技能名 str，result 为技能返回的 dict；无记录/成功写入都隐式返回 None。
        if skill_name not in ("take_photo", "auto_photo"):
            # return 不带值等价于返回 None；未纳入的技能不会写成已执行经历。
            return
        # isinstance 是内置类型检查；or 短路，非字典时不会继续访问 result.get。
        if not isinstance(result, dict) or not isinstance(result.get("success"), bool):
            return
        # message 是结果里的说明 str，截取前500字符；get 缺键给空字符串，避免 KeyError。
        message = str(result.get("message", ""))[:500]
        with closing(sqlite3.connect(str(self.path))) as db, db:
            # int(True)=1 / int(False)=0，以 SQLite INTEGER 保存；source 固定为 skill_result。
            db.execute("INSERT INTO execution_events(skill,success,message,source,created_at) "
                       "VALUES(?,?,?,?,?)", (skill_name, int(result["success"]), message,
                                            "skill_result", self._now()))

    def recent_execution_events(self, limit=2):
        """Camera outcomes prove neither encounters nor travel."""
        # 调用方：NpcTalkSkill._self_history_reply；limit 为期望条数，返回最新在前的 list[dict]。
        with closing(sqlite3.connect(str(self.path))) as db:
            # row_factory 改变后续读取行的表示：sqlite3.Row 支持按列名取值，再转成普通 dict。
            db.row_factory = sqlite3.Row
            # int/min/max 把请求条数规范到1~10；LIMIT ? 通过参数绑定接收这个整数。
            # SELECT 只读真实拍照结果，ORDER BY id DESC 最新在前；列表推导式逐行复制为字典。
            return [dict(row) for row in db.execute(
                "SELECT id,skill,success,message,source,created_at FROM execution_events "
                "WHERE source='skill_result' AND skill IN ('take_photo','auto_photo') "
                "ORDER BY id DESC LIMIT ?", (max(1, min(10, int(limit))),))]

    @staticmethod
    def _terms(text):
        # 本类 recall 调用；输入字符串，返回关键词 set，用字符重合召回，不是 embedding。
        # re.findall 返回所有匹配字符串 list；text.lower 是 str 方法，统一英文大小写。
        # r 原始字符串避免 Python 先解释反斜杠；{2,} 要求英文数字片段至少两个字符。
        terms = set(re.findall(r"[a-zA-Z0-9_]{2,}", text.lower()))
        # part 是匹配到的连续中文字符串；for 依次处理每一段中文，不把整段当一个英文词。
        for part in re.findall(r"[\u3400-\u9fff]+", text):
            # range(len(part)-1) 枚举相邻双字起点；part[i:i+2] 是长度最多2的字符串切片。
            # set.update 把生成器产出的所有双字加入集合，自动去重；它不是覆盖整个 terms。
            terms.update(part[i:i + 2] for i in range(len(part) - 1))
        # 集合差集删除常见记忆命令词，避免每条记忆都因“记住/我的”等词获得分数。
        return terms - {"记得", "记住", "什么", "一下", "之前", "告诉", "我的", "我是", "你还"}

    def recall(self, user_text, limit=3, *, include_execution=True):
        # 调用方：NpcTalkSkill.prepare_route_context / _execute；当前这两处都传 include_execution=False。
        # user_text 为当前问题 str，limit 为最多自述条数；返回拼接后的提示词 str，没匹配则空串。
        # * 后参数必须按名称传；include_execution 只控制附加事件，不改变自述召回。
        # terms 是前面 _terms 返回的 set；general 是 any 计算出的 bool。
        terms = self._terms(user_text)
        # any 对生成器逐项判断，只要用户句子包含一个“回顾所有记忆”的短语就停止并返回 True。
        general = any(word in user_text for word in ("记得我", "关于我的", "记住了什么", "我说过什么"))
        with closing(sqlite3.connect(str(self.path), timeout=0.1)) as db:
            # .fetchall 把查询结果取成 list[tuple]；每行恰好是 (statement,created_at)。
            # 按 id 倒序取最近500条自述，而不是读取 conversations 中的助手回复。
            rows = db.execute("SELECT statement,created_at FROM user_memories ORDER BY id DESC LIMIT 500").fetchall()
            # enumerate 给每行附上从0开始的 index；嵌套解包把一行拆成文本和时间戳。
            # & 是集合交集，len 是共用词数；scored 每项为 (分数,新近顺序,文本,时间)。
            scored = [(len(terms & self._terms(statement)), index, statement, timestamp)
                      for index, (statement, timestamp) in enumerate(rows)]
            # 保留分数非0的候选，或者 general=True 时保留所有；sorted 返回新的排序列表。
            # lambda 是匿名函数，key 返回 (-分数,index)：先高分，再同分中较新的记录。
            # [:limit] 是列表切片，取前 limit 项；selected 是筛选后的 tuple 列表。
            selected = sorted((item for item in scored if item[0] or general),
                              key=lambda item: (-item[0], item[1]))[:limit]
            # lines 是字符串列表；两个 _ 表示分数/序号在这里不需要，仍是普通解包变量。
            # stamp[:10] 取日期部分，statement[:180] 限单条长度；f 字符串插入这些变量值。
            lines = [f"[user_report，用户自述，{stamp[:10]}] {statement[:180]}"
                     for _, _, statement, stamp in selected]
            if include_execution and any(word in user_text for word in ("上次", "刚才", "拍过", "拍了", "做过", "经历", "回忆")):
                # 只有显式允许且问到经历时，才额外查最近两条执行结果；events 为行 tuple 列表。
                events = db.execute("SELECT skill,success,message,created_at FROM execution_events "
                                    "ORDER BY id DESC LIMIT 2").fetchall()
                # event_lines 为另一组提示词字符串；条件表达式把数据库0/1转成失败/成功文字。
                event_lines = [f"[skill_result，真实执行，{stamp}] {skill}："
                               f"{'成功' if success else '失败'}，{message[:140]}"
                               for skill, success, message, stamp in events]
                # Actual outcomes must survive the small prompt budget first.
                lines = event_lines + lines
        # str.join 用换行符连接列表；返回一段文本给 PromptBuilder，不返回数据库行对象。
        return "\n".join(lines)

    def clear_all(self):
        # 调用方：NpcTalkSkill.clear_all_memory；返回 {表名:删除行数} dict，权限在调用方处理。
        counts = {}
        with closing(sqlite3.connect(str(self.path))) as db, db:
            # table 只来自代码内固定元组，可用于 SQL 标识符拼接；不是用户输入。
            # SQL 的 ? 参数只能绑定值，不能替换表名；这里的 f字符串用途与上面的参数绑定不同。
            for table in ("conversations", "user_memories", "execution_events"):
                counts[table] = db.execute(f"DELETE FROM {table}").rowcount
        return counts


class SessionMemory:
    """A visitor's execution facts last only for their current interaction.

    No database is opened, no utterance is audited, and no owner's explicit
    notes are readable through this object. Conversation continuity remains in
    NpcTalkSkill's bounded deques. Handover discards this object.
    """
    def __init__(self):
        # ConversationState 的 default_factory 创建本类，访客切换时由 NpcTalkSkill 换新会话。
        # deque 来自 collections；maxlen=10 表示最多保留10项，append 满后自动丢最旧项。
        # self.events 是内存队列，不创建 SQLite 文件；self 是这次访客记忆实例。
        self.events = deque(maxlen=10)

    def log_turn(self, user_text, reply):
        # 与 MemoryStore 同名同参数，供 NPC._remember 统一调用；访客文本不落盘，直接返回 None。
        return None

    def record_execution(self, skill_name, result):
        # 调用链与 MemoryStore.record_execution 相同，接收真实拍照技能结果，保存到当前实例队列。
        # or 短路检查名称、字典、布尔 success；不满足约定时返回 None，不写入事件。
        if (skill_name not in ("take_photo", "auto_photo") or not isinstance(result, dict)
                or not isinstance(result.get("success"), bool)):
            return
        # append 是 deque 方法，加入一个新 dict；MemoryStore._now 是上方静态时间函数。
        # id 用当前队列长度+1，仅为本地记录标签；队列满后长度固定，不能把它当全局唯一ID。
        self.events.append({
            "id": len(self.events) + 1, "skill": skill_name,
            "success": int(result["success"]), "message": str(result.get("message", ""))[:500],
            "source": "skill_result", "created_at": MemoryStore._now(),
        })

    def recent_execution_events(self, limit=2):
        # NPC._self_history_reply 和下方 recall 调用；返回最新在前的 list[dict]。
        # 先 list(self.events) 复制队列，再 [-n:] 取最后n项，最后 [::-1] 倒序。
        # dict(event) 给每条记录新建外层字典，调用方修改返回字段不会直接改队列里的字典。
        return [dict(event) for event in list(self.events)[-max(1, min(10, int(limit))):][::-1]]

    def recall(self, user_text, limit=3, *, include_execution=True):
        # 与长期记忆接口一致；返回本次访客的执行说明 str，不访问主人自述表。
        # limit 参数保留接口兼容，目前本实现固定读2条事件，没有用它决定数量。
        if not include_execution:
            return ""
        if not any(word in user_text for word in ("上次", "刚才", "拍过", "拍了", "做过", "经历", "回忆")):
            return ""
        # join 消费生成器中的字符串；event 是 recent_execution_events(2) 的每个结果 dict。
        # 没有事件时 join 空序列得到空串，不会凭空生成“做过某件事”的记录。
        return "\n".join(
            f"[skill_result，本次访客会话真实执行，{event['created_at']}] {event['skill']}："
            f"{'成功' if event['success'] else '失败'}，{event['message'][:140]}"
            for event in self.recent_execution_events(2))

    def remember(self, statement):
        # 保留同名接口但明确不支持访客持久写入；raise 抛 RuntimeError，由上层决定如何报告。
        raise RuntimeError("访客会话不写入主人的长期记忆")

    def forget(self, query):
        # 同理不允许访客对象删除主人记忆；本函数没有打开数据库或执行 DELETE。
        raise RuntimeError("访客会话不能删除主人的长期记忆")

    def clear_all(self):
        # 接口返回与 MemoryStore 同形状的计数 dict；这里只清本实例事件，不删除文件。
        # 先 len 保存原有条数，再 deque.clear 清空；清空后再 len 就无法报告删除了多少条。
        count = len(self.events)
        self.events.clear()
        return {"conversations": 0, "user_memories": 0, "execution_events": count}
