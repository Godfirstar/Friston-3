"""Small discourse rules; they resolve conversation, not lore truth."""
# re 是 Python 标准库正则表达式模块；本文件只处理字符串，不调用 LLM/数据库/设备。
import re


# frozenset 是不可变集合；外层 (...) 传入多条短语。用 q in EXPLANATIONS 检查整句是否命中。
# skills/npc.py 和 npc/prompts.py 也读取它，用来识别“继续解释上轮”的简短追问。
EXPLANATIONS = frozenset((
    "为什么", "为什么呢", "为什么这么觉得", "为什么这样觉得", "为什么这样说",
    "为什么这么说", "为什么喜欢她", "为什么信任她", "怎么说", "怎么说呢",
    "具体说说", "具体说说吧", "继续", "接着说", "然后呢",
))


def bare_text(text):
    # 调用方：本文件各函数、NpcTalkSkill、PromptBuilder；输入 str，返回去指定空白/标点的 str。
    # re.sub(pattern,replacement,text) 替换所有匹配；r 是原始字符串，\s 匹配空白。
    # [...] 是字符类：其中任一字符匹配就替换成空串；返回新字符串，不改原 text。
    return re.sub(r"[\s，,。.!！?？~～]", "", text)


def conversation_act(text):
    """A bounded style hint, never a factuality or retrieval exemption.

    Mixed questions stay general so a greeting cannot hide a lore request.
    This is intentionally a small heuristic, not a semantic intent classifier.
    """
    # 调用方：NpcTalkSkill、PromptBuilder、persona、response_guard、knowledge/retrieve_lore。
    # 输入一句话，返回 greeting/emotion/.../general 这样的风格标签 str，不决定最终检索路由。
    # raw 首次定义：把 None 等空值替成 ''，转 str 并去首尾空白；原参数不变。
    raw = str(text or '').strip()
    # q 是本地规范化文本；casefold 是字符串方法，适合忽略英文大小写比较，中文不受影响。
    q = bare_text(raw).casefold()
    # statement 是 bool：re.search 查到问题/请求线索返回 Match，查不到返回 None；not 取反。
    # 后面的规则会用它区分普通陈述与混合请求，避免一句“谢谢，帮我拍照”只被当成感谢。
    statement = not re.search(
        r'什么|为什么|怎么|如何|是谁|哪(?:个|位|里)|是不是|是否|能否|介绍|解释|'
        r'想(?:知道|了解|问|听|聊)|谈谈|讲讲|请|帮我|拍照|拍张|拍摄|倒计时|打开|关闭|执行|运行|[?？]', raw)
    # fullmatch 要求整句匹配；(?:...) 是不捕获内容的分组，| 表示任选，? 表示可出现零或一次。
    # 以下 if 按书写顺序检查；某条 return 后，后面规则不会继续执行。
    if re.fullmatch(r'(?:你好|您好|嗨|哈喽|哈罗|早安|早上好|上午好|下午好|晚上好|晚安|早|hello|hi|hey)(?:呀|啊|哇|哦)?(?:friston(?:-?3)?|弗里斯顿)?', q):
        return 'greeting'
    # 感谢允许整句模板，或短陈述以感谢开头；len(q) 算字符数，不是模型 token 数。
    # and 优先级高于 or：这里第二种情况必须同时满足 statement/长度/前缀匹配。
    if (re.fullmatch(r'(?:谢谢(?:你|您|啦|了)?|多谢(?:你|您)?|感谢你|辛苦了|thankyou|thanks)(?:啊|呀|哦|呢)?', q)
            or statement and len(q) <= 50 and re.match(r'^(?:谢谢|多谢|感谢你|辛苦了)', q)):
        return 'thanks'
    # 正则中的 * 表示前一分组可重复零次或多次，和 Python 函数参数中的 * 不是一回事。
    if re.fullmatch(r'(?:嗯|嗯嗯|好|好的|哦|噢|啊|原来如此)*(?:我)?(?:明白了|知道了|了解了|懂了)(?:呀|啊|呢|谢谢)?', q):
        return 'acknowledgement'
    # Match a complete personal report, not an emotion word anywhere in a
    # short sentence. A named character's feelings remain a lore question.
    # time/personal/degree/feeling 都是“正则片段字符串”，不是时间对象或识别结果。
    # + 把字符串拼成更大规则；这里 time 是局部变量，与标准库 time 模块无关。
    time = r'今天|最近|现在|刚刚|这几天|这会儿|此刻'
    personal = r'(?:(?:我(?:' + time + r')?|(?:' + time + r')(?:我)?))?'
    degree = r'(?:(?:真的|实在|感觉|觉得|有点|有些|有一点|有点儿|好|很|挺|太|特别|非常))*'
    feeling = r'(?:累|困|疲惫|疲倦|难过|孤独|害怕|失落|伤心|焦虑|烦躁|不开心|开心|高兴|不好|糟糕|很差)'
    # re.match 从开头匹配，^/$ 明确句首/句尾；此分支要求个人或时间开头、情绪词结尾。
    if (statement and len(q) <= 65
            and re.match(r'^(?:我|' + time + r')', q)
            and re.search(degree + feeling + r'(?:了|呢|啊|呀|唉|嘛)*$', q)):
        return 'emotion'
    # support 是可选的休息/陪聊后缀；括号内相邻字符串自动拼接为同一个 str。
    support = (r'(?:(?:只)?想(?:休息|歇|睡)(?:一下|一会儿?|会儿?)?(?:了)?|'
               r'(?:你)?(?:能|可以)?(?:陪我|和我)(?:聊聊|聊一会儿?|说说话|说两句)(?:吗|吧)?)?')
    # 以下两个整句模板分别处理普通情绪自述、逛展/排队后的情绪；允许模板指定的省略成分。
    if len(q) <= 65 and re.fullmatch(
            personal + r'(?:心情)?' + degree + feeling + r'(?:了|的|呢|啊|呀|唉|嘛)*' + support, q):
        return 'emotion'
    if len(q) <= 65 and re.fullmatch(
            personal + r'(?:逛(?:了)?(?:漫)?展|排(?:了)?队|走(?:了)?(?:好多路|一天|好久)?|站(?:了)?(?:一天|好久)?)'
            + r'(?:走|逛|排|站)?(?:得)?' + degree + feeling
            + r'(?:了|的|呢|啊|呀|唉|嘛)*' + support, q):
        return 'emotion'
    # Keep explicit information requests out of social-only routing.
    # 仍有明显信息请求时回到 general，让后续规划负责，不把它压成纯社交回复。
    if re.search(r'什么|为什么|怎么|如何|是谁|哪(?:个|位|里)|介绍|解释|知道|认识|[?？]', raw):
        return 'general'
    # 赞美分支同时检查对象、正面表达和否定；多项 and 要全部成立。
    if (len(q) <= 45 and re.search(r'你|这(?:个|副|台)|设计|造型|回答', q)
            and re.search(r'不错|很好|很棒|真棒|真厉害|真好|做得好|很厉害|漂亮|喜欢你|夸你|很有帮助|有意思|有趣', q)
            and not re.search(r'不(?:错才怪|太|怎么|喜欢)|一点也不|并不|谈不上', q)):
        return 'compliment'
    # personal_report 要有第一人称和时间开头；前面均未命中则最终返回兜底标签 general。
    if (statement and len(q) <= 100 and re.match(
            r'^(?:(?:我|我们|咱们)(?:今天|刚才|最近|现在)|(?:今天|刚才|最近|现在)(?:我|我们|咱们))', q)):
        return 'personal_report'
    return 'general'


def reference_description(text, subjects):
    """An unnamed nickname/appearance is not the last character's identity."""
    # 调用方 NpcTalkSkill._execute；text 是问题 str，subjects 是实体字典 list。
    # 返回待澄清描述 str；空串表示本规则没有发现需追问的外观/昵称指代。
    # any(生成器) 遇到第一个真值就返回 True；s.get 是内置 dict 方法，缺键返回 None。
    if any(s.get("kind") in ("character", "machine", "fan_machine") for s in subjects):
        # The speaker's own node does not resolve an unnamed other person.
        from friston3.npc.identity import IDENTITY
        # IDENTITY 来自 identity.py；自己的实体节点不能替别的未指明人物消除歧义。
        if any(s.get('id') != IDENTITY.entity_id for s in subjects):
            return ""
    if '兜帽男' in text:
        return '兜帽男'
    # match 是 re.Match 或 None；([^...]{1,14}) 捕获一段非句末标点的 1~14 字描述。
    match = re.search(r"(?:有(?:一)?(?:只|个|位)|那(?:只|个|位))([^，。！？?]{1,14})", text)
    if not match:
        return ""
    # Match.group(1) 取第一个捕获组；re.sub 去掉描述末尾的疑问语气，得到 description 字符串。
    description = re.sub(r"(?:是不是|是谁|叫什么|吗|吧|呢|呀|啊)+$", "", match.group(1))
    # word in description 是“子串是否出现”；这里不是字典查键，也不是向量语义判断。
    if any(word in description for word in ("兔", "猫", "狐狸", "耳朵", "尾巴", "头发", "家伙", "姑娘")):
        return description
    return ""


def discourse_act(text, subjects, previous_question, pending_reference=None):
    """Only a short explicit name can settle a pending referent/correction."""
    # 调用方 NpcTalkSkill 的知识任务/最终回答流程；返回 explanation/correction/question 标签。
    # previous_question 是上一条问题字符串；pending_reference 是尚待澄清的字典或 None。
    q = bare_text(text)
    if q in EXPLANATIONS:
        return "explanation"
    # 列表推导式只保留人物/机器实体的 name 字符串；names 与原 subjects 字典列表不同。
    names = [s["name"] for s in subjects if s.get("kind") in ("character", "machine")]
    if previous_question and names:
        # re.escape 让名字中的正则特殊符号按普通文字匹配；'|'.join 把多个名字拼成或分支。
        name_pattern = "(?:" + "|".join(re.escape(name) for name in names) + ")"
        if re.fullmatch(r"(?:不对|不是|应该是|是|我说的是|我指的是)?" + name_pattern + r"(?:吧|才对|啊|呀|对吧|不是吗)?", q):
            # startswith/endswith 是 str 方法，传 tuple 表示任一前/后缀匹配即可，不是多次函数调用。
            if pending_reference or q.endswith(("吧", "才对", "对吧", "不是吗")) or q.startswith(("不对", "应该是", "我指的是", "我说的是")):
                return "correction"
    return "question"
