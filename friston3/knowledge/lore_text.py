"""Shared, dependency-free text utilities for the lore wiki graph."""
import re
import unicodedata


# 本文件的函数由导入器、实体解析器、检索器和回复核对共用；只处理传入文字，不访问模型或数据库。
def normalize(text):
    # 返回便于比较的字符串。unicodedata 是标准库模块，NFKC 统一全角等兼容字符形式。
    # str(text or '') 处理空值；casefold 做不区分大小写的折叠，不等于中文语义理解。
    value = unicodedata.normalize('NFKC', str(text or '')).casefold()
    # re 是标准库正则模块；sub(模式, 替换文字, 原文) 把匹配的空白/标点替换为空串。
    return re.sub(r'[\s\-_.·•:：,，。？！!?()（）“”‘’\[\]<>]+', '', value)


def tokenize(text):
    # import_lore_wiki._passage 写 FTS 索引、lore_search 构造查询时使用，返回空格分隔的词串。
    # tokens 是本函数新建的列表；这里用中文相邻二字片段，不是深度学习 tokenizer 的 token ID。
    tokens = []
    # findall 返回所有匹配串；正则中的 | 表示“中文连续串或英文/数字连续串”。
    for term in re.findall(r'[\u3400-\u9fff]+|[a-z0-9_]+', unicodedata.normalize('NFKC', text).casefold()):
        if '\u3400' <= term[0] <= '\u9fff':
            # term[i:i+2] 取从 i 开始的两个字；extend 消费生成器，把各片段依次加入列表。
            tokens.extend(term[i:i+2] for i in range(len(term)-1))
        elif len(term) >= 2:
            tokens.append(term)
    return ' '.join(tokens)


def phrase_query(text):
    # lore_search 调用；split 取词列表，dict.fromkeys 保序去重，join 组成 FTS 的 AND 查询。
    # 每个词先转义内部双引号再加引号；返回的是查询字符串，不会在这里执行 SQL。
    return ' AND '.join('"' + word.replace('"', '""') + '"'
                        for word in dict.fromkeys(tokenize(text).split()))


def plain_text(text):
    # 导入器调用：去 Markdown 加粗/列表标记，再合并空白；返回新的 str，字符串本身不可原地修改。
    # 正则 (...) 是捕获组，替换串 r'\1' 表示保留第一组内容，所以去的是 ** 而不是正文。
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def complete_sentences(text):
    """Split only outside balanced quotes/brackets; retain source punctuation."""
    # source_spans 调用；返回完整句子列表，避免在引号/括号内遇到句号就错误截断。
    text = str(text or '')
    # zip 把左右括号逐对配起来，dict 变成 {左括号:右括号} 查表。
    pairs = dict(zip('“‘「『（(【[〔{《', '”’」』）)】]〕}》'))
    # stack 当作栈保存待闭合符号；result 存完整句；start 是当前句开始的字符下标。
    stack, result, start = [], [], 0
    # enumerate 同时给出下标 index 和字符 char；stack[-1] 是最后入栈的闭合符号。
    for index, char in enumerate(text):
        if stack and char == stack[-1]:
            stack.pop()
        elif char in pairs:
            stack.append(pairs[char])
        elif char in ('"', "'"):
            # Apostrophes within Latin words are not quotation boundaries.
            if not (char == "'" and index and index + 1 < len(text)
                    and text[index - 1].isascii() and text[index - 1].isalnum()
                    and text[index + 1].isascii() and text[index + 1].isalnum()):
                stack.append(char)
        if not stack and char in '。！？!?；;\n':
            # 栈空表示已回到引号/括号外；切片右端不包含，所以 index+1 才把标点保留下来。
            part = text[start:index + 1].strip()
            if part:
                result.append(part)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        result.append(tail)
    return result


def source_spans(text):
    """Return up to four complete source prefixes, preferring 60–180 chars.

    Prefixes retain the subject and original attribution for later sentences.
    Corrections stay whole when shortening could change their meaning.
    """
    # NpcTalkSkill 的来源节选路径使用；返回可供选择的原文前缀列表，而不是模型概括。
    # 下方先检查后文更正/转折，再决定可在哪些完整句末尾截断，避免只取到被否定的前半句。
    text = str(text or '').strip()
    sentences = complete_sentences(text)
    if len(sentences) <= 1:
        return [text] if text else []
    earlier = r'最初|起初|当初|曾|原先|过去|以前|疑似|传闻|自称'
    corrected = r'后来|此后|随后|最终|如今|现(?:在|任|为|作为)|真相|澄清|实为|并非|不是|非感染|未曾|实际上|事实上'
    # Every candidate is a prefix, so its original speaker/frame is retained.
    # Only a later correction can make an earlier account unsafe to select.
    if (any(re.search(r'澄清|真相|更正|实际上|事实上|实为', part) for part in sentences[1:])
            or any(re.search(corrected, text[match.end():]) for match in re.finditer(earlier, text))):
        return [text]

    # compile 把重复使用的正则编译为对象；后面 qualification.search(...) 调它的方法。
    qualification = re.compile(
        r'但|然而|不过|相反|只是|并非|并不|不是|不等于|未必|未曾|不代表|不意味|'
        r'实际上|事实上|后来|此后|随后|最终|如今|澄清|真相|更正')
    ends, cursor = [], 0
    for index, sentence in enumerate(sentences):
        start = text.find(sentence, cursor)
        end = start + len(sentence)
        cursor = end
        following = sentences[index + 1] if index + 1 < len(sentences) else ''
        first_clause = re.split(r'[，,：:]', following, maxsplit=1)[0]
        if not qualification.search(first_clause):
            ends.append(end)
    # A selector can choose a concise introduction or a longer factual
    # prefix. Never offer an isolated pronoun or cut a qualifying next clause.
    short = [end for end in ends if end <= 180]
    chosen = [short[0]] if short else []
    for target in (60, 100):
        end = next((end for end in short if end >= target and end not in chosen), None)
        if end is not None:
            chosen.append(end)
    chosen.append(len(text))
    # chosen 是截止位置列表；set 去重、sorted 排序，列表推导式生成每个长度的前缀。
    return [text[:end] for end in sorted(set(chosen))]


def chunks(text, max_chars=420):
    """Retain complete sentences; a long indivisible sentence stays intact."""
    # import_lore_wiki._paragraphs 调用；yield 使这个函数成为生成器，逐块提供字符串。
    # 调用 chunks(...) 先得到生成器对象；for 真正迭代时才执行并陆续取到各块。
    part = ''
    for sentence in re.findall(r'[^。！？!?；;]+[。！？!?；;]*', plain_text(text)):
        if part and len(part) + len(sentence) > max_chars:
            yield part.strip()
            part = ''
        part += sentence
    if part.strip():
        yield part.strip()


def split_sentences(text):
    # 检索筛选/句子打分使用的轻量分句器；返回保留句末标点的非空片段列表。
    return [part.strip() for part in re.findall(
        r'[^。！？!?；;]+[。！？!?；;]+[”」』\"]*|[^。！？!?；;]+$', text) if part.strip()]


def narrative_excerpt(text):
    """Preserve factual narration and quotations under their source-owned label.

    Quotation marks also wrap project names, and first-person words can occur in
    third-person narration. Removing those entire sentences destroys evidence.
    The enclosing evidence block carries the owner; quoted 'I' is source text,
    never independent proof of the current speaker's memories.
    """
    # retrieval_policy/retrieve_lore 调用；仅把多段空白收为一个空格，保留叙述与引语内容。
    return re.sub(r'\s+', ' ', str(text or '')).strip()
