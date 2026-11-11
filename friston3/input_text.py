"""语音输入只有一个格式限制；绝不纠正单词或推断意图。"""
import re


# re 是标准库正则表达式模块；r'...' 为原始字符串，反斜杠交给正则引擎解释。
# 相邻的两个字符串在括号里会自动拼接；_HAN 保存字符范围描述，不是某个输入句子。
_HAN = (r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
        r'\U00020000-\U0002fa1f\U00030000-\U000323af')
_HAN_TOKEN_SPACE = re.compile(r'(?<=[' + _HAN + r'])\s+(?=[' + _HAN + r'])')
# compile() 返回可复用的模式对象；(?<=...)看左侧，(?=...)看右侧，它们本身不吞掉字符。
# 此模式只匹配左右都是汉字的一个或多个空白，之后的 sub('', text) 仅删除这些空白。
_MEMORY_PREFIX = re.compile(r'^(?:记住|忘记)\s*[:：]')
# ^ 指开头，| 表示二选一，\s* 表示零到多个空白，[:：] 接受半角/全角冒号。


# 调用方：main、Agent.handle、NpcTalkSkill；输出规范化记录字典，不进行语义改写或查库。
# * 后的 input_source 必须按名字传参；默认'text'表示键盘输入。
def prepare_input_text(text, *, input_source='text'):
    """Retain the raw input and produce a format-normalized working copy.

    Some ASR decoders delimit Chinese tokens with spaces. Join only whitespace
    between two Han characters. English word boundaries, digits, punctuation,
    negation and the recognized characters themselves are preserved. Keyboard
    input is only stripped at its outer boundary, as before.
    """
    if not isinstance(text, str):
        raise TypeError('输入必须是文字')
    # 条件表达式只取其中一项；未知来源也按text处理，这是当前代码的约定。
    source = 'speech' if input_source == 'speech' else 'text'
    # strip()返回去掉两端空白的新字符串；raw是这种意义下保留的原文，不是逐字节录音转写。
    raw = text.strip()
    # .sub是前面re.compile返回对象的方法；语音才处理汉字间空格，英文词间空格不匹配。
    normalized = _HAN_TOKEN_SPACE.sub('', raw) if source == 'speech' else raw
    # {}是字典，[]是列表；normalizations记录实际发生的格式变化，未变化则返回空列表。
    return {'input_source': source, 'raw_text': raw, 'normalized_text': normalized,
            'normalizations': ['han_token_whitespace'] if normalized != raw else []}


def is_memory_command(text):
    """Recognize the existing colon-delimited protocol, including its spacing."""
    # main/Agent直接调用；match返回匹配对象或None，bool把它转成True/False，供分支判断。
    return bool(_MEMORY_PREFIX.match(text))
