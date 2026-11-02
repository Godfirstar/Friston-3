"""Shared, dependency-free text utilities for the lore wiki graph."""
import re
import unicodedata


def normalize(text):
    value = unicodedata.normalize('NFKC', str(text or '')).casefold()
    return re.sub(r'[\s\-_.·•:：,，。？！!?()（）“”‘’\[\]<>]+', '', value)


def tokenize(text):
    tokens = []
    for term in re.findall(r'[\u3400-\u9fff]+|[a-z0-9_]+', unicodedata.normalize('NFKC', text).casefold()):
        if '\u3400' <= term[0] <= '\u9fff':
            tokens.extend(term[i:i+2] for i in range(len(term)-1))
        elif len(term) >= 2:
            tokens.append(term)
    return ' '.join(tokens)


def phrase_query(text):
    return ' AND '.join('"' + word.replace('"', '""') + '"'
                        for word in dict.fromkeys(tokenize(text).split()))


def plain_text(text):
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'^\s*[-*]\s+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def complete_sentences(text):
    """Split only outside balanced quotes/brackets; retain source punctuation."""
    text = str(text or '')
    pairs = dict(zip('“‘「『（(【[〔{《', '”’」』）)】]〕}》'))
    stack, result, start = [], [], 0
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
    return [text[:end] for end in sorted(set(chosen))]


def chunks(text, max_chars=420):
    """Retain complete sentences; a long indivisible sentence stays intact."""
    part = ''
    for sentence in re.findall(r'[^。！？!?；;]+[。！？!?；;]*', plain_text(text)):
        if part and len(part) + len(sentence) > max_chars:
            yield part.strip()
            part = ''
        part += sentence
    if part.strip():
        yield part.strip()


def split_sentences(text):
    return [part.strip() for part in re.findall(
        r'[^。！？!?；;]+[。！？!?；;]+[”」』\"]*|[^。！？!?；;]+$', text) if part.strip()]


def narrative_excerpt(text):
    """Preserve factual narration and quotations under their source-owned label.

    Quotation marks also wrap project names, and first-person words can occur in
    third-person narration. Removing those entire sentences destroys evidence.
    The enclosing evidence block carries the owner; quoted 'I' is source text,
    never independent proof of the current speaker's memories.
    """
    return re.sub(r'\s+', ' ', str(text or '')).strip()
