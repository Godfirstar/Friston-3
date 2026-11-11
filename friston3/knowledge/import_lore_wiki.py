"""Import only released Wiki summaries, never upstream generation prompts.

The source's explicit relationship paragraphs become directed descriptions.
Direction denotes whose page made the statement, not a guessed semantic role.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

from friston3.knowledge.lore_text import chunks, normalize, plain_text, tokenize
from friston3.npc.identity import IDENTITY

REPOSITORY = 'https://github.com/littlepangding/arknights_lore_wiki'
CHAR_SECTIONS = {
    '简要介绍': 'identity', '详细介绍': 'background', '相关角色': 'relationship',
    '剧情高光': 'experience', '战斗表现': 'ability', '战斗能力': 'ability', '相关活动': 'event',
    'trivia': 'trivia', '角色点评': 'personality',
}
EVENT_SECTIONS = {
    '剧情总结': 'event_summary', '剧情高光': 'experience', 'trivia': 'trivia',
    '关键人物': 'participants', '角色剧情概括': 'participants', '关键阵营': 'faction',
}
UNCERTAIN = re.compile(r'可能|或许|疑似|似乎|推测|尚不明确|尚不清楚|未明确|无法确定|暗示|推断')


def _field(text, tag):
    # import_lore_wiki 读取 <名称>...</名称> 等字段；返回匹配内容字符串，缺失时返回 ''。
    # re.escape 让 tag 按字面匹配，re.S 让 . 也能跨行；(.*?) 是非贪婪捕获组。
    match = re.search(r'<' + re.escape(tag) + r'>\s*(.*?)\s*</' + re.escape(tag) + r'>', text, re.S)
    return match.group(1).strip() if match else ''


def _sections(text, tag):
    # 主导入循环使用的生成器；逐个 yield (章节正文, 正文起始行号)，给后续出处定位使用。
    pattern = r'<' + re.escape(tag) + r'>(.*?)</' + re.escape(tag) + r'>'
    for match in re.finditer(pattern, text, re.S):
        yield match.group(1), text.count('\n', 0, match.start(1)) + 1


def _date(value):
    try:
        return datetime.strptime(value, '%Y-%m-%d-%H:%M:%S')
    except ValueError:
        return datetime.min


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _issue(db, path, reason, detail=''):
    db.execute('INSERT INTO import_issues(path,reason,detail) VALUES(?,?,?)', (path, reason, detail))


def _alias(db, entity_id, alias, origin, priority=50):
    # register/别名表导入调用；把一个名字与已存在实体关联，无业务返回值。
    # normalize 用于比较，alias 保留可读写法；SQL 的 ON CONFLICT 处理同一复合主键的重复记录。
    alias = plain_text(alias)
    value = normalize(alias)
    if not value or len(alias) > 100 or alias in ('无', '未知', '暂无', '不详', 'None'):
        return
    db.execute('''INSERT INTO aliases VALUES(?,?,?,?,?) ON CONFLICT(alias_norm,entity_id)
        DO UPDATE SET priority=MAX(aliases.priority,excluded.priority),
        origin=CASE
          WHEN aliases.origin='canonical_name' OR excluded.origin='canonical_name' THEN 'canonical_name'
          WHEN aliases.origin='alias_table' OR excluded.origin='alias_table' THEN 'alias_table'
          ELSE aliases.origin END''',
        (value, entity_id, alias, origin, priority))


def _aliases(text):
    # Semicolons are the upstream format. Do not split slashes into identities.
    return [value.strip() for value in re.split(r'[;；\n]+', text) if value.strip()]


def _passage(db, owner, kind, section, category, content, source_id, url, start, end, scope='canon'):
    # 导入循环调用；owner 是段落所属实体 ID，source_id 是出处记录 ID，start/end 是原文件行号。
    # 返回新段落的整数 ID（空正文返回 None）；同时写普通段落表和全文检索索引。
    content = plain_text(content)
    if not content:
        return None
    source = f'{url}#L{start}-L{end}' if url.startswith('https://') else url
    # db 是 terra_kb.build_db 传入的临时数据库连接；cursor 是 INSERT 返回的游标。
    # VALUES 的 ? 与下面元组按位置对应，正文不会作为 SQL 代码拼接执行。
    cursor = db.execute('''INSERT INTO passages
        (kind,ref_id,section,category,content,source,scope,source_id,line_start,line_end)
        VALUES(?,?,?,?,?,?,?,?,?,?)''',
        (kind, owner, section, category, content, source, scope, source_id, start, end))
    # lastrowid 取得刚插入行的自增 ID；FTS 使用相同 rowid，检索命中后才能回到完整段落。
    passage_id = cursor.lastrowid
    db.execute('INSERT INTO passage_fts(rowid,tokens) VALUES(?,?)',
               (passage_id, tokenize(content + ' ' + section)))
    return passage_id


def _paragraphs(body, start):
    """Chunks retain the enclosing paragraph's exact source line span."""
    # 主导入循环调用；body 是章节正文，start 为章节起始行；yield (片段, 起始行, 结束行)。
    for match in re.finditer(r'\S[^\n]*(?:\n(?!\s*\n)[^\n]+)*', body):
        text = match.group().strip()
        line_start = start + body.count('\n', 0, match.start())
        line_end = start + body.count('\n', 0, match.end())
        # chunks 来自 lore_text，也是生成器；拆长段落时仍保留其所属原段落的行号。
        for part in chunks(text):
            yield part, line_start, line_end


def _relation_rows(body, start):
    """Parse one explicit target label plus its description, including wraps."""
    # 解析“相关角色”章节；yield [对象标签, 原述, 起始行, 结束行]，不推断描述里没有的关系。
    current = None
    for offset, raw in enumerate(body.splitlines()):
        text = plain_text(raw)
        if not text:
            continue
        match = re.match(r'^(.{1,100}?)[：:]\s*(\S.*)$', text)
        if match:
            if current:
                yield current
            current = [match[1].strip(), match[2].strip(), start + offset, start + offset]
        else:
            if current:
                yield current
                current = None
            yield ['', text, start + offset, start + offset]
    if current:
        yield current


def import_lore_wiki(db, source):
    # 唯一主入口由 terra_kb.build_db 调用；db 是已建好表的临时库，source 是本地 Wiki 根目录。
    # 返回 metrics 统计字典；实际写入由传入连接承载，事务提交与最终替换旧库由 build_db 管理。
    root = Path(source).expanduser().resolve()
    # Path 对象封装路径；/ 拼目录，expanduser 展开 ~，resolve 转为绝对路径。
    if not (root / 'data' / 'char_v3').is_dir() or not (root / 'data' / 'stories').is_dir():
        raise FileNotFoundError(f'需要 arknights_lore_wiki 仓库目录，内含 data/char_v3 和 data/stories：{root}')
    try:
        # subprocess 是标准库进程工具；传 argv 列表运行 Git 读版本信息，不在这里下载源资料。
        # capture_output 收集 stdout，check=True 让非零退出码变成异常；这里只读来源版本。
        revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True, timeout=15).stdout.strip()
        dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--', 'data/char_v3',
            'data/stories', 'data/char_alias.txt'], capture_output=True, text=True, check=True, timeout=15).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = 'local', True
    # metrics 记录本次导入事实；documents 暂存解析好的页面，尚不是数据库查询结果。
    metrics = {'repository': REPOSITORY, 'revision': revision, 'working_tree_modified': dirty,
               'source_type': 'llm_summary', 'excluded_prompt_files': 0,
               'character_files_seen': 0, 'duplicate_pages': 0, 'invalid_pages': 0}
    documents = {'character': [], 'event': []}
    # defaultdict(list) 为首次访问的键自动创建空列表，用于保留被较新页面替代的旧别名。
    duplicate_aliases = defaultdict(list)
    # 第一阶段：扫描人物/活动文本，按字段解析并选择同名页面，不把上游生成提示词当资料。
    for kind, folder, name_tag, summary_tag in (
        ('character', 'char_v3', '名称', '简要介绍'),
        ('event', 'stories', '活动名称', '剧情总结'),
    ):
        by_name = {}
        # glob 得到匹配路径，sorted 固定遍历顺序；path 是 Path，read_bytes 返回原始字节。
        for path in sorted((root / 'data' / folder).glob('*.txt')):
            if path.name.startswith('prompt_'):
                metrics['excluded_prompt_files'] += 1
                continue
            if kind == 'character':
                metrics['character_files_seen'] += 1
            raw = path.read_bytes()
            # decode 只做编码转换；utf-8-sig 兼容文件开头的 BOM，原字节仍用于 SHA256 来源摘要。
            text = raw.decode('utf-8-sig')
            name = plain_text(_field(text, name_tag))
            summary = plain_text(_field(text, summary_tag))
            key = _field(text, 'ID')
            relpath = path.relative_to(root).as_posix()
            if not name or not key or (kind == 'event' and not summary):
                _issue(db, relpath, 'invalid_page', '缺少名称、摘要或ID；没有导入拒答/错误正文')
                metrics['invalid_pages'] += 1
                continue
            if not summary:
                _issue(db, relpath, 'empty_profile', '上游只有名称，保留名称索引，不补造人物简介')
            doc = {'name': name, 'summary': summary, 'key': key, 'path': relpath, 'text': text,
                   'sha256': hashlib.sha256(raw).hexdigest(), 'time': _field(text, 'time'),
                   'version': _field(text, 'version'), 'kind': kind}
            # Different operator records may have the same story title.
            canonical = normalize(name) if kind == 'character' else key
            old = by_name.get(canonical)
            if old:
                # lambda 是排序键函数；元组逐项比较，先较新日期，再来源ID特征和路径。
                preference = lambda d: (_date(d['time']), d['key'].startswith('char_'), d['path'])
                newest, replaced = sorted((old, doc), key=preference, reverse=True)
                by_name[canonical] = newest
                duplicate_aliases[canonical].extend(_aliases(_field(replaced['text'], '其他名称')))
                duplicate_aliases[canonical].append(replaced['key'])
                metrics['duplicate_pages'] += 1
                _issue(db, replaced['path'], 'superseded_duplicate', newest['path'])
            else:
                by_name[canonical] = doc
        documents[kind] = list(by_name.values())
    if not documents['character'] or not documents['event']:
        raise ValueError('源数据没有有效人物或活动页面，拒绝替换现有知识库。')

    def register(doc):
        # 内部闭包可读取外层 db/revision/dirty；先登记 sources/entities，再写对应 aliases。
        # doc 是一个页面字典；末尾 update 将新 ID/URL 写回，后面正文导入会复用它。
        source_id = 'wiki:' + doc['path']
        # A local modified snapshot has hashes but is not falsely attributed to the commit.
        url_revision = revision if revision != 'local' and not dirty else 'main'
        url = REPOSITORY + '/blob/' + url_revision + '/' + quote(doc['path'], safe='/')
        db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',
            (source_id, doc['path'], url, revision, doc['sha256'], doc['time'], doc['version'], 'llm_summary'))
        entity_id = ('closure' if doc['name'] == IDENTITY.creator and doc['kind'] == 'character'
                     else ('wiki:' if doc['kind'] == 'character' else 'event:') + doc['key'])
        db.execute('INSERT INTO entities VALUES(?,?,?,?,?,?,?,?)',
            (entity_id, doc['name'], doc['kind'], 'canon', doc['summary'], url, doc['key'],
             'summary' if doc['summary'] else 'reference_only'))
        _alias(db, entity_id, doc['name'], 'canonical_name', 100)
        _alias(db, entity_id, doc['key'], 'source_id', 20)
        for alias in _aliases(_field(doc['text'], '其他名称')) + duplicate_aliases.get(normalize(doc['name']), []):
            _alias(db, entity_id, alias, 'page_alias', 50)
        doc.update(entity_id=entity_id, source_id=source_id, url=url)

    # 第二阶段：先登记所有实体，之后处理关系时目标 ID 才有机会在库中找到。
    for doc in documents['character'] + documents['event']:
        register(doc)
    # names 是 {规范化人物名:实体ID集合}，保留同名冲突，不强行一名一人。
    names = defaultdict(set)
    for entity_id, name in db.execute("SELECT id,name FROM entities WHERE kind='character'"):
        names[normalize(name)].add(entity_id)
    alias_file = root / 'data' / 'char_alias.txt'
    if alias_file.is_file():
        raw = alias_file.read_bytes()
        db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',
            ('wiki:data/char_alias.txt', 'data/char_alias.txt', REPOSITORY + '/blob/' +
             (revision if revision != 'local' and not dirty else 'main') + '/data/char_alias.txt',
             revision, hashlib.sha256(raw).hexdigest(), '', '', 'llm_summary'))
        for line in raw.decode('utf-8-sig').splitlines():
            aliases = _aliases(line)
            # * 展开多个集合参数，union 求并集；一行多个别名可能指向多个同名实体。
            targets = set().union(*(names.get(normalize(a), set()) for a in aliases))
            if len(targets) > 1:
                _issue(db, 'data/char_alias.txt', 'ambiguous_alias_group', line)
            for entity_id in targets:
                for alias in aliases:
                    _alias(db, entity_id, alias, 'alias_table', 40)

    name_map = {key: next(iter(ids)) for key, ids in names.items() if len(ids) == 1}
    alias_map = defaultdict(set)
    for alias, entity_id in db.execute('''SELECT a.alias_norm,a.entity_id FROM aliases a
            JOIN entities e ON e.id=a.entity_id WHERE e.kind='character' '''):
        alias_map[alias].add(entity_id)

    def resolve(label):
        # 本次导入内部使用；标签唯一对应实体时返回 ID，否则返回 None，交给 reference 保留原述。
        key = normalize(label)
        if key in name_map:
            return name_map[key]
        # A qualifier such as “奥达 (母亲)” distinguishes a namesake relative.
        # It must survive even when the upstream alias list is contradictory.
        if re.search(r'[（(][^()（）]*(?:父亲|母亲|哥哥|姐姐|弟弟|妹妹|祖父|祖母)[^()（）]*[)）]', label):
            return None
        matches = alias_map.get(key, set())
        if len(matches) == 1:
            return next(iter(matches))
        return None

    def reference(label, doc):
        # 给无法唯一解析的标签建立“该来源页面内”的引用节点，返回其 ID，不捏造完整人物档案。
        # An unresolved label has identity only within its source page. Two
        # unrelated people naming “父亲” must not create a shared ancestor.
        # 来源实体 ID 参与摘要，因此两页都写“父亲”不会被合并为同一个人；[:20] 取摘要前缀。
        entity_id = 'ref:' + _digest(doc['entity_id'] + ':' + normalize(label))[:20]
        db.execute('INSERT OR IGNORE INTO entities VALUES(?,?,?,?,?,?,?,?)',
            (entity_id, label, 'reference', 'canon', '', doc['url'], label, 'reference_only'))
        # Reference labels are explored through their owner, never matched as
        # independent people in questions such as “多萝西的助手是谁”.
        return entity_id

    def targets(label, doc):
        # 返回一个或多个目标 ID；群体标签只在每个成员都能明确解析时拆开，否则保留来源引用。
        direct = resolve(label)
        if direct and direct != doc['entity_id']:
            return [direct]
        if direct == doc['entity_id']:
            _issue(db, doc['path'], 'ambiguous_self_reference', label)
            return [reference(label, doc)]
        parts = [p.strip() for p in re.split(r'[;；、]', label) if p.strip()]
        matches = [resolve(p) for p in parts]
        if len(parts) > 1 and all(matches) and doc['entity_id'] not in matches:
            return list(dict.fromkeys(matches))
        return [reference(label, doc)]

    # 第三阶段：按已声明章节类别导入正文与明确关系；所有关系都回指一条原文段落和来源。
    for doc in documents['character'] + documents['event']:
        sections = CHAR_SECTIONS if doc['kind'] == 'character' else EVENT_SECTIONS
        for section, category in sections.items():
            for body, start in _sections(doc['text'], section):
                if section == '相关角色':
                    for index, (label, description, first, last) in enumerate(_relation_rows(body, start)):
                        if not label:
                            _issue(db, doc['path'], 'unparsed_relation', description)
                            continue
                        paragraph = label + '：' + description
                        passage_id = _passage(db, doc['entity_id'], 'relationship', section, category,
                            paragraph, doc['source_id'], doc['url'], first, last)
                        for target in targets(label, doc):
                            relation_id = 'rel:' + _digest(f'{doc["source_id"]}:{first}:{target}')[:24]
                            # UNCERTAIN 是预编译正则；标记“原述存疑”与“原述断言”，都不等于人工事实核验。
                            status = 'uncertain' if UNCERTAIN.search(description) else 'source_asserted'
                            db.execute('INSERT INTO relations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                                (relation_id, doc['entity_id'], target, 'described_relationship',
                                 doc['name'] + ' → ' + label + '：' + description,
                                 'canon', f'{doc["url"]}#L{first}-L{last}', doc['source_id'],
                                 passage_id, status, 0, label))
                    continue
                kind = 'event' if doc['kind'] == 'event' else ('profile' if section == '简要介绍' else 'character')
                for content, first, last in _paragraphs(body, start):
                    _passage(db, doc['entity_id'], kind, section, category, content,
                             doc['source_id'], doc['url'], first, last)

    # Explicit textual mentions attach people to event passages, never to each other.
    # Single-character names are only accepted as exact roster labels.
    # 第四阶段：在活动正文里找明确名字，写 event_mentions；不会把同场提及转成两人关系边。
    mention_names = {name: entity_id for name, entity_id in db.execute(
        "SELECT name,id FROM entities WHERE kind='character'") if len(name) >= 2}
    # 正则 | 连接多种名字，长名在前；escape 防止名称中的括号等符号改变匹配规则。
    pattern = re.compile('|'.join(re.escape(n) for n in sorted(mention_names, key=len, reverse=True))) if mention_names else None
    for passage_id, event_id, content, section in db.execute(
            "SELECT id,ref_id,content,section FROM passages WHERE kind='event'").fetchall():
        mentioned = set()
        if pattern:
            for match in pattern.finditer(content):
                name = match.group()
                if re.fullmatch('[A-Za-z0-9_]+', name) and (
                    (match.start() and content[match.start()-1].isascii() and content[match.start()-1].isalnum()) or
                    (match.end() < len(content) and content[match.end()].isascii() and content[match.end()].isalnum())):
                    continue
                mentioned.add(mention_names[name])
        if section in ('关键人物', '角色剧情概括'):
            for label in re.split(r'[;；、：:\n]', content):
                if target := resolve(label.strip()):
                    mentioned.add(target)
        for entity_id in sorted(mentioned):
            db.execute('INSERT OR IGNORE INTO event_mentions VALUES(?,?,?,?)',
                       (event_id, entity_id, passage_id, 'explicit_name'))

    # 返回统计供 build_db 记录/检查；执行到这里仍不由本函数替换正式数据库文件。
    metrics.update(characters=len(documents['character']), events=len(documents['event']))
    metrics['alias_collisions'] = db.execute('''SELECT count(*) FROM
        (SELECT alias_norm FROM aliases GROUP BY alias_norm HAVING count(*)>1)''').fetchone()[0]
    return metrics


