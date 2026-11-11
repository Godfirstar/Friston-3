"""SQLite property graph: statements, source spans and mentions stay distinct."""
# 这个模块只提供常量，不在 import 时建库；terra_kb.build_db 把 DDL 传给连接的 executescript。
# DB_VERSION 是当前数据格式标记，terra_kb.connect 读取 meta 表里的 schema 值与它比较。
DB_VERSION = 'lore_wiki_graph_2'
# DDL 是下面整个三引号字符串，内部内容是 SQL，不是 Python 语句。
# 表的关系：entities 存实体；aliases 用 entity_id 指向实体；sources 存原文件/版本/摘要。
# passages 用 ref_id 指向“这段资料属于谁”，用 source_id 指向出处，并保留原文行号。
# relations 的 src/dst 分别指向两个实体，passage_id 回指关系原述；方向表示谁的页面作了陈述。
# event_mentions 只记录活动段落明确提及谁，不能从“共同提及”自动推导两人的关系。
# PRIMARY KEY 是行的唯一标识；REFERENCES 声明外键；CREATE INDEX 为常用查询建立查找索引。
# passage_fts 是 FTS5 全文索引：存 tokenize 产生的检索词，rowid 与 passages.id 对应。
# import_issues 保存导入异常的来源与原因；meta 保存版本/导入统计，不承载对话历史。
DDL = '''
PRAGMA foreign_keys=ON;
CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE sources(
 id TEXT PRIMARY KEY,path TEXT NOT NULL,url TEXT NOT NULL,revision TEXT NOT NULL,
 sha256 TEXT NOT NULL,generated_at TEXT NOT NULL,version TEXT NOT NULL,
 source_type TEXT NOT NULL CHECK(source_type IN ('llm_summary','project_config'))
);
CREATE TABLE entities(
 id TEXT PRIMARY KEY,name TEXT NOT NULL,kind TEXT NOT NULL,
 scope TEXT NOT NULL CHECK(scope IN ('canon','fan')),
 summary TEXT NOT NULL,source TEXT NOT NULL,source_key TEXT NOT NULL,
 coverage TEXT NOT NULL
);
CREATE TABLE aliases(
 alias_norm TEXT NOT NULL,entity_id TEXT NOT NULL REFERENCES entities(id),
 alias TEXT NOT NULL,origin TEXT NOT NULL,priority INTEGER NOT NULL,
 PRIMARY KEY(alias_norm,entity_id)
);
CREATE INDEX idx_alias_norm ON aliases(alias_norm);
CREATE TABLE passages(
 id INTEGER PRIMARY KEY,kind TEXT NOT NULL,ref_id TEXT NOT NULL REFERENCES entities(id),
 section TEXT NOT NULL,category TEXT NOT NULL,content TEXT NOT NULL,source TEXT NOT NULL,
 scope TEXT NOT NULL,source_id TEXT NOT NULL REFERENCES sources(id),
 line_start INTEGER NOT NULL,line_end INTEGER NOT NULL
);
CREATE INDEX idx_passage_owner ON passages(ref_id,category);
CREATE VIRTUAL TABLE passage_fts USING fts5(tokens,content='',tokenize='unicode61');
CREATE TABLE relations(
 id TEXT PRIMARY KEY,src TEXT NOT NULL REFERENCES entities(id),
 dst TEXT NOT NULL REFERENCES entities(id),relation TEXT NOT NULL,description TEXT NOT NULL,
 scope TEXT NOT NULL,source TEXT NOT NULL,source_id TEXT NOT NULL REFERENCES sources(id),
 passage_id INTEGER NOT NULL REFERENCES passages(id),
 status TEXT NOT NULL CHECK(status IN ('source_asserted','uncertain','project')),
 verified INTEGER NOT NULL DEFAULT 0,target_label TEXT NOT NULL
);
CREATE INDEX idx_rel_src ON relations(src,dst);
CREATE INDEX idx_rel_dst ON relations(dst,src);
CREATE TABLE event_mentions(
 event_id TEXT NOT NULL REFERENCES entities(id),entity_id TEXT NOT NULL REFERENCES entities(id),
 passage_id INTEGER NOT NULL REFERENCES passages(id),basis TEXT NOT NULL,
 PRIMARY KEY(event_id,entity_id,passage_id)
);
CREATE INDEX idx_event_entity ON event_mentions(entity_id,event_id);
CREATE TABLE import_issues(
 id INTEGER PRIMARY KEY,path TEXT NOT NULL,reason TEXT NOT NULL,detail TEXT NOT NULL
);
'''
