"""统一管理项目路径和默认配置。"""
from pathlib import Path

# 本文件被 llm/planner/视觉/知识等模块导入，集中提供默认值；导入它不会启动模型或硬件。
# Path 是标准库 pathlib 的路径类；__file__ 是 Python 提供的“本文件所在路径”。
# resolve() 得到绝对路径，.parent 取上一层目录；两次 parent 从 friston3/config.py 找到项目根。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Path 对象的 / 运算符表示拼接子路径，不是数值除法；这里只构造路径，没有创建文件夹。
MODELS_DIR = PROJECT_ROOT / "models"
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
PHOTOS_DIR = PROJECT_ROOT / "photos"
LORE_WIKI_SOURCE = PROJECT_ROOT / "data" / "external" / "arknights_lore_wiki"
KNOWLEDGE_DB = PROJECT_ROOT / "data" / "runtime" / "terra_lore.db"

# 大写名称是“作为配置常量使用”的约定，Python 本身仍允许重新赋值。
# 模型名是传给 Ollama 的字符串；写在这里不代表该模型已安装、已加载或正在生成。
OLLAMA_MODEL = "qwen3.5:9b"
# 127.0.0.1 指本机；11434 是服务端口；/api/chat 是请求路径。发送过程在 llm.chat。
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
NPC_CONTEXT_TOKENS = 4096
# One resident model and one context size avoid swapping weights between routing
# and dialogue. The configured 9B Q4 model fits this machine's 8 GB GPU.
OLLAMA_OPTIONS = {"num_ctx": NPC_CONTEXT_TOKENS, "num_gpu": 99}
# 这是传输给服务的选项字典，num_gpu 不是“本机有99块GPU”。llm.chat 会复制字典，
# 在机械臂策略驻留时把本次请求的 num_gpu 改成0；不会原地改这里的公共默认值。
OLLAMA_KEEP_ALIVE = -1
# keep_alive 控制模型驻留；下面分别是回复token预算、保留历史轮数、历史token粗估预算。
# token 是模型分词单位，不直接等于汉字数；具体裁剪/组装发生在 npc/prompts.py 等调用方。
NPC_REPLY_TOKENS = 192
NPC_HISTORY_TURNS = 3
NPC_HISTORY_TOKEN_BUDGET = 1100
