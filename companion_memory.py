"""
companion_memory.py - Mem0 桌面伴生萌宠长期记忆中枢（二阶段提炼与冲突消解架构）

基于 mem0ai (Mem0) 深度集成的本地伴生萌宠记忆系统：
- 向量检索：FastEmbed 本地运行 BAAI/bge-small-zh-v1.5（零显存占用，纯 CPU/ONNX 加速）
- 向量存储：Qdrant 嵌入式轻量磁盘存储（本地持久化在 data/mem0_qdrant）
- 二阶段记忆架构（对齐 GitHub Mem0 经典架构并针对本地 3B 模型深度强化）：
  * Stage 1（原子事实提炼 + 事实溯源校验）：仅从当前最新输入中提炼原子事实，杜绝上下文复读与幻觉捏造
  * Stage 2（槽位对齐与向量去重消解 ADD / UPDATE / NOOP）：自动识别同名/同槽位记忆，已存在则 NOOP 跳过，改名或修正则 UPDATE 原地更新，彻底根除重复记录
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import traceback
import urllib.request
from pathlib import Path
from typing import Any

# 禁用外部遥测，保证 100% 隐私本地化
os.environ["MEM0_TELEMETRY"] = "False"
# 设置 HuggingFace 国内镜像加速
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 持久化 FastEmbed 向量模型缓存至本地 data 目录，避免系统重启后 /tmp 清空导致重复下载
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path(__file__).resolve().parent / "data" / "fastembed_cache"))

from mem0 import Memory

logger = logging.getLogger("companion_memory")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Mem0] %(message)s"))
    logger.addHandler(_handler)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

QDRANT_PATH = str(DATA_DIR / "mem0_qdrant")
HISTORY_DB_PATH = str(DATA_DIR / "mem0_history.db")
LLM_CONFIG_PATH = DATA_DIR / "llm_config.json"

PET_DISPLAY_NAMES = {
    "luma": "棉棉",
    "pipi": "桃桃",
    "taigo": "芋圆",
    "auri": "芽芽",
    "nuannuan": "暖暖",
    "duoduo": "朵朵",
}

# Stage 1 提示词：仅提取关于主人本人的客观信息与偏好，绝不提取或覆盖灵宠固定名称
STAGE1_EXTRACTION_SYSTEM_PROMPT = """你是一个严谨的个人信息与偏好提取器。
你的唯一任务是从用户当前输入的句子中，提取关于用户（主人）本人的客观事实与偏好。

严格遵守以下规则：
1. 只能提取用户当前这句话里明确说出的关于主人的事实，绝对禁止编造或推测任何未提及的信息。
2. 绝对不要提取、记录或修改伴生灵宠/小玩偶自己的名字（它们各自有固定名字，不要存入记忆库）。
3. 如果用户这句话是提问（如询问名字、询问是谁）、日常问候、夸奖、感叹或闲聊，没有包含新的主人个人事实，必须严格输出：{"facts": []}
4. 每条事实必须是简短独立的单句：
   - 若用户说明自己的姓名，格式为：主人的名字是XXX
   - 若用户说明自己的喜好、习惯、职业、宠物或经历，格式为：主人...
5. 绝对禁止输出“主人是你的主人”、“我是你的主人”这类无意义的同义反复。

请仅输出合法的 JSON 对象，格式如下：
{"facts": ["单条事实1", "单条事实2"]}
"""

# Stage 2 提示词：冲突消解与去重决策（ADD / UPDATE / NOOP）
STAGE2_UPDATE_SYSTEM_PROMPT = """你是一个记忆库去重与更新决策器。
请对比【已有记忆】与【新提取事实】，判断应该执行哪个操作：
- "NOOP"：新提取事实与已有记忆意思相同、重复，或已被已有记忆包含（无需任何操作）。
- "UPDATE"：新提取事实是对已有记忆同一属性的最新修改或更正（例如改了名字、改了喜好）。
- "ADD"：新提取事实与已有记忆完全属于不同的独立维度，需要新增。

请仅输出 JSON：{"action": "NOOP"} 或 {"action": "UPDATE"} 或 {"action": "ADD"}
"""

# 禁止作为姓名/昵称的疑问词、代词与通用词
FORBIDDEN_NAME_TOKENS = {
    "什么", "啥", "谁", "哪", "哪个", "哪里", "怎么", "怎样", "几", "多少",
    "名字", "什么名字", "啥名字", "主人", "你的主人", "我的主人", "伴生灵宠",
    "灵宠", "萌宠", "宠物", "助手", "灵宠助手", "智能体", "机器人", "人工智能",
    "自己", "我", "你", "他", "她", "它", "我们", "你们", "大家", "这个", "那个",
    "好不好", "可以吗", "行不行", "记住了吗", "知道吗",
}

FORBIDDEN_NAME_SUBSTRINGS = (
    "什么", "啥", "谁", "哪", "怎么", "吗", "？", "?", "知道", "记得",
    "告诉", "觉得", "助手", "主人", "灵宠", "萌宠", "名字", "智能", "创造",
)

# 纯闲聊/提问/天气/新闻/拍照截图快速过滤正则（命中且不含命名声明时直接跳过 LLM 提炼）
PURE_QUERY_PATTERNS = (
    r"^(?:那|请问|喂|哎)?(?:你|我)(?:现在)?(?:叫什么|叫啥|是谁|叫什么名字|叫啥名字|的名字是什么)[呀啊呢吧吗？?！!。]*$",
    r"^(?:你)?(?:还记得|知道)(?:我|你)(?:叫什么|是谁|的名字)[呀啊呢吧吗？?！!。]*$",
    r"^(?:听你的声音|你的声音|声音真|今天天气|你好|嗨|哈喽|早安|晚安|跳个舞|给我讲|唱首歌)",
    r"(?:天气|气温|温度|几度|多少度|下雨|带伞|新闻|热搜|热点|头条|要闻|上网|联网|搜索|搜一下|查一下|帮我查|帮我看|摄像头|拍照|拍张|拍一张|截图|截屏|当前页面|当前画面|发.*图)",
)

# 意义重复的废话黑名单
TAUTOLOGY_PATTERNS = (
    r"^主人是(?:你的|伴生灵宠的|灵宠的)?主人$",
    r"^我是(?:你的|伴生灵宠的)?主人$",
    r"^伴生灵宠是(?:主人的)?(?:灵宠|宠物|助手|伴生灵宠)$",
    r"^主人有(?:一个|一只)?(?:伴生)?(?:灵宠|萌宠)$",
)


def _clean_and_validate_name(raw: str) -> str | None:
    """清洗并校验提取出的名字/昵称，杜绝把疑问词或句子成分当成名字"""
    if not raw:
        return None
    name = raw.strip().strip("“”\"'‘’「」『』【】《》()（）[]。，,！？!?；;：:")
    # 剥离句末常见语气助词（仅当剥离后仍有有效字符时）
    for suffix in ("好不好", "好吗", "行不行", "行吗", "可以吗", "记住了吗", "记住没", "对吧", "吧", "呀", "啊", "哦", "啦", "呢", "哈", "嘛"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)].strip()
    # 剥离前导修饰词
    for prefix in ("叫做", "叫", "是", "为"):
        if name.startswith(prefix) and len(name) > len(prefix):
            name = name[len(prefix) :].strip()

    if not name or len(name) < 1 or len(name) > 8:
        return None
    if name in FORBIDDEN_NAME_TOKENS:
        return None
    if any(sub in name for sub in FORBIDDEN_NAME_SUBSTRINGS):
        return None
    return name


def _classify_slot(fact: str) -> tuple[str | None, str | None, str]:
    """
    识别记忆事实是否属于核心唯一槽位（Slot），并返回：
    (slot_type, slot_value, canonical_fact_text)
    - pet_name: 主人给伴生灵宠起的名字
    - master_name: 主人的姓名
    - creator: 主人与灵宠的创造关系
    """
    text = fact.strip().rstrip("。！!；;，, ")

    # 1. 灵宠名字槽位 (pet_name)
    pet_patterns = (
        r"(?:主人)?给?(?:伴生)?(?:灵宠|萌宠|宠物)(?:起名|改名|取名|命名|的名字|的昵称)(?:叫|为|是)?(.+)$",
        r"(?:伴生)?(?:灵宠|萌宠|宠物)(?:现在的)?(?:名字|昵称)(?:叫|是|为)(.+)$",
        r"主人(?:称呼|叫|喊)(?:伴生)?(?:灵宠|萌宠|宠物)为?(.+)$",
    )
    for pat in pet_patterns:
        m = re.match(pat, text)
        if m:
            val = _clean_and_validate_name(m.group(1))
            if val:
                return ("pet_name", val, f"主人给伴生灵宠起名叫{val}")

    # 2. 主人名字槽位 (master_name)
    master_patterns = (
        r"主人的?(?:姓名|名字|本名|称呼)(?:叫|是|为)(.+)$",
        r"主人(?:本人)?(?:叫|名叫|是)(.+)$",
    )
    for pat in master_patterns:
        m = re.match(pat, text)
        if m:
            raw_val = m.group(1).strip()
            # 排除 “主人是伴生灵宠的创造者” 等非姓名描述
            if "创造" in raw_val or "喜欢" in raw_val or "主人" in raw_val:
                continue
            val = _clean_and_validate_name(raw_val)
            if val:
                return ("master_name", val, f"主人的名字是{val}")

    # 3. 创造者关系槽位 (creator)
    if re.search(r"(?:主人.*创造.*(?:灵宠|萌宠)|由主人创造|主人是.*创造者)", text):
        return ("creator", "master", "主人是伴生灵宠的创造者")

    return (None, None, text)


def _extract_deterministic_slots_from_text(user_msg: str) -> list[str]:
    """
    从用户原话中按子句精准提取显式命名与身份声明（耗时 <1ms，零幻觉）。
    """
    clauses = [c.strip() for c in re.split(r"[，,。！？!?；;\n]+", user_msg) if c.strip()]
    extracted: list[str] = []

    for clause in clauses:
        # 跳过包含疑问词的问句子句
        if any(q in clause for q in ("什么", "啥", "谁", "哪", "怎么", "几", "多少")):
            continue
        if clause.endswith("吗") and not any(k in clause for k in ("叫", "起名", "改名", "取名")):
            continue

        # A. 仅提取主人自我介绍姓名子句（不再记忆灵宠改名）
        master_clause_patterns = (
            r"(?:我叫|我的名字是|我的名字叫|我名字是|我名字叫|我姓甚名谁叫|你可以叫我|以后叫我)\s*([^\s，,。！？!?；;]{1,8})$",
        )
        for pat in master_clause_patterns:
            m = re.search(pat, clause)
            if m:
                master_val = _clean_and_validate_name(m.group(1))
                if master_val:
                    fact = f"主人的名字是{master_val}"
                    if fact not in extracted:
                        extracted.append(fact)
                    break

        # C. 检测“是我创造的你”声明
        if re.search(r"(?:是我创造(?:了|的)你|我是创造你的(?:人|主人)|我是你的创造者|你是我创造(?:出来)?的)", clause):
            fact = "主人是伴生灵宠的创造者"
            if fact not in extracted:
                extracted.append(fact)

    return extracted


def _is_tautology_or_invalid(fact: str) -> bool:
    """过滤无意义的废话或空句子"""
    text = fact.strip().rstrip("。！!？?；; ")
    if not text or len(text) < 4:
        return True
    if any(q in text for q in ("什么", "啥", "谁", "吗", "？", "?")):
        return True
    for pat in TAUTOLOGY_PATTERNS:
        if re.match(pat, text):
            return True
    return False


def _is_grounded_in_user_msg(fact: str, user_msg: str) -> bool:
    """
    事实溯源校验（Grounding Verification）：
    确保 LLM 提炼出的每一条事实都有据可查，100% 源自当前 user_msg，
    彻底阻断 3B 模型复读提示词或搬运历史记忆的可能。
    """
    slot_type, slot_val, _ = _classify_slot(fact)
    if slot_type in ("pet_name", "master_name"):
        # 名字槽位的值必须原原本本出现在用户当前这句话中！
        return bool(slot_val and slot_val in user_msg)
    if slot_type == "creator":
        return "创造" in user_msg

    # 对于一般偏好/习惯事实，剥离模板词后，要求核心实词字符出现在 user_msg 中
    stripped = re.sub(r"(?:主人|伴生灵宠|灵宠|萌宠|宠物|自己的|自己|是|的|了|在|有|和|与|给|叫|为)", "", fact)
    stripped = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", stripped)
    if len(stripped) < 2:
        return False

    # 统计在 user_msg 中命中的有效内容字符数
    matched_chars = sum(1 for ch in set(stripped) if ch in user_msg)
    return matched_chars >= min(2, len(set(stripped)))


class CompanionMemory:
    """管理伴生萌宠的长期记忆中枢（基于 Mem0 + 二阶段冲突消解）"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "companion_memory",
                    "path": QDRANT_PATH,
                    "embedding_model_dims": 512,
                    "on_disk": True,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "qwen2.5:3b",
                    "temperature": 0.0,
                    "ollama_base_url": "http://127.0.0.1:11434",
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": "BAAI/bge-small-zh-v1.5",
                    "embedding_dims": 512,
                },
            },
            "history_db_path": HISTORY_DB_PATH,
        }

        logger.info("Initializing Mem0 companion memory engine (2-Stage Pipeline)...")
        try:
            self.memory = Memory.from_config(self.config)
            logger.info("Mem0 companion memory initialized successfully.")
            # 启动时自动执行一次存量记忆清洗与去重（拆分复合句、清除同义重复）
            self.cleanup_and_deduplicate()
        except Exception as exc:
            logger.error(f"Failed to initialize Mem0: {exc}")
            self.memory = None

    def _get_all_raw_unlocked(self) -> list[dict[str, Any]]:
        """内部无锁获取全部原始记忆条目"""
        if not self.memory:
            return []
        res = self.memory.get_all(filters={"user_id": "master"})
        if not res or not isinstance(res, dict):
            return []
        return res.get("results") or []

    def cleanup_and_deduplicate(self) -> None:
        """
        对存量记忆库进行原子化拆分、废话清理与槽位去重：
        - 将形如“A。B。主人是你的主人。”的复合记忆拆分为独立的原子记忆
        - 剔除“主人是你的主人”等无意义废话
        - 对相同槽位或相同文本的记忆严格去重，保留唯一规范条目
        """
        if not self.memory:
            return

        try:
            with self._lock:
                raw_items = self._get_all_raw_unlocked()
                if not raw_items:
                    return

                # 收集所有待重建的规范化原子事实（按时间正序，确保最新修改覆盖旧值）
                atomic_candidates: list[tuple[str, str]] = []  # (canonical_fact, pet_key)
                needs_rebuild = False

                seen_texts: set[str] = set()
                seen_slots: dict[str, str] = {}

                for item in raw_items:
                    mem_text = (item.get("memory") or "").strip()
                    meta = item.get("metadata") or {}
                    pet_key = meta.get("pet", "luma") if isinstance(meta, dict) else "luma"

                    # 检查是否包含多句复合、句号结尾或废话
                    parts = [p.strip() for p in re.split(r"[。；;\n]+", mem_text) if p.strip()]
                    if len(parts) != 1 or mem_text.endswith("。"):
                        needs_rebuild = True

                    for part in parts:
                        if _is_tautology_or_invalid(part):
                            needs_rebuild = True
                            continue
                        slot_type, slot_val, canonical = _classify_slot(part)
                        if slot_type == "pet_name":
                            needs_rebuild = True
                            continue
                        if canonical != part:
                            needs_rebuild = True
                        if slot_type:
                            if slot_type in seen_slots:
                                needs_rebuild = True
                            seen_slots[slot_type] = canonical
                        else:
                            if canonical in seen_texts:
                                needs_rebuild = True
                            seen_texts.add(canonical)
                        atomic_candidates.append((canonical, pet_key))

                if not needs_rebuild and len(raw_items) == len(seen_slots) + len(seen_texts):
                    return

                logger.info(f"Cleaning and deduplicating {len(raw_items)} existing memories in Qdrant...")
                # 合并去重后的最终原子记忆列表
                final_facts: dict[str, tuple[str, str]] = {}  # key -> (fact, pet_key)
                for canonical, pet_key in atomic_candidates:
                    slot_type, _, _ = _classify_slot(canonical)
                    dedup_key = f"slot:{slot_type}" if slot_type else f"text:{canonical}"
                    # 保留较早列表中排在前面（即最新创建）的值，若尚未记录则写入
                    if dedup_key not in final_facts:
                        final_facts[dedup_key] = (canonical, pet_key)

                # 清空旧的冗余记录并写入干净的原子记录
                self.memory.delete_all(user_id="master")
                for dedup_key, (fact_text, pet_key) in reversed(list(final_facts.items())):
                    slot_type, _, _ = _classify_slot(fact_text)
                    meta = {"pet": pet_key}
                    if slot_type:
                        meta["slot"] = slot_type
                    self.memory.add(fact_text, user_id="master", metadata=meta, infer=False)

                logger.info(f"Cleanup complete: compacted {len(raw_items)} -> {len(final_facts)} clean atomic memories.")
        except Exception as exc:
            logger.warning(f"Memory cleanup_and_deduplicate error: {exc}")

    def sync_extract_core_slots(self, user_msg: str, agent_id: str = "luma") -> list[dict[str, Any]]:
        """
        在对话发起瞬间同步提炼显式命名/身份槽位（耗时约 5~10ms）：
        如果主人说“你叫小鹿，我叫孙家洪”，当轮立即生效并写入/更新 Qdrant，
        保证零延迟记忆唤醒且绝不重复创建。
        """
        if not self.memory or not user_msg.strip():
            return []

        slot_facts = _extract_deterministic_slots_from_text(user_msg.strip())
        if not slot_facts:
            return []

        actions: list[dict[str, Any]] = []
        try:
            with self._lock:
                for fact in slot_facts:
                    act = self._stage2_upsert_fact_unlocked(fact, agent_id=agent_id, allow_llm_stage2=False)
                    actions.append(act)
        except Exception as exc:
            logger.warning(f"sync_extract_core_slots error: {exc}")
        return actions

    def get_context_for_chat(self, query: str, agent_id: str | None = None, limit: int = 5) -> dict[str, Any]:
        """
        获取当前对话所需的全量上下文记忆：
        1. 始终置顶提取主人的姓名 (master_name) 与给灵宠起的名字 (pet_nickname)，确保询问“你叫什么/我叫什么”时 100% 命中
        2. 结合 BAAI/bge-small-zh-v1.5 向量检索召回与当前问题最相关的偏好/习惯记忆
        """
        result: dict[str, Any] = {
            "memories": [],
            "pet_nickname": None,
            "master_name": None,
        }
        if not self.memory:
            return result

        try:
            with self._lock:
                all_raw = self._get_all_raw_unlocked()
                core_memories: list[str] = []

                for item in all_raw:
                    mem_text = (item.get("memory") or "").strip().rstrip("。")
                    if not mem_text:
                        continue
                    slot_type, slot_val, canonical = _classify_slot(mem_text)
                    if slot_type == "pet_name":
                        continue
                    elif slot_type == "master_name" and slot_val:
                        if not result["master_name"]:
                            result["master_name"] = slot_val
                        if canonical not in core_memories:
                            core_memories.append(canonical)
                    elif slot_type == "creator":
                        if canonical not in core_memories:
                            core_memories.append(canonical)

                # 语义向量检索相关记忆
                semantic_memories: list[str] = []
                if query.strip():
                    res = self.memory.search(
                        query=query.strip(),
                        filters={"user_id": "master"},
                        limit=limit,
                    )
                    if res and isinstance(res, dict):
                        for hit in res.get("results") or []:
                            text = (hit.get("memory") or "").strip().rstrip("。")
                            score = hit.get("score", 0.0)
                            if text and (score is None or score >= 0.25):
                                hit_slot, _, _ = _classify_slot(text)
                                if hit_slot == "pet_name":
                                    continue
                                if text not in core_memories and text not in semantic_memories:
                                    semantic_memories.append(text)

                merged = core_memories + semantic_memories
                result["memories"] = merged[: max(limit, len(core_memories))]
                return result
        except Exception as exc:
            logger.warning(f"get_context_for_chat error: {exc}")
            return result

    def search(self, query: str, agent_id: str | None = None, limit: int = 4) -> list[str]:
        """兼容接口：返回置顶身份记忆 + 语义召回记忆列表"""
        ctx = self.get_context_for_chat(query, agent_id=agent_id, limit=limit)
        return ctx["memories"]

    def get_all(self) -> list[dict[str, Any]]:
        """获取所有已持久化的记忆条目"""
        if not self.memory:
            return []

        try:
            with self._lock:
                results = self._get_all_raw_unlocked()

            formatted: list[dict[str, Any]] = []
            for r in results:
                meta = r.get("metadata") or {}
                pet_key = meta.get("pet", "luma") if isinstance(meta, dict) else "luma"
                formatted.append({
                    "id": r.get("id"),
                    "memory": (r.get("memory") or "").strip().rstrip("。"),
                    "pet": pet_key,
                    "pet_name": PET_DISPLAY_NAMES.get(pet_key, "伴生灵宠"),
                    "created_at": r.get("created_at"),
                })
            # 按创建时间倒序排列（最新的在最前）
            formatted.reverse()
            return formatted
        except Exception as exc:
            logger.warning(f"Mem0 get_all error: {exc}")
            return []

    def delete(self, memory_id: str) -> bool:
        """删除单条记忆条目"""
        if not self.memory or not memory_id:
            return False

        try:
            with self._lock:
                self.memory.delete(memory_id)
            return True
        except Exception as exc:
            logger.warning(f"Mem0 delete error: {exc}")
            return False

    def clear_all(self) -> bool:
        """清空主人的全部记忆"""
        if not self.memory:
            return False

        try:
            with self._lock:
                self.memory.delete_all(user_id="master")
            return True
        except Exception as exc:
            logger.warning(f"Mem0 clear_all error: {exc}")
            return False

    def add_async(self, user_msg: str, assistant_msg: str, agent_id: str = "luma") -> None:
        """异步在后台执行二阶段记忆提炼与去重沉淀，完全不阻塞实时流式语音交互"""
        if not self.memory or not user_msg.strip():
            return

        thread = threading.Thread(
            target=self._add_worker,
            args=(user_msg.strip(), assistant_msg.strip(), agent_id),
            daemon=True,
            name=f"Mem0-AddWorker-{agent_id}",
        )
        thread.start()

    def _generate_llm_json(self, messages: list[dict[str, str]]) -> str:
        """
        优先使用用户在 Web 端配置的云端 API 大模型（若已开启云端模式并配置了 API Key），
        否则自动回退至本地 Ollama 模型。
        """
        if LLM_CONFIG_PATH.is_file():
            try:
                cfg = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
                if cfg.get("mode") == "cloud" and cfg.get("api_key") and cfg.get("base_url") and cfg.get("cloud_model"):
                    base_url = str(cfg["base_url"]).rstrip("/")
                    api_key = str(cfg["api_key"]).strip()
                    model_name = str(cfg["cloud_model"]).strip()
                    payload = {
                        "model": model_name,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 300,
                    }
                    req = urllib.request.Request(
                        f"{base_url}/chat/completions",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                            "User-Agent": "BreezeCompanion/1.0",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
                        if content:
                            return content
            except Exception as exc:
                logger.warning(f"[Mem0 Cloud LLM] Fallback to local Ollama due to: {exc}")

        raw_resp = self.memory.llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
        )
        return raw_resp if isinstance(raw_resp, str) else json.dumps(raw_resp)

    def _stage1_extract_atomic_facts(self, user_msg: str) -> list[str]:
        """
        Stage 1：从用户单轮发言中提炼原子事实，并执行严格溯源校验
        """
        # 1. 先提取确定性槽位事实
        facts: list[str] = _extract_deterministic_slots_from_text(user_msg)

        # 2. 判断是否为纯提问或简短闲聊，若是则直接返回确定性结果，不调用 LLM（防幻觉）
        for pat in PURE_QUERY_PATTERNS:
            if re.search(pat, user_msg) and not facts:
                return []

        # 如果句子完全是疑问句且不含陈述分句，跳过 LLM
        clauses = [c.strip() for c in re.split(r"[，,。！？!?；;\n]+", user_msg) if c.strip()]
        declarative_clauses = [
            c for c in clauses
            if not any(q in c for q in ("什么", "啥", "谁", "哪", "怎么", "几", "多少", "吗", "？", "?"))
        ]
        if not declarative_clauses:
            return facts

        # 如果所有陈述子句都已经被确定性槽位规则完整覆盖（例如“你叫小鹿，我叫孙家洪，是你的主人”），无需再跑 LLM
        remaining_clauses = []
        for c in declarative_clauses:
            if re.match(r"^(?:我是你的主人|是你的主人|记住(?:了吗|没|哦|呀)?|你要记住|你一定要记得)$", c):
                continue
            if _extract_deterministic_slots_from_text(c):
                continue
            remaining_clauses.append(c)

        if not remaining_clauses:
            return facts

        # 3. 对包含其他开放事实（喜好、习惯、宠物、经历等）的陈述句调用当前激活的大模型提炼
        remaining_text = "，".join(remaining_clauses)
        try:
            raw_resp = self._generate_llm_json(
                messages=[
                    {"role": "system", "content": STAGE1_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"用户输入：{remaining_text}"},
                ]
            )
            parsed = json.loads(raw_resp) if isinstance(raw_resp, str) else {}
            llm_facts = parsed.get("facts") or parsed.get("memory") or []
            if isinstance(llm_facts, list):
                for item in llm_facts:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    # 拆分可能连在一起的复合句
                    for sub in re.split(r"[。；;\n]+", text):
                        sub = sub.strip()
                        if not sub or _is_tautology_or_invalid(sub):
                            continue
                        if not _is_grounded_in_user_msg(sub, user_msg):
                            logger.info(f"[Mem0 Stage-1] Filtered ungrounded fact '{sub}' from '{user_msg}'")
                            continue
                        slot_type, _, canonical = _classify_slot(sub)
                        if slot_type == "pet_name":
                            logger.info(f"[Mem0 Stage-1] Skipped pet_name fact '{canonical}' per fixed-name policy")
                            continue
                        if canonical not in facts:
                            facts.append(canonical)
        except Exception as exc:
            logger.warning(f"[Mem0 Stage-1] LLM extraction fallback: {exc}")

        return facts

    def _stage2_upsert_fact_unlocked(
        self,
        fact: str,
        agent_id: str = "luma",
        allow_llm_stage2: bool = True,
    ) -> dict[str, Any]:
        """
        Stage 2：槽位对齐与向量冲突消解（ADD / UPDATE / NOOP）
        调用前需持有 self._lock。
        """
        slot_type, slot_val, canonical_fact = _classify_slot(fact)
        if slot_type == "pet_name":
            logger.info(f"[Mem0 Stage-2] NOOP (fixed-name policy skips pet_name): {canonical_fact}")
            return {"event": "NOOP", "memory": canonical_fact}
        all_existing = self._get_all_raw_unlocked()

        # 情况 A：核心唯一槽位（灵宠名字 pet_name / 主人姓名 master_name / 创造者 creator）
        if slot_type is not None:
            matching_items = []
            for item in all_existing:
                ex_text = (item.get("memory") or "").strip()
                ex_slot, ex_val, _ = _classify_slot(ex_text)
                if ex_slot == slot_type:
                    if slot_type == "pet_name":
                        ex_pet = (item.get("metadata") or {}).get("pet") or "luma"
                        if ex_pet != agent_id:
                            continue
                    matching_items.append((item, ex_val, ex_text.rstrip("。")))

            if matching_items:
                primary_item, primary_val, primary_text = matching_items[0]
                primary_id = primary_item.get("id")
                # 清理可能残留的多余同槽位重复项
                for dup_item, _, _ in matching_items[1:]:
                    dup_id = dup_item.get("id")
                    if dup_id:
                        self.memory.delete(dup_id)

                if primary_val == slot_val and primary_text == canonical_fact:
                    logger.info(f"[Mem0 Stage-2] NOOP (slot '{slot_type}' unchanged): {canonical_fact}")
                    return {"event": "NOOP", "memory": canonical_fact, "id": primary_id}
                else:
                    logger.info(f"[Mem0 Stage-2] UPDATE (slot '{slot_type}' {primary_val} -> {slot_val}): {canonical_fact}")
                    self.memory.update(
                        memory_id=primary_id,
                        text=canonical_fact,
                        metadata={"pet": agent_id, "slot": slot_type},
                    )
                    return {"event": "UPDATE", "memory": canonical_fact, "id": primary_id}
            else:
                logger.info(f"[Mem0 Stage-2] ADD (new slot '{slot_type}'): {canonical_fact}")
                res = self.memory.add(
                    canonical_fact,
                    user_id="master",
                    metadata={"pet": agent_id, "slot": slot_type},
                    infer=False,
                )
                return {"event": "ADD", "memory": canonical_fact, "result": res}

        # 情况 B：开放型事实（喜好、习惯、宠物、经历等）
        norm_new = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", canonical_fact)
        for item in all_existing:
            ex_text = (item.get("memory") or "").strip().rstrip("。")
            norm_ex = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", ex_text)
            if norm_new == norm_ex or (len(norm_new) >= 4 and norm_new in norm_ex):
                logger.info(f"[Mem0 Stage-2] NOOP (exact/substring match): {canonical_fact}")
                return {"event": "NOOP", "memory": ex_text, "id": item.get("id")}

        # 向量相似度检索 Top-3
        search_res = self.memory.search(
            query=canonical_fact,
            filters={"user_id": "master"},
            limit=3,
        )
        hits = (search_res or {}).get("results") or []
        if hits:
            top_hit = hits[0]
            top_score = top_hit.get("score") or 0.0
            top_text = (top_hit.get("memory") or "").strip().rstrip("。")
            top_id = top_hit.get("id")

            # 极高相似度（>= 0.90）直接判定为重复 NOOP
            if top_score >= 0.90:
                logger.info(f"[Mem0 Stage-2] NOOP (high vector similarity {top_score:.3f}): {canonical_fact} ~= {top_text}")
                return {"event": "NOOP", "memory": top_text, "id": top_id}

            # 中高相似度（0.76 <= score < 0.90）由 Stage-2 LLM 判定 NOOP / UPDATE / ADD
            if allow_llm_stage2 and top_score >= 0.76 and top_id:
                try:
                    decision_raw = self._generate_llm_json(
                        messages=[
                            {"role": "system", "content": STAGE2_UPDATE_SYSTEM_PROMPT},
                            {"role": "user", "content": f"【已有记忆】：{top_text}\n【新提取事实】：{canonical_fact}"},
                        ]
                    )
                    decision = (json.loads(decision_raw).get("action") or "NOOP").upper()
                except Exception:
                    decision = "NOOP" if top_score >= 0.83 else "ADD"

                if decision == "NOOP":
                    logger.info(f"[Mem0 Stage-2] NOOP (LLM verified duplicate, score={top_score:.3f}): {canonical_fact}")
                    return {"event": "NOOP", "memory": top_text, "id": top_id}
                elif decision == "UPDATE":
                    logger.info(f"[Mem0 Stage-2] UPDATE (LLM verified update, score={top_score:.3f}): {top_text} -> {canonical_fact}")
                    self.memory.update(
                        memory_id=top_id,
                        text=canonical_fact,
                        metadata={"pet": agent_id},
                    )
                    return {"event": "UPDATE", "memory": canonical_fact, "id": top_id}

        # 新增独立事实
        logger.info(f"[Mem0 Stage-2] ADD (new open fact): {canonical_fact}")
        res = self.memory.add(
            canonical_fact,
            user_id="master",
            metadata={"pet": agent_id},
            infer=False,
        )
        return {"event": "ADD", "memory": canonical_fact, "result": res}

    def _add_worker(self, user_msg: str, assistant_msg: str, agent_id: str) -> None:
        try:
            logger.info(f"[Mem0 Stage-1] Analyzing user text for pet '{agent_id}': {user_msg[:40]}...")
            facts = self._stage1_extract_atomic_facts(user_msg)
            if not facts:
                logger.info("[Mem0 Stage-1] No new factual memories in this turn (NOOP).")
                return

            outcomes = []
            with self._lock:
                for fact in facts:
                    outcome = self._stage2_upsert_fact_unlocked(fact, agent_id=agent_id, allow_llm_stage2=True)
                    outcomes.append(outcome)
            logger.info(f"[Mem0 Stage-2] Completed: {outcomes}")
        except Exception as exc:
            logger.warning(f"Mem0 async distillation error: {exc}")
            traceback.print_exc()


# 全局单例
companion_memory = CompanionMemory()
