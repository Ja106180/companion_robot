from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
import random
import re
import shutil
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from companion_alarm import companion_alarm
from companion_asr import _normalize_and_detect_pet, companion_asr
from companion_face import companion_face
from companion_memory import companion_memory
from companion_mobile import companion_mobile
from companion_sentinel import companion_sentinel
from companion_vision import companion_vision
from companion_web import companion_web

BASE_DIR = Path(__file__).resolve().parent
BREEZE_DIR = BASE_DIR / "bin" / "breeze-tts-2-v0.1.7-linux-x64-vulkan"
BREEZE_SERVER_BIN = BREEZE_DIR / "breeze-server"
VOICES_DIR = BASE_DIR / "voices"
MODELS_DIR = BASE_DIR / "models"
MODEL_GGUF = MODELS_DIR / "breeze-tts-2-q4_k.gguf"
REF_VOICE_WAV = MODELS_DIR / "ref_voice.wav"

BREEZE_PORT = 8137
BREEZE_URL = f"http://127.0.0.1:{BREEZE_PORT}"
OLLAMA_URL = "http://127.0.0.1:11434"
QWEN_MODEL = "qwen2.5:3b"
WEB_PORT = 7860
SAMPLE_RATE = 24000

VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "female_warm": {
        "label": "棉花奶甜音 (Voice Design)",
        "instruction": "一位像雪白棉花糖般软绵纯真的小女孩，声音软糯奶甜、轻柔乖巧，语速柔和带着治愈的软萌感，令人倍感放松。",
        "cfg_scale": 1.65,
        "seed": 318,
    },
    "female_lively": {
        "label": "蜜桃元气甜音 (Voice Design)",
        "instruction": "一位长着粉嫩长发的鬼马元气小少女，声音像水蜜桃气泡水一样清脆灵动、古灵精怪，语调轻快上扬、爱笑又俏皮。",
        "cfg_scale": 1.85,
        "seed": 616,
    },
    "male_calm": {
        "label": "软萌害羞少年 (Voice Design)",
        "instruction": "一位性格憨萌又有点小害羞的温柔小少年，声线清澈柔和带着微微的少年感，语气真诚腼腆、暖心又呆萌。",
        "cfg_scale": 1.55,
        "seed": 707,
    },
    "english_narrator": {
        "label": "森系灵动精灵 (Voice Design)",
        "instruction": "一位机灵博学、充满森林朝气的精灵小伙伴，声音清亮悦耳、灵动俏皮，吐字清晰自然，中英文都能流利生动地表达。",
        "cfg_scale": 1.6,
        "seed": 912,
    },
    "female_sunny": {
        "label": "暖阳治愈萌音 (Voice Design)",
        "instruction": "一位像午后阳光一样软萌治愈的小女孩，声音清亮甜暖，带着俏皮亲昵的撒娇感，让人倍感温暖。",
        "cfg_scale": 1.7,
        "seed": 520,
    },
    "female_cloud": {
        "label": "云朵软糖萌音 (Voice Design)",
        "instruction": "一位如冰川棉花糖般软糯灵动的少女音，语调轻盈梦幻，带着天真好奇与温柔甜笑。",
        "cfg_scale": 1.6,
        "seed": 888,
    },
}

VOICE_TO_PET: dict[str, tuple[str, str]] = {
    "female_warm": ("luma", "棉棉"),
    "female_lively": ("pipi", "桃桃"),
    "male_calm": ("taigo", "芋圆"),
    "english_narrator": ("auri", "芽芽"),
    "female_sunny": ("nuannuan", "暖暖"),
    "female_cloud": ("duoduo", "朵朵"),
}

PET_NAMES_MAP: dict[str, str] = {
    "luma": "棉棉",
    "pipi": "桃桃",
    "taigo": "芋圆",
    "auri": "芽芽",
    "nuannuan": "暖暖",
    "duoduo": "朵朵",
}


class LocalSpeakerPlayer:
    """Streams raw 24kHz 16-bit mono PCM audio directly to system speakers."""

    def __init__(self) -> None:
        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._stop_gen = 0
        self._play_until = 0.0
        self._cmd = self._detect_player_cmd()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    @staticmethod
    def _detect_player_cmd() -> list[str] | None:
        if shutil.which("paplay"):
            return ["paplay", "--raw", f"--rate={SAMPLE_RATE}", "--channels=1", "--format=s16le"]
        if shutil.which("ffplay"):
            return [
                "ffplay",
                "-f",
                "s16le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-",
            ]
        if shutil.which("aplay"):
            return ["aplay", "-r", str(SAMPLE_RATE), "-c", "1", "-f", "S16_LE", "-q", "-"]
        return None

    def _ensure_proc(self) -> subprocess.Popen[bytes] | None:
        if not self._cmd:
            return None
        if self._proc is None or self._proc.poll() is not None:
            try:
                self._proc = subprocess.Popen(
                    self._cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                self._proc = None
        return self._proc

    def play_pcm(self, pcm_bytes: bytes, gen_id: int) -> None:
        if not pcm_bytes or gen_id != self._stop_gen:
            return
        dur_s = len(pcm_bytes) / float(SAMPLE_RATE * 2)
        now = time.time()
        self._play_until = max(self._play_until, now) + dur_s
        self._q.put(pcm_bytes)

    def remaining_seconds(self) -> float:
        return max(0.0, self._play_until - time.time())

    def is_playing_or_cooldown(self, cooldown_s: float = 1.5) -> bool:
        return time.time() < (self._play_until + cooldown_s)

    def mark_busy(self, dur_s: float = 5.0) -> None:
        """提前预占扬声器（在文本思考或TTS首包未到达前），防止其他主动播报并发插队"""
        with self._lock:
            now = time.time()
            self._play_until = max(self._play_until, now + dur_s)

    def stop(self) -> int:
        with self._lock:
            self._stop_gen += 1
            self._play_until = 0.0
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                self._proc = None
            return self._stop_gen

    @property
    def current_gen(self) -> int:
        return self._stop_gen

    def _run_loop(self) -> None:
        while True:
            chunk = self._q.get()
            if chunk is None:
                break
            with self._lock:
                proc = self._ensure_proc()
                if proc and proc.stdin:
                    try:
                        proc.stdin.write(chunk)
                        proc.stdin.flush()
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        self._proc = None


speaker_player = LocalSpeakerPlayer()
companion_asr.set_speaker_busy_checker(lambda: speaker_player.is_playing_or_cooldown(1.5))
breeze_proc: subprocess.Popen[bytes] | None = None


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def is_breeze_ready() -> bool:
    try:
        req = urllib.request.Request(f"{BREEZE_URL}/health")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode())
            return data.get("status") == "ok"
    except Exception:
        return False


def ensure_breeze_server() -> bool:
    global breeze_proc
    if is_breeze_ready():
        return True
    if not MODEL_GGUF.is_file():
        return False
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{BREEZE_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        str(BREEZE_SERVER_BIN),
        str(MODEL_GGUF),
        "--host",
        "127.0.0.1",
        "--port",
        str(BREEZE_PORT),
        "--voices-dir",
        str(VOICES_DIR),
        "--cors",
    ]
    log_path = BASE_DIR / "breeze_server.log"
    log_file = open(log_path, "ab")
    breeze_proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    for _ in range(90):
        if is_breeze_ready():
            return True
        if breeze_proc.poll() is not None:
            break
        time.sleep(0.5)
    return is_breeze_ready()


def get_gpu_status() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0],
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "gpu_util": int(parts[3]),
            "temp_c": int(parts[4]),
            "breeze_ready": is_breeze_ready(),
        }
    except Exception as exc:
        return {
            "name": "NVIDIA RTX 4060",
            "vram_used_mb": 0,
            "vram_total_mb": 8188,
            "gpu_util": 0,
            "temp_c": 0,
            "breeze_ready": is_breeze_ready(),
            "error": str(exc),
        }


def clean_text_for_tts(text: str) -> str:
    # Remove markdown formatting that shouldn't be spoken aloud
    text = re.sub(r"\*\*|__|\*|`|#+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text.strip()


def stream_tts_pcm(
    text: str,
    instruction: str,
    cfg_scale: float = 1.5,
    seed: int = 42,
) -> Generator[bytes, None, None]:
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return
    payload = {
        "model": "tts-1",
        "input": cleaned,
        "instructions": instruction,
        "response_format": "pcm",
        "cfg_scale": cfg_scale,
        "seed": seed,
    }
    req = urllib.request.Request(
        f"{BREEZE_URL}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    tail = b""
    with urllib.request.urlopen(req, timeout=60) as resp:
        while True:
            chunk = resp.read(8192)
            if not chunk:
                break
            data = tail + chunk
            even_len = (len(data) // 2) * 2
            tail = data[even_len:]
            if even_len > 0:
                yield data[:even_len]


# 实时视线哨兵动态迎候生成与回调注册移至下方 stream_qwen_tokens 之后定义


LLM_CONFIG_PATH = BASE_DIR / "data" / "llm_config.json"

CLOUD_PROVIDERS: dict[str, dict[str, Any]] = {
    "siliconflow": {
        "name": "硅基流动 SiliconFlow（强推 · 含永久免费模型）",
        "base_url": "https://api.siliconflow.cn/v1",
        "key_url": "https://cloud.siliconflow.cn/account/ak",
        "models": [
            {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "千问 Qwen2.5-7B（永久免费 0元 · 推荐）"},
            {"id": "THUDM/glm-4-9b-chat", "label": "智谱 GLM-4-9B（永久免费 0元）"},
            {"id": "deepseek-ai/DeepSeek-V3", "label": "DeepSeek-V3 满血版（671B 旗舰 · 量大管饱）"},
            {"id": "Qwen/Qwen2.5-72B-Instruct", "label": "千问 Qwen2.5-72B（720亿参数旗舰）"},
        ],
    },
    "zhipu": {
        "name": "智谱开放平台 BigModel（官方永久免费 GLM-4-Flash）",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_url": "https://open.bigmodel.cn/usercenter/apikeys",
        "models": [
            {"id": "glm-4-flash", "label": "GLM-4-Flash（官方永久免费 0元 · 极速）"},
            {"id": "glm-4-air", "label": "GLM-4-Air（高性价比主力模型）"},
            {"id": "glm-4-plus", "label": "GLM-4-Plus（智谱满血旗舰）"},
        ],
    },
    "deepseek": {
        "name": "DeepSeek 官方开放平台（满血 V3 · 白菜价量大管饱）",
        "base_url": "https://api.deepseek.com/v1",
        "key_url": "https://platform.deepseek.com/api_keys",
        "models": [
            {"id": "deepseek-chat", "label": "DeepSeek-V3 满血对话模型（deepseek-chat）"},
        ],
    },
    "aliyun": {
        "name": "阿里云百炼 · 通义千问官方（开通送百万免费额度）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_url": "https://bailian.console.aliyun.com/?apiKey=1",
        "models": [
            {"id": "qwen-plus", "label": "通义千问 Plus（均衡高智商 · 推荐）"},
            {"id": "qwen-turbo", "label": "通义千问 Turbo（闪电响应 · 极低价）"},
            {"id": "qwen-max", "label": "通义千问 Max（千问最强满血旗舰）"},
        ],
    },
    "custom": {
        "name": "自定义 OpenAI 兼容接口（任意中转站 / 火山豆包 / Kimi）",
        "base_url": "https://api.openai.com/v1",
        "key_url": "",
        "models": [],
    },
}


def load_llm_config() -> dict[str, Any]:
    default_cfg: dict[str, Any] = {
        "mode": "local",
        "local_model": QWEN_MODEL,
        "provider": "siliconflow",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": "",
        "cloud_model": "Qwen/Qwen2.5-7B-Instruct",
        "saved_keys": {},
    }
    if LLM_CONFIG_PATH.is_file():
        try:
            saved = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                default_cfg.update(saved)
        except Exception:
            pass
    return default_cfg


def save_llm_config(new_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = load_llm_config()
    for k in ("mode", "local_model", "provider", "base_url", "api_key", "cloud_model"):
        if k in new_cfg and new_cfg[k] is not None:
            cfg[k] = str(new_cfg[k]).strip()
    saved_keys = cfg.get("saved_keys") if isinstance(cfg.get("saved_keys"), dict) else {}
    prov = cfg.get("provider") or "siliconflow"
    if "api_key" in new_cfg:
        saved_keys[prov] = cfg["api_key"]
    elif prov in saved_keys and not cfg.get("api_key"):
        cfg["api_key"] = saved_keys[prov]
    cfg["saved_keys"] = saved_keys
    LLM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def stream_qwen_tokens(
    messages: list[dict[str, str]],
    keep_alive: str | int = "30m",
) -> Generator[str, None, None]:
    cfg = load_llm_config()
    if cfg.get("mode") == "cloud":
        api_key = (cfg.get("api_key") or "").strip()
        base_url = (cfg.get("base_url") or "https://api.siliconflow.cn/v1").strip().rstrip("/")
        cloud_model = (cfg.get("cloud_model") or "Qwen/Qwen2.5-7B-Instruct").strip()
        if not api_key:
            yield "您当前选择了云端大模型模式，但尚未填写 API 密钥。请点击右上角【设置】填入密钥，或切回【本地模型】模式。"
            return

        payload = {
            "model": cloud_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 512,
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
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            token = delta.get("content") or ""
                            if token:
                                yield token
                    except Exception:
                        continue
        except urllib.error.HTTPError as he:
            err_body = he.read().decode("utf-8", errors="ignore")[:160]
            yield f"云端模型调用失败（HTTP {he.code}），请检查 API 密钥或模型名称是否正确：{err_body}"
        except Exception as exc:
            yield f"云端模型连接异常：{exc}"
        return

    # 默认：本地 Ollama 模型流式生成
    local_model = (cfg.get("local_model") or QWEN_MODEL).strip()
    payload = {
        "model": local_model,
        "messages": messages,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": 4096,
            "temperature": 0.7,
        },
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                yield token
            if data.get("done"):
                break


def split_sentences_incremental(
    token_iter: Generator[str, None, None],
    on_token: Any = None,
) -> Generator[str, None, None]:
    buf = ""
    strong_punct = set("。！？；!?;:\n")
    weak_punct = set("，,、")

    for tok in token_iter:
        if on_token:
            on_token(tok)
        buf += tok
        while True:
            split_idx = -1
            for i, ch in enumerate(buf):
                if ch in strong_punct and i >= 2:
                    split_idx = i + 1
                    break
                if ch in weak_punct and i >= 16:
                    split_idx = i + 1
                    break
            if split_idx != -1:
                segment = buf[:split_idx].strip()
                buf = buf[split_idx:]
                if segment and re.search(r"[\w\u4e00-\u9fff]", segment):
                    yield segment
            else:
                break

    rem = buf.strip()
    if rem and re.search(r"[\w\u4e00-\u9fff]", rem):
        yield rem


VOICE_TO_PET: dict[str, tuple[str, str]] = {
    "female_warm": ("luma", "棉棉"),
    "female_lively": ("pipi", "桃桃"),
    "male_calm": ("taigo", "芋圆"),
    "english_narrator": ("auri", "芽芽"),
    "female_sunny": ("nuannuan", "暖暖"),
    "female_cloud": ("duoduo", "朵朵"),
}
PET_NAMES_MAP = {
    "luma": "棉棉",
    "pipi": "桃桃",
    "taigo": "芋圆",
    "auri": "芽芽",
    "nuannuan": "暖暖",
    "duoduo": "朵朵",
}
PET_TRAITS = {
    "luma": "你是一只雪白棉花糖般软绵纯真、乖巧治愈的毛绒小玩偶，说话轻柔甜糯。",
    "pipi": "你是一只粉发飘逸、鬼马古灵精怪又元气满满的毛绒小玩偶，说话俏皮灵动。",
    "taigo": "你是一只香芋紫色、性格憨萌有点小害羞但暖心真诚的小少年毛绒玩偶。",
    "auri": "你是一只苔藓森绿色的森林精灵毛绒玩偶，机灵博学、充满朝气，中英文都很擅长。",
    "nuannuan": "你是一只向日葵暖黄色、像阳光一样热情甜暖、爱撒娇治愈的毛绒小玩偶。",
    "duoduo": "你是一只冰川云朵蓝、像棉花糖一样软糯梦幻又天真好奇的毛绒小玩偶。",
}

# =========================================================================
# 实时视线哨兵：大模型动态自思考迎候语生成引擎（绝不机械重复一字不变）
# =========================================================================
_recent_sentinel_greeting_texts: list[str] = []


def generate_dynamic_sentinel_greeting(
    identity: str,
    label: str,
    score: float,
    pet_name: str,
    speaker_pet: str,
    face_analysis: dict[str, Any] | None = None,
) -> str:
    """结合真实电脑时间、伙伴人设与大模型自由思考，生成每次都不同、鲜活生动的主动迎候问候语"""
    global _recent_sentinel_greeting_texts
    now = datetime.now()
    weekdays_all = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday_cn = weekdays_all[now.weekday()]
    hour = now.hour

    if 5 <= hour < 9:
        period = "清晨"
        time_topic = "早晨新的一天开始，提醒吃温暖早餐、打起精神"
    elif 9 <= hour < 11:
        period = "上午"
        time_topic = "上午工作或学习时光，陪伴、专注与适时喝水"
    elif 11 <= hour < 14:
        period = "中午"
        time_topic = "午餐时间与午休，关心按时吃饭、小憩休息"
    elif 14 <= hour < 18:
        period = "下午"
        time_topic = "下午时光，提神、轻松、伸个懒腰"
    elif 18 <= hour < 22:
        period = "傍晚"
        time_topic = "下班放学回家，晚饭与温馨放松陪伴"
    elif 22 <= hour < 24:
        period = "夜间"
        time_topic = "夜深了，注意身体、早点休息少熬夜"
    else:
        period = "凌晨"
        time_topic = "深夜凌晨，轻声关怀，催促主人快快睡觉"

    # 提取主人姓名槽位（如果长期记忆已记录）
    master_name = None
    try:
        mem_ctx = companion_memory.get_context_for_chat("", agent_id=speaker_pet)
        master_name = mem_ctx.get("master_name")
    except Exception:
        pass

    is_both = (identity == "both")
    is_followup = bool(face_analysis and face_analysis.get("followup_after_stranger"))

    if is_both:
        target_desc = f"主人（名字：{master_name}）与一位新朋友一同现身" if master_name else "主人与一位新朋友一同现身"
    elif is_followup:
        target_desc = f"主人（名字：{master_name}）在刚向新朋友问好后现身" if master_name else "主人在刚向新朋友问好后现身"
    elif identity == "master":
        target_desc = f"主人（名字：{master_name}）" if master_name else "主人"
    elif identity == "friend":
        target_desc = f"好友【{label}】"
    else:
        target_desc = "新朋友"

    pet_trait = PET_TRAITS.get(speaker_pet, "软萌可爱，奶声奶气，充满关怀与陪伴感")

    # 尝试调用大模型自由思考生成台词（限时极速生成）
    candidate_text = None
    try:
        if is_both:
            prompt_mission = f"你在摄像头前同时捕捉到主人与新朋友现身。请用温暖亲切的语气，同时问候主人并热情礼貌欢迎新朋友（严格控制在18到28个字以内）。"
        elif is_followup:
            prompt_mission = f"你刚才刚跟新朋友打完招呼，现在主人也现身在镜头前。请用惊喜温暖的语气向主人打招呼问候（严格控制在12到25个字以内）。"
        else:
            prompt_mission = f"你在摄像头前刚刚捕捉到【{target_desc}】现身。请根据当前真实时段（{period}，主题：{time_topic}）以及你的毛绒玩偶个性，主动说一句自然、鲜活、温暖的打招呼迎候语（严格控制在12到25个字以内）。"

        prompt_sys = (
            f"你是桌面伴生毛绒玩偶【{pet_name}】（性格特征：{pet_trait}）。"
            f"当前电脑本地真实硬件时间：{now.year}年{now.month}月{now.day}日 {weekday_cn} {period} {now.strftime('%H:%M')}。"
            f"{prompt_mission}\n"
            "【规则】：\n"
            "1. 适合语音朗读，口语亲切，语调欢快自然。\n"
            "2. 每次都要有新鲜感，自由思考发挥，绝对禁止千篇一律或使用固定套话！\n"
            "3. 直接输出你要说的那句话，禁止带有括号描写、动作描写、引号或解释说明。"
        )
        tokens: list[str] = []
        for tok in stream_qwen_tokens([
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": "请迎候说一句："}
        ]):
            tokens.append(tok)
            if len("".join(tokens)) > 35:
                break
        candidate_text = "".join(tokens).strip()
    except Exception as exc:
        print(f"[Sentinel] LLM dynamic greeting fallback: {exc}")

    # 清洗 LLM 生成的文本
    if candidate_text:
        candidate_text = re.sub(r"[\"\'“”‘’]", "", candidate_text)
        candidate_text = re.sub(r"[（\(].*?[）\)]", "", candidate_text)
        candidate_text = re.sub(r"[*#]", "", candidate_text).strip()
        if len(candidate_text) > 35:
            candidate_text = candidate_text[:30] + "哦！"

    # 如果大模型未返回，或者生成内容与近期完全重复，使用高维度动态自思考候选库（带时段与随机变量，杜绝重样）
    if not candidate_text or candidate_text in _recent_sentinel_greeting_texts[-4:]:
        import random
        m_title = f"{master_name}主人" if master_name else "主人"
        if is_both:
            pool = [
                f"{m_title}，你带新朋友来啦！新朋友你好呀，我是{pet_name}，欢迎来玩！",
                f"看到{m_title}和新朋友一起在镜头前啦，新朋友你好，{m_title}好呀！",
                f"{m_title}你回来啦，旁边这位新朋友也你好呀！很高兴认识你！",
                f"哇，镜头前有{m_title}和新朋友两位呢，新朋友你好呀，欢迎来找主人玩！",
            ]
        elif is_followup:
            pool = [
                f"{m_title}，你也来啦！刚正跟新朋友打招呼呢，欢迎回来呀！",
                f"{m_title}你回来啦！刚才看到新朋友，现在看到你，今天好热闹呀！",
                f"{m_title}好呀！刚才正在跟新朋友打招呼呢，有需要随时叫{pet_name}！",
                f"欢迎{m_title}！刚跟新朋友问好，看到你也在，{pet_name}太开心啦！",
            ]
        elif identity == "master":
            noon_list = [
                f"{m_title}，中午好呀！忙碌了半天，快去吃顿热乎的午饭吧！",
                f"捕捉到{m_title}现身啦！现在正好是饭点，今天午饭吃什么好吃的呀？",
                f"{m_title}辛苦啦！午后记得稍微闭目养神休息一下，{pet_name}在这守着你呢！",
                f"又在镜头前看到{m_title}啦！肚子饿不饿呀，要按时吃饭保重身体哦！",
                f"中午的太阳暖洋洋的，{m_title}今天上午顺利吗？{pet_name}时刻陪着你！",
                f"{m_title}，工作再忙也要吃好午饭哦，今天也要元气满满呢！",
            ]
            morning_list = [
                f"{m_title}，早安呀！今天又是元气满满的一天，我们一起加油！",
                f"清晨捕捉到{m_title}的身影啦！昨晚睡得好吗？记得喝杯温水哦！",
                f"早晨好，{m_title}！今天的风很温柔，{pet_name}已经元气待命啦！",
                f"新的一天开始啦！看到{m_title}神采奕奕的样子，{pet_name}心情超好！",
                f"{m_title}早呀！新的一天开始，有任何事随时吩咐{pet_name}！",
            ]
            afternoon_list = [
                f"{m_title}，下午好呀！学了这么久累不累，起来伸个懒腰活动一下吧！",
                f"捕捉到{m_title}！下午茶时间到啦，来杯水润润喉咙提提神吧！",
                f"{m_title}你坐回电脑前啦！看到你，{pet_name}立刻精神百倍！",
                f"下午的阳光正好，{m_title}处理任务辛苦啦，有需要随时叫{pet_name}哦！",
                f"伸个懒腰喝口温水吧，{m_title}，{pet_name}一直在旁边默默陪着你呢！",
            ]
            evening_list = [
                f"{m_title}，傍晚好呀！忙碌了一整天，今晚好好放松犒劳一下自己吧！",
                f"欢迎回家，{m_title}！看到你平安归来，{pet_name}心里特别踏实！",
                f"{m_title}今天辛苦啦！晚饭吃饱饱了吗？今晚{pet_name}陪你聊聊天！",
                f"夜幕降临啦，{m_title}！把白天的疲惫放下，享受属于你的惬意时光吧！",
                f"哈喽{m_title}！今天过得怎么样呀，快坐下来跟{pet_name}歇一歇！",
            ]
            night_list = [
                f"{m_title}，这么晚还在电脑前呀，别太累了，要注意早点休息哦！",
                f"夜深啦，{m_title}！眼睛酸不酸呀，再忙也要记得早些睡个美容觉呢！",
                f"时间不早啦，{m_title}！虽然{pet_name}想一直陪着你，但更想你身体健康！",
                f"深夜守望中捕捉到{m_title}，快快收工钻进被窝吧，明天还要元气满满呢！",
                f"小夜灯亮起啦，{m_title}，今天任务完成得很棒，早点休息吧！",
            ]
            if 5 <= hour < 9:
                pool = morning_list
            elif 9 <= hour < 11:
                pool = morning_list + afternoon_list
            elif 11 <= hour < 14:
                pool = noon_list
            elif 14 <= hour < 18:
                pool = afternoon_list
            elif 18 <= hour < 22:
                pool = evening_list
            else:
                pool = night_list
        elif identity == "friend":
            pool = [
                f"检测到熟悉的朋友【{label}】在镜头前，你好呀！欢迎来找主人玩！",
                f"哈喽【{label}】！很高兴又在镜头前看到你，今天心情怎么样呀？",
                f"【{label}】你来啦！欢迎光临，需要我帮你转告主人什么事情吗？",
                f"看到【{label}】在镜头前啦，今天有什么开心的事想和大家分享吗？",
            ]
        else:
            pool = [
                f"检测到新朋友在镜头前，你好呀！我是桌面伴侣{pet_name}，很高兴认识你！",
                f"哈喽新朋友！欢迎来到主人的电脑前，我是{pet_name}，有什么我可以帮你的吗？",
                f"视线捕捉到新面孔啦，你好呀新朋友！今天过得开心吗？",
                f"新朋友你好呀！我是陪伴主人的{pet_name}，祝你今天有份好心情！",
            ]

        fresh_choices = [c for c in pool if c not in _recent_sentinel_greeting_texts[-5:]]
        candidate_text = random.choice(fresh_choices or pool)

    _recent_sentinel_greeting_texts.append(candidate_text)
    if len(_recent_sentinel_greeting_texts) > 20:
        _recent_sentinel_greeting_texts = _recent_sentinel_greeting_texts[-20:]

    return candidate_text


_greeting_exec_lock = threading.Lock()


def handle_sentinel_greeting(
    identity: str,
    label: str,
    score: float,
    face_analysis: dict[str, Any] | None,
) -> None:
    """实时视线哨兵识别到人脸时的主动问候执行逻辑（防打断 + 完整播报 + 顺畅接续）"""
    with _greeting_exec_lock:
        companion_sentinel.set_greeting_in_progress(True)
        try:
            # 优雅等待此前正在播放的音频（如闹钟或对话）完全读完，坚决不中途掐断
            t_wait = time.time()
            while speaker_player.remaining_seconds() > 0.05 and (time.time() - t_wait < 15.0):
                time.sleep(0.1)

            # 排空后预占扬声器状态（防并发与LLM思考/TTS合成延迟）
            speaker_player.mark_busy(6.0)

            speaker_pet = companion_mobile.active_pet_id or "female_warm"
            voice_key = speaker_pet
            if speaker_pet in PET_NAMES_MAP:
                for vk, (pid, _) in VOICE_TO_PET.items():
                    if pid == speaker_pet:
                        voice_key = vk
                        break
                pet_name = PET_NAMES_MAP[speaker_pet]
            else:
                _, pet_name = VOICE_TO_PET.get(voice_key, ("luma", "棉棉"))

            # 动态智能思考生成每次都不同、鲜活生动的打招呼台词（绝不机械重复一个字都不变）
            text = generate_dynamic_sentinel_greeting(
                identity=identity,
                label=label,
                score=score,
                pet_name=pet_name,
                speaker_pet=speaker_pet,
                face_analysis=face_analysis,
            )

            preset = VOICE_PRESETS.get(voice_key, VOICE_PRESETS["female_warm"])
            instruction = preset["instruction"]
            cfg_scale = float(preset["cfg_scale"])
            seed = int(preset["seed"])

            gen_id = speaker_player.stop()
            all_pcm = bytearray()
            for pcm_chunk in stream_tts_pcm(text, instruction, cfg_scale=cfg_scale, seed=seed):
                all_pcm.extend(pcm_chunk)
                speaker_player.play_pcm(pcm_chunk, gen_id)

            companion_asr.record_pet_speech(text)
            wav_bytes = pcm_to_wav(bytes(all_pcm), SAMPLE_RATE)
            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
            turn_id = f"sentinel_{int(time.time()*1000)}"

            ev = {
                "type": "chat_sentinel_greeting",
                "turn_id": turn_id,
                "pet_id": speaker_pet,
                "pet_name": pet_name,
                "user_text": f"【实时视线感知 · 识别到{label}】",
                "assistant_text": text,
                "identity": identity,
                "label": label,
                "score": score,
                "wav_b64": wav_b64,
                "duration_s": round(len(all_pcm) / 2 / SAMPLE_RATE, 2),
                "timestamp": time.time(),
            }
            companion_sentinel.record_greeting(ev)
            companion_mobile.broadcast(ev)

            # 等待本句语音在物理扬声器硬件上完整播完，坚决不中途掐断
            t_wait_play = time.time()
            while speaker_player.remaining_seconds() > 0.05 and (time.time() - t_wait_play < 25.0):
                time.sleep(0.1)

            # 留出 0.8s 自然呼吸停顿，给对话以生命呼吸感，绝不急躁抢麦
            time.sleep(0.8)

        except Exception as exc:
            print(f"[Sentinel] Greeting handler error: {exc}")
        finally:
            companion_sentinel.set_greeting_in_progress(False)


GUARD_WARNING_POOL = [
    "警报！检测到陌生人闯入，请问你是谁？主人不在，请立刻离开！",
    "警告！检测到未授权人员靠近！这里是主人专属空间，请退后！",
    "警报！发现陌生人进入警戒区域，请说明来意！主人已收到实时画面！",
    "警告！未经允许请勿靠近电脑，主人正在注视这里，请立刻离开！",
]


def handle_sentinel_guard_alert(
    score: float,
    face_analysis: dict[str, Any] | None,
    snapshot_data_uri: str,
) -> None:
    """警惕模式陌生人闯入警报执行逻辑（持续警报语音提醒、陌生人抓拍照片与警报双端实时推送）"""
    try:
        speaker_pet = companion_mobile.active_pet_id or "female_warm"
        voice_key = speaker_pet
        if speaker_pet in PET_NAMES_MAP:
            for vk, (pid, _) in VOICE_TO_PET.items():
                if pid == speaker_pet:
                    voice_key = vk
                    break
            pet_name = PET_NAMES_MAP[speaker_pet]
        else:
            _, pet_name = VOICE_TO_PET.get(voice_key, ("luma", "棉棉"))

        guard_cfg = companion_sentinel.get_guard_config()
        alert_count = guard_cfg.get("alert_count", 3)
        alert_duration_s = guard_cfg.get("alert_duration_s", 5)

        # 选取警报警告口语
        warning_text = random.choice(GUARD_WARNING_POOL)

        preset = VOICE_PRESETS.get(voice_key, VOICE_PRESETS["female_warm"])
        instruction = "用严肃警惕、清晰果断的语气发出警报警告"
        cfg_scale = float(preset["cfg_scale"])
        seed = int(preset["seed"])

        all_pcm = bytearray()
        for pcm_chunk in stream_tts_pcm(warning_text, instruction, cfg_scale=cfg_scale, seed=seed):
            all_pcm.extend(pcm_chunk)

        pcm_bytes = bytes(all_pcm)
        wav_bytes = pcm_to_wav(pcm_bytes, SAMPLE_RATE)
        wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

        turn_id = f"guard_alert_{int(time.time()*1000)}"
        now_dt = datetime.now()
        time_str = now_dt.strftime("%H:%M:%S")

        ev = {
            "type": "sentinel_guard_alert",
            "turn_id": turn_id,
            "pet_id": speaker_pet,
            "pet_name": pet_name,
            "alert_title": "警惕模式 · 陌生人闯入警报",
            "alert_text": f"在警惕时段（{guard_cfg.get('start_time', '10:00')}~{guard_cfg.get('end_time', '20:00')}）检测到陌生人员闯入！",
            "assistant_text": warning_text,
            "identity": "stranger",
            "label": "陌生人",
            "score": round(score, 2),
            "image_data_uri": snapshot_data_uri,
            "wav_b64": wav_b64,
            "alert_count": alert_count,
            "alert_duration_s": alert_duration_s,
            "timestamp": time.time(),
            "time_str": time_str,
        }
        companion_sentinel.record_greeting(ev)
        companion_mobile.broadcast(ev)

        # 启动语音循环警报播报线程：警报 alert_count 次，每次持续/间隔 alert_duration_s 秒语音提醒
        def _alert_speech_loop():
            try:
                speech_dur = len(pcm_bytes) / 2 / SAMPLE_RATE
                for idx in range(alert_count):
                    if companion_sentinel._cancel_alert_event.is_set():
                        break
                    gen_id = speaker_player.stop()
                    speaker_player.play_pcm(pcm_bytes, gen_id)

                    # 等待该轮语音播完（支持被主人现身实时打断）
                    t_end = time.time() + speech_dur
                    while time.time() < t_end:
                        if companion_sentinel._cancel_alert_event.is_set():
                            speaker_player.stop()
                            break
                        time.sleep(0.1)

                    # 警报间隔秒数冷却（若尚未达到设定的总次数）
                    if idx < alert_count - 1 and not companion_sentinel._cancel_alert_event.is_set():
                        t_gap = time.time() + alert_duration_s
                        while time.time() < t_gap:
                            if companion_sentinel._cancel_alert_event.is_set():
                                break
                            time.sleep(0.1)
            finally:
                companion_sentinel._guard_alerting = False
                companion_sentinel._cancel_alert_event.clear()

        threading.Thread(target=_alert_speech_loop, daemon=True, name="GuardAlertSpeechLoop").start()

    except Exception as exc:
        print(f"[Sentinel] Guard alert handler error: {exc}")
        companion_sentinel._guard_alerting = False


companion_sentinel.set_busy_checkers(
    speaker_checker=lambda: (not is_breeze_ready()) or speaker_player.is_playing_or_cooldown(1.5),
    mic_checker=lambda: bool(getattr(companion_asr, "_recording_pc_mic", False)),
)
companion_sentinel.set_greeting_callback(handle_sentinel_greeting)
companion_sentinel.set_guard_alert_callback(handle_sentinel_guard_alert)


def handle_alarm_briefing(alarm: dict[str, Any], stop_reason: str) -> None:
    """闹钟关停后（手动点关停或5分钟超时）触发的今日天气/新闻晨报或知识库日程备忘智能语音播报"""
    def _run():
        try:
            speaker_pet = companion_mobile.active_pet_id or "female_warm"
            voice_key = speaker_pet
            if speaker_pet in PET_NAMES_MAP:
                for vk, (pid, _) in VOICE_TO_PET.items():
                    if pid == speaker_pet:
                        voice_key = vk
                        break
                pet_name = PET_NAMES_MAP[speaker_pet]
            else:
                _, pet_name = VOICE_TO_PET.get(voice_key, ("luma", "棉棉"))

            alarm_type = alarm.get("type", "morning")
            alarm_label = alarm.get("label", "日程提醒")
            user_memo = (alarm.get("content") or "").strip()
            now_dt = datetime.now()
            time_hm = now_dt.strftime("%H:%M")

            weather_res: dict[str, Any] | None = None
            news_items: list[dict[str, str]] = []

            if alarm_type == "morning":
                # 拉取福州实时天气
                weather_res = companion_web.fetch_weather("福州")
                weather_desc = weather_res.get("weather", "多云")
                temp_now = weather_res.get("temp_now", "22")
                t_min = weather_res.get("temp_min", "18")
                t_max = weather_res.get("temp_max", "24")
                rain_prob = weather_res.get("rain_prob", 0)

                weather_oral = f"今天福州天气{weather_desc}，当前气温{temp_now}度，全天气温{t_min}到{t_max}度。"
                if rain_prob and rain_prob >= 40:
                    weather_oral += "今天可能有降雨，出门请记得随身带把雨伞哦。"
                else:
                    weather_oral += "今天基本无雨，很适合出门活动。"

                # 拉取今日实时热点新闻
                news_res = companion_web.fetch_daily_hot_news(limit=3)
                news_items = news_res.get("items", [])
                news_oral_parts = []
                for idx, it in enumerate(news_items[:3], 1):
                    t_clean = it.get("title", "").strip()
                    news_oral_parts.append(f"第{idx}条，{t_clean}")
                if news_oral_parts:
                    news_oral = "接下来为您播报今日全网实时热点新闻：" + "；".join(news_oral_parts) + "。"
                else:
                    news_oral = "今天的全网实时热点新闻正在持续更新中。"

                memo_oral = f"另外，主人在早晨知识库中备忘的提醒：{user_memo}。" if user_memo else ""
                timeout_hint = "闹钟已响满5分钟自动关停，" if stop_reason == "timeout" else ""

                spoken_text = (
                    f"主人，早上好呀！{timeout_hint}新的一天开始了，快伸个懒腰准备起床啦！{pet_name}为您播报今天的晨间全息简报：\n"
                    f"{weather_oral}\n"
                    f"{news_oral}\n"
                    f"{memo_oral}\n"
                    f"新的一天，祝主人精力充沛，心情愉快！"
                )
                card_title = "早晨唤醒 · 今日晨间全息简报"
            else:
                # 知识库日程备忘提醒
                memo_core = user_memo if user_memo else "请注意按时完成今日规划的重点工作。"
                timeout_hint = "闹钟已响满5分钟自动关停，" if stop_reason == "timeout" else ""
                spoken_text = (
                    f"主人，{timeout_hint}现在是{time_hm}，您设置的【{alarm_label}】日程备忘提醒时间到啦！"
                    f"根据您在知识库中添加的内容：{memo_core}。"
                    f"请主人注意及时处理哦，{pet_name}随时在您身边协助您！"
                )
                card_title = f"知识库日程备忘提醒 · {alarm_label}"

            # 使用 Breeze TTS 2 合成语音
            preset = VOICE_PRESETS.get(voice_key, VOICE_PRESETS["female_warm"])
            instruction = "用欢快温暖、亲切温柔的语气播报"
            cfg_scale = float(preset["cfg_scale"])
            seed = int(preset["seed"])

            gen_id = speaker_player.stop()
            all_pcm = bytearray()
            for pcm_chunk in stream_tts_pcm(spoken_text, instruction, cfg_scale=cfg_scale, seed=seed):
                all_pcm.extend(pcm_chunk)
                speaker_player.play_pcm(pcm_chunk, gen_id)

            companion_asr.record_pet_speech(spoken_text)
            wav_bytes = pcm_to_wav(bytes(all_pcm), SAMPLE_RATE)
            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
            turn_id = f"alarm_briefing_{int(time.time() * 1000)}"

            ev = {
                "type": "chat_alarm_briefing",
                "turn_id": turn_id,
                "pet_id": speaker_pet,
                "pet_name": pet_name,
                "alarm_id": alarm.get("id"),
                "alarm_label": alarm_label,
                "alarm_type": alarm_type,
                "card_title": card_title,
                "stop_reason": stop_reason,
                "assistant_text": spoken_text,
                "weather": weather_res,
                "news": news_items,
                "memo_content": user_memo,
                "wav_b64": wav_b64,
                "duration_s": round(len(all_pcm) / 2 / SAMPLE_RATE, 2),
                "timestamp": time.time(),
                "time_str": now_dt.strftime("%H:%M:%S"),
            }
            companion_sentinel.record_greeting(ev)
            companion_mobile.broadcast(ev)
        except Exception as exc:
            print(f"[Alarm] Briefing runner error: {exc}")

    threading.Thread(target=_run, daemon=True, name="AlarmBriefingThread").start()


companion_alarm.speaker_player = speaker_player
companion_alarm.broadcast_callback = companion_mobile.broadcast
companion_alarm.trigger_briefing_callback = handle_alarm_briefing


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=ensure_breeze_server, daemon=True).start()
    threading.Thread(target=companion_asr.ensure_loaded, daemon=True).start()
    companion_mobile.start_tunnel_async()
    yield
    if breeze_proc and breeze_proc.poll() is None:
        try:
            breeze_proc.terminate()
        except Exception:
            pass


app = FastAPI(title="Qwen 3B + Breeze TTS 2 实时语音对话系统", lifespan=lifespan)
(BASE_DIR / "static").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse(get_gpu_status())


@app.post("/api/stop")
async def api_stop() -> JSONResponse:
    speaker_player.stop()
    return JSONResponse({"status": "stopped"})


@app.get("/api/llm_config")
async def api_get_llm_config() -> JSONResponse:
    cfg = load_llm_config()
    return JSONResponse({
        "success": True,
        "config": cfg,
        "providers": CLOUD_PROVIDERS,
    })


@app.post("/api/llm_config")
async def api_save_llm_config(request: Request) -> JSONResponse:
    body = await request.json()
    cfg = save_llm_config(body)
    return JSONResponse({
        "success": True,
        "config": cfg,
    })


@app.post("/api/llm_test")
async def api_test_llm(request: Request) -> JSONResponse:
    body = await request.json()
    mode = (body.get("mode") or "local").strip()
    t0 = time.time()
    try:
        if mode == "cloud":
            api_key = (body.get("api_key") or "").strip()
            base_url = (body.get("base_url") or "https://api.siliconflow.cn/v1").strip().rstrip("/")
            model_name = (body.get("cloud_model") or "Qwen/Qwen2.5-7B-Instruct").strip()
            if not api_key:
                return JSONResponse({"success": False, "error": "请先粘贴填写您的 API 密钥（通常以 sk- 开头）"})
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "请用十个字以内简短回复：云端大脑连接成功"}],
                "max_tokens": 32,
                "temperature": 0.3,
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
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                ms = int((time.time() - t0) * 1000)
                return JSONResponse({
                    "success": True,
                    "reply": reply or "云端模型响应正常",
                    "latency_ms": ms,
                    "model": model_name,
                })
        else:
            local_model = (body.get("local_model") or QWEN_MODEL).strip()
            payload = {
                "model": local_model,
                "messages": [{"role": "user", "content": "请回复：本地大脑连接正常"}],
                "stream": False,
            }
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data.get("message", {}).get("content", "").strip()
                ms = int((time.time() - t0) * 1000)
                return JSONResponse({
                    "success": True,
                    "reply": reply or "本地 Ollama 响应正常",
                    "latency_ms": ms,
                    "model": local_model,
                })
    except urllib.error.HTTPError as he:
        err_msg = he.read().decode("utf-8", errors="ignore")[:180]
        return JSONResponse({"success": False, "error": f"HTTP {he.code} 错误：{err_msg}"})
    except Exception as exc:
        return JSONResponse({"success": False, "error": f"连接失败：{exc}"})


@app.get("/api/memories")
async def api_get_memories() -> JSONResponse:
    items = companion_memory.get_all()
    return JSONResponse({"success": True, "count": len(items), "memories": items})


@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: str) -> JSONResponse:
    ok = companion_memory.delete(memory_id)
    return JSONResponse({"success": ok})


@app.post("/api/memories/clear")
async def api_clear_memories() -> JSONResponse:
    ok = companion_memory.clear_all()
    return JSONResponse({"success": ok})


@app.post("/api/chat_stream")
async def api_chat_stream(request: Request) -> StreamingResponse:
    body = await request.json()
    user_text = (body.get("message") or "").strip()
    history = body.get("history") or []
    gpu_mode = body.get("gpu_mode") or "concurrent"  # "concurrent" or "timeshare"
    voice_key = body.get("voice_preset") or "female_warm"
    custom_instruction = (body.get("custom_instruction") or "").strip()
    local_speaker = bool(body.get("local_speaker", True))
    expressive_tags = bool(body.get("expressive_tags", True))

    VOICE_TO_PET: dict[str, tuple[str, str]] = {
        "female_warm": ("luma", "棉棉"),
        "female_lively": ("pipi", "桃桃"),
        "male_calm": ("taigo", "芋圆"),
        "english_narrator": ("auri", "芽芽"),
        "female_sunny": ("nuannuan", "暖暖"),
        "female_cloud": ("duoduo", "朵朵"),
    }
    PET_NAMES_MAP = {
        "luma": "棉棉",
        "pipi": "桃桃",
        "taigo": "芋圆",
        "auri": "芽芽",
        "nuannuan": "暖暖",
        "duoduo": "朵朵",
    }
    PET_TRAITS = {
        "luma": "你是一只雪白棉花糖般软绵纯真、乖巧治愈的毛绒小玩偶，说话轻柔甜糯。",
        "pipi": "你是一只粉发飘逸、鬼马古灵精怪又元气满满的毛绒小玩偶，说话俏皮灵动。",
        "taigo": "你是一只香芋紫色、性格憨萌有点小害羞但暖心真诚的小少年毛绒玩偶。",
        "auri": "你是一只苔藓森绿色的森林精灵毛绒玩偶，机灵博学、充满朝气，中英文都很擅长。",
        "nuannuan": "你是一只向日葵暖黄色、像阳光一样热情甜暖、爱撒娇治愈的毛绒小玩偶。",
        "duoduo": "你是一只冰川云朵蓝、像棉花糖一样软糯梦幻又天真好奇的毛绒小玩偶。",
    }
    speaker_pet = (body.get("pet_id") or "").strip()
    if not speaker_pet or speaker_pet not in PET_NAMES_MAP:
        speaker_pet, pet_name = VOICE_TO_PET.get(voice_key, ("luma", "棉棉"))
    else:
        pet_name = PET_NAMES_MAP[speaker_pet]

    preset = VOICE_PRESETS.get(voice_key, VOICE_PRESETS["female_warm"])
    instruction = custom_instruction if custom_instruction else preset["instruction"]
    cfg_scale = float(body.get("cfg_scale") or preset["cfg_scale"])
    seed = int(body.get("seed") or preset["seed"])

    gen_id = speaker_player.stop()

    # 同步极速提炼当前发言中的主人姓名槽位（仅记录主人信息，不存储灵宠名字）
    sync_actions = companion_memory.sync_extract_core_slots(user_text, agent_id=speaker_pet)

    # 检索伴生灵宠长期记忆库（置顶主人姓名槽位 + Mem0 向量语义召回）
    mem_ctx = companion_memory.get_context_for_chat(user_text, agent_id=speaker_pet, limit=5)
    memories: list[str] = mem_ctx["memories"]
    master_name: str | None = mem_ctx["master_name"]
    effective_pet_name = pet_name

    pet_trait = PET_TRAITS.get(speaker_pet, "")
    role_summary = f"你是主人的专属桌面伴生灵宠，你自己的名字永远叫「{effective_pet_name}」（绝对禁止自称“灵宠助手”或“AI助手”）。{pet_trait}"
    if master_name:
        role_summary += f"正在和你说话的用户是你的主人，主人的名字叫「{master_name}」。"

    now_dt = datetime.now()
    weekdays_all = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday_cn = weekdays_all[now_dt.weekday()]
    h_cur = now_dt.hour
    period_cur = "凌晨" if h_cur < 6 else ("早晨" if h_cur < 9 else ("上午" if h_cur < 12 else ("中午" if h_cur < 14 else ("下午" if h_cur < 18 else ("晚上" if h_cur < 23 else "深夜")))))
    now_full_str = f"{now_dt.year}年{now_dt.month}月{now_dt.day}日 {weekday_cn} {period_cur} {now_dt.strftime('%H:%M:%S')}"

    sys_prompt = (
        f"【当前电脑本地硬件权威时间】：{now_full_str}（24小时制本地真实时钟，问时间必须严格依据此回答）\n"
        f"{role_summary}"
        "请用自然口语化的中文回答，每次回复尽量精炼亲切（控制在1到3句话以内），"
        "不要使用 Markdown 符号（如 **、#、列表符号等），以便直接朗读。"
    )
    if expressive_tags:
        sys_prompt += (
            "你可以在极其自然的语境下偶尔在句首或句中加入一个副语言情感标签，"
            "中文仅限使用 [笑] 或 [叹气]，英文仅限使用 (laugh) 或 (sigh)，不要滥用。"
        )

    if memories:
        mem_lines = []
        for m in memories:
            if "主人的名字是" in m and master_name:
                mem_lines.append(f"- 主人（用户）的名字是：{master_name}")
            else:
                mem_lines.append(f"- {m}")
        mem_block = "\n".join(mem_lines)
        sys_prompt += f"\n\n【你的长期记忆库】：\n{mem_block}"

    # 检索全息视觉之眼（若用户要求拍电脑摄像头照片或发当前页面/屏幕截图，自动0显存极速抓拍）
    enable_ai_vision = bool(body.get("enable_ai_vision", True))
    vision_ctx = companion_vision.analyze_and_capture(
        user_text, load_llm_config(), enable_ai_vision=enable_ai_vision
    )
    if vision_ctx and vision_ctx.get("context_text"):
        sys_prompt += f"\n\n{vision_ctx['context_text']}"

    # 检索全息联网感知中枢（若未触发拍照截图且提问涉及实时天气、今日新闻或全网搜索，自动抓取实时数据）
    web_ctx = None if vision_ctx else companion_web.analyze_and_search(user_text)
    if web_ctx and web_ctx.get("context_text"):
        sys_prompt += f"\n\n【全息联网实时感知数据（100%真实最新数据，请严格据此作答）】：\n{web_ctx['context_text']}"

    # 针对 3B 小模型及云端模型的当轮意图精准引导
    asking_pet_name = bool(re.search(r"你(?:现在)?(?:叫什么|叫啥|是谁|的名字)", user_text))
    asking_master_name = bool(re.search(r"我(?:现在)?(?:叫什么|叫啥|是谁|的名字)", user_text))
    asking_time = bool(re.search(r"(?:现在几点|几点了|几点啦|当前时间|报个时|现在是什么时间|今天几号|今天星期几|今天周几|今年是哪年|今年几几年|日期是什么|今天是哪一天|现在时间)", user_text))
    if vision_ctx:
        sys_prompt += (
            "\n【当前回复指引】：主人要求你抓拍电脑摄像头照片或截取当前电脑页面发给他。系统刚刚已经把高清图片实时显示在了主人的聊天窗口里！"
            "请严格结合上方【画面实时感知信息】，用自然亲切活泼的口语（1到3句话）告诉主人图片已经发过去啦，并顺带点评一下画面内容（绝对禁止说“我发不了图片”或“我是AI看不到画面”）！"
        )
    elif asking_time:
        sys_prompt += (
            f"\n【当前回复指引】：主人正在询问当前电脑时间或日期！"
            f"请严格依据上方【当前电脑本地硬件权威时间】（{now_dt.year}年{now_dt.month}月{now_dt.day}日 {weekday_cn} {period_cur} {now_dt.strftime('%H点%M分')}），"
            f"用极其亲切自然的口语准确回答主人（例如：“现在是{period_cur}{now_dt.strftime('%H点%M分')}啦，今天是{weekday_cn}哦！”），严禁编造其它时间！"
        )
    elif web_ctx and web_ctx.get("category") == "weather":
        sys_prompt += (
            "\n【当前回复指引】：主人正在询问天气。请严格依据上方【全息联网实时感知数据】中的真实温度、天气状况、最高最低温和贴心建议，"
            "用自然亲切的口语（2到3句话）准确播报给主人听，绝对禁止自己瞎编其他温度数字！"
        )
    elif web_ctx and web_ctx.get("category") == "news":
        sys_prompt += (
            "\n【当前回复指引】：主人想了解今天的新闻。请严格从上方【全息联网实时感知数据】的今日实时热点新闻列表中挑选 3 到 4 条重点新闻，"
            "用自然亲切的口语帮主人总结播报（控制在 3 到 5 句话以内，严禁使用 ** 或 # 等符号，严禁脱离列表瞎编旧新闻）！"
        )
    elif web_ctx and web_ctx.get("category") in ("search", "time"):
        sys_prompt += (
            "\n【当前回复指引】：请严格依据上方【全息联网实时感知数据】中的真实信息，用自然亲切的口语（2到4句话）准确回答主人的问题。"
        )
    elif sync_actions and master_name:
        sys_prompt += f"\n【当前回复指引】：主人刚刚介绍了他叫「{master_name}」，请亲切地称呼「{master_name}」主人并表示已牢牢记住！"
    elif asking_pet_name and not asking_master_name:
        sys_prompt += f"\n【当前回复指引】：主人正在问你叫什么名字，请直接亲切地回答“我叫{effective_pet_name}呀！”"
    elif asking_master_name and not asking_pet_name:
        if master_name:
            sys_prompt += f"\n【当前回复指引】：主人正在问他自己叫什么名字，请直接回答“你是我的主人{master_name}呀！”，不要答成你自己的名字。"
    else:
        sys_prompt += "\n【当前回复指引】：请顺着主人的话题自然、温暖地聊天，不要在没有问你名字的时候突兀地自报姓名说“我叫某某”。"

    messages = [{"role": "system", "content": sys_prompt}]
    for item in history[-8:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            content = str(item["content"])
            # 修正历史对话中可能残留的旧“灵宠助手”自称，防止 3B 模型模仿旧错误回复
            if item["role"] == "assistant" and "灵宠助手" in content:
                content = content.replace("灵宠助手", effective_pet_name)
            messages.append({"role": item["role"], "content": content})
    messages.append({"role": "user", "content": user_text})

    from_mobile = bool(body.get("from_mobile", False))
    turn_id = f"turn_{int(time.time() * 1000)}"
    companion_mobile.active_pet_id = speaker_pet

    def event_stream() -> Generator[str, None, None]:
        t0 = time.time()
        first_token_ms: float | None = None
        first_audio_ms: float | None = None
        all_pcm = bytearray()

        companion_mobile.broadcast({
            "type": "chat_turn_start",
            "turn_id": turn_id,
            "user_text": user_text,
            "pet_id": speaker_pet,
            "from_mobile": from_mobile,
        })

        if vision_ctx:
            companion_mobile.broadcast({
                "type": "chat_vision_captured",
                "turn_id": turn_id,
                "pet_id": speaker_pet,
                "mode": vision_ctx.get("mode"),
                "chip_title": vision_ctx.get("chip_title"),
                "image_data_uri": vision_ctx.get("image_data_uri", ""),
                "face_analysis": vision_ctx.get("face_analysis"),
            })
            yield f"data: {json.dumps({'type': 'vision_captured', 'mode': vision_ctx.get('mode'), 'chip_title': vision_ctx.get('chip_title'), 'image_data_uri': vision_ctx.get('image_data_uri', ''), 'face_analysis': vision_ctx.get('face_analysis')})}\n\n"

        if web_ctx and web_ctx.get("chip_title"):
            companion_mobile.broadcast({
                "type": "chat_web_searched",
                "turn_id": turn_id,
                "pet_id": speaker_pet,
                "chip_title": web_ctx.get("chip_title"),
            })
            yield f"data: {json.dumps({'type': 'web_searched', 'category': web_ctx.get('category'), 'chip_title': web_ctx.get('chip_title')})}\n\n"

        if memories:
            companion_mobile.broadcast({
                "type": "chat_memory_recalled",
                "turn_id": turn_id,
                "pet_id": speaker_pet,
                "count": len(memories),
            })
            yield f"data: {json.dumps({'type': 'memory_recalled', 'count': len(memories), 'memories': memories, 'pet': speaker_pet, 'pet_name': effective_pet_name})}\n\n"

        if gpu_mode == "timeshare":
            # Mode 2: Time-sharing GPU mode (Qwen runs on GPU -> unloads from VRAM -> Breeze TTS 2 runs on GPU)
            yield f"data: {json.dumps({'type': 'stage', 'stage': 'qwen_gpu', 'msg': '千问 3B 正在使用 GPU 生成回复...'})}\n\n"
            full_text = ""
            for tok in stream_qwen_tokens(messages, keep_alive=0):
                if gen_id != speaker_player.current_gen:
                    return
                if first_token_ms is None:
                    first_token_ms = (time.time() - t0) * 1000
                full_text += tok
                companion_mobile.broadcast({
                    "type": "chat_text_delta",
                    "turn_id": turn_id,
                    "pet_id": speaker_pet,
                    "delta": tok,
                })
                yield f"data: {json.dumps({'type': 'text', 'delta': tok, 'first_token_ms': first_token_ms})}\n\n"

            # 异步后台提炼沉淀记忆，不阻塞后续语音合成
            if full_text.strip():
                companion_asr.record_pet_speech(full_text)
                companion_memory.add_async(user_text, full_text, agent_id=speaker_pet)

            yield f"data: {json.dumps({'type': 'stage', 'stage': 'breeze_gpu', 'msg': '千问已释放 GPU 显存，Breeze TTS 2 正在 GPU 合成语音...'})}\n\n"
            for pcm_chunk in stream_tts_pcm(full_text, instruction, cfg_scale=cfg_scale, seed=seed):
                if gen_id != speaker_player.current_gen:
                    return
                if first_audio_ms is None:
                    first_audio_ms = (time.time() - t0) * 1000
                all_pcm.extend(pcm_chunk)
                if local_speaker:
                    speaker_player.play_pcm(pcm_chunk, gen_id)
                b64_pcm = base64.b64encode(pcm_chunk).decode("ascii")
                yield f"data: {json.dumps({'type': 'audio_pcm', 'pcm_b64': b64_pcm, 'sample_rate': SAMPLE_RATE, 'first_audio_ms': first_audio_ms})}\n\n"
        else:
            # Mode 1: Concurrent Dual-GPU Streaming Mode (Qwen 3B + Breeze TTS 2 both resident in VRAM)
            event_q: queue.Queue[dict[str, Any] | None] = queue.Queue()
            sent_q: queue.Queue[str | None] = queue.Queue()
            full_llm_text = ""

            def llm_producer() -> None:
                nonlocal first_token_ms, full_llm_text
                try:
                    def on_tok(tok: str) -> None:
                        nonlocal first_token_ms, full_llm_text
                        if first_token_ms is None:
                            first_token_ms = (time.time() - t0) * 1000
                        full_llm_text += tok
                        companion_mobile.broadcast({
                            "type": "chat_text_delta",
                            "turn_id": turn_id,
                            "pet_id": speaker_pet,
                            "delta": tok,
                        })
                        event_q.put(
                            {
                                "type": "text",
                                "delta": tok,
                                "first_token_ms": first_token_ms,
                            }
                        )

                    for seg in split_sentences_incremental(
                        stream_qwen_tokens(messages, keep_alive="30m"),
                        on_token=on_tok,
                    ):
                        if gen_id != speaker_player.current_gen:
                            break
                        companion_asr.record_pet_speech(seg)
                        sent_q.put(seg)
                except Exception as exc:
                    event_q.put({"type": "error", "error": f"LLM Error: {exc}"})
                finally:
                    sent_q.put(None)

            def tts_consumer() -> None:
                nonlocal first_audio_ms
                try:
                    while True:
                        seg = sent_q.get()
                        if seg is None or gen_id != speaker_player.current_gen:
                            break
                        for pcm_chunk in stream_tts_pcm(
                            seg, instruction, cfg_scale=cfg_scale, seed=seed
                        ):
                            if gen_id != speaker_player.current_gen:
                                break
                            if first_audio_ms is None:
                                first_audio_ms = (time.time() - t0) * 1000
                            all_pcm.extend(pcm_chunk)
                            if local_speaker:
                                speaker_player.play_pcm(pcm_chunk, gen_id)
                            b64_pcm = base64.b64encode(pcm_chunk).decode("ascii")
                            event_q.put(
                                {
                                    "type": "audio_pcm",
                                    "pcm_b64": b64_pcm,
                                    "sample_rate": SAMPLE_RATE,
                                    "first_audio_ms": first_audio_ms,
                                }
                            )
                except Exception as exc:
                    event_q.put({"type": "error", "error": f"TTS Error: {exc}"})
                finally:
                    event_q.put(None)

            threading.Thread(target=llm_producer, daemon=True).start()
            threading.Thread(target=tts_consumer, daemon=True).start()

            while True:
                item = event_q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

            # 异步后台提炼沉淀记忆
            if full_llm_text.strip():
                companion_asr.record_pet_speech(full_llm_text)
                companion_memory.add_async(user_text, full_llm_text, agent_id=speaker_pet)

        if all_pcm:
            wav_bytes = pcm_to_wav(bytes(all_pcm), SAMPLE_RATE)
            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
            yield f"data: {json.dumps({'type': 'audio_wav', 'wav_b64': wav_b64, 'duration_s': round(len(all_pcm) / 2 / SAMPLE_RATE, 2)})}\n\n"

        rem_s = round(speaker_player.remaining_seconds(), 2)
        companion_mobile.broadcast({
            "type": "chat_turn_done",
            "turn_id": turn_id,
            "pet_id": speaker_pet,
            "speaker_remaining_s": rem_s,
        })
        yield f"data: {json.dumps({'type': 'done', 'first_token_ms': first_token_ms, 'first_audio_ms': first_audio_ms, 'memory_recalled_count': len(memories), 'speaker_remaining_s': rem_s})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
MOBILE_TEMPLATE_PATH = BASE_DIR / "templates" / "mobile.html"


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if TEMPLATE_PATH.is_file():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Template not found</h1>")


@app.api_route("/mobile", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def mobile_index() -> HTMLResponse:
    if MOBILE_TEMPLATE_PATH.is_file():
        return HTMLResponse(MOBILE_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Mobile template not found</h1>")


@app.get("/api/mobile/info")
async def api_mobile_info() -> JSONResponse:
    return JSONResponse(companion_mobile.get_connection_info())


@app.post("/api/mobile/tunnel/restart")
async def api_mobile_tunnel_restart() -> JSONResponse:
    companion_mobile.start_tunnel_async(force_restart=True)
    return JSONResponse({"success": True, "status": "starting"})


@app.post("/api/mobile/action")
async def api_mobile_action(request: Request) -> JSONResponse:
    body = await request.json()
    action = (body.get("action") or "").strip()
    if action == "select_pet":
        pet_id = (body.get("pet_id") or "luma").strip()
        companion_mobile.active_pet_id = pet_id
        companion_mobile.broadcast({"type": "pet_switched", "pet_id": pet_id})
    elif action == "stage_mode":
        mode = (body.get("mode") or "").strip()
        if mode == "stop_speech":
            speaker_player.stop()
        companion_mobile.broadcast({"type": "stage_mode_triggered", "mode": mode})
    return JSONResponse({"success": True})


@app.post("/api/mobile/send")
async def api_mobile_send(request: Request) -> JSONResponse:
    body = await request.json()
    text = (body.get("message") or "").strip()
    pet_id = (body.get("pet_id") or companion_mobile.active_pet_id or "luma").strip()
    if not text:
        return JSONResponse({"success": False, "error": "消息为空"})

    clean_text, called_pet_id, _ = _normalize_and_detect_pet(text)
    if clean_text:
        text = clean_text
    if called_pet_id:
        pet_id = called_pet_id
        companion_mobile.broadcast({"type": "pet_selected", "pet_id": pet_id})

    companion_mobile.active_pet_id = pet_id
    primary_pc = companion_mobile.get_primary_pc_client_id()

    if primary_pc:
        # 电脑端已打开 Web 主舞台：推送指令给电脑主舞台自动弹出气泡并触发语音与灵宠动画
        companion_mobile.broadcast({
            "type": "mobile_trigger_chat",
            "message": text,
            "pet_id": pet_id,
            "target_pc_id": primary_pc,
        })
        return JSONResponse({"success": True, "mode": "pc_synced", "target_pc_id": primary_pc, "pet_id": pet_id})
    else:
        # 若电脑端尚未打开浏览器标签页，后台自动兜底执行对话流并驱动本机音箱播报
        PET_TO_VOICE = {
            "luma": "female_warm",
            "pipi": "female_lively",
            "taigo": "male_calm",
            "auri": "english_narrator",
            "nuannuan": "female_sunny",
            "duoduo": "female_cloud",
        }
        def _bg_fallback() -> None:
            try:
                payload = {
                    "message": text,
                    "history": [],
                    "gpu_mode": "concurrent",
                    "voice_preset": PET_TO_VOICE.get(pet_id, "female_warm"),
                    "pet_id": pet_id,
                    "local_speaker": True,
                    "expressive_tags": True,
                    "from_mobile": True,
                }
                req = urllib.request.Request(
                    f"http://127.0.0.1:{WEB_PORT}/api/chat_stream",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    for _ in resp:
                        pass
            except Exception:
                pass
        threading.Thread(target=_bg_fallback, daemon=True).start()
        return JSONResponse({"success": True, "mode": "headless_fallback", "pet_id": pet_id})


# ============================================================
# 离线语音转文字与声纹门禁中枢 (GTCRN + SileroVAD + CAM++ + SenseVoice CPU)
# ============================================================
async def _extract_audio_bytes(request: Request) -> bytes:
    """从 JSON / multipart-formdata / 原始二进制请求中安全提取音频字节"""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        body = await request.json()
        b64 = (body.get("audio_b64") or "").strip()
        return base64.b64decode(b64) if b64 else b""
    if "multipart/form-data" in content_type:
        raw = await request.body()
        # 提取 multipart 中文件字段二进制荷载
        file_marker = b'name="file"'
        idx = raw.find(file_marker)
        start_from = idx if idx != -1 else 0
        sep = raw.find(b"\r\n\r\n", start_from)
        if sep != -1:
            part_end = raw.find(b"\r\n--", sep + 4)
            if part_end != -1 and part_end > sep + 4:
                return raw[sep + 4 : part_end]
        return raw
    return await request.body()


@app.get("/api/asr/status")
async def api_asr_status() -> JSONResponse:
    status_data = companion_asr.status()
    status_data["speaker_busy"] = speaker_player.is_playing_or_cooldown(1.5)
    status_data["speaker_remaining_s"] = round(speaker_player.remaining_seconds(), 2)
    return JSONResponse({"success": True, **status_data})


@app.post("/api/asr/transcribe")
async def api_asr_transcribe(request: Request) -> JSONResponse:
    source = (request.query_params.get("source") or "manual").strip()
    audio_bytes = await _extract_audio_bytes(request)
    if not audio_bytes:
        return JSONResponse({"success": False, "text": "", "error": "收到空音频数据"})

    res = await asyncio.to_thread(companion_asr.transcribe_bytes, audio_bytes, source)
    return JSONResponse(res)


@app.post("/api/asr/voiceprint/enroll")
async def api_asr_voiceprint_enroll(request: Request) -> JSONResponse:
    append = request.query_params.get("append", "0") in ("1", "true", "yes")
    audio_bytes = await _extract_audio_bytes(request)
    if not audio_bytes:
        return JSONResponse({"success": False, "error": "收到空录音数据"})

    res = await asyncio.to_thread(companion_asr.enroll_voiceprint_bytes, audio_bytes, append)
    return JSONResponse(res)


@app.post("/api/asr/voiceprint/config")
async def api_asr_voiceprint_config(request: Request) -> JSONResponse:
    body = await request.json()
    enabled = body.get("enabled") if "enabled" in body else None
    threshold = body.get("threshold") if "threshold" in body else None
    res = companion_asr.set_voiceprint_config(enabled=enabled, threshold=threshold)
    return JSONResponse(res)


@app.post("/api/asr/voiceprint/clear")
async def api_asr_voiceprint_clear() -> JSONResponse:
    res = companion_asr.clear_voiceprint()
    return JSONResponse(res)


@app.post("/api/asr/pc_mic_start")
async def api_asr_pc_mic_start() -> JSONResponse:
    speaker_player.stop()
    res = await asyncio.to_thread(companion_asr.start_pc_mic)
    return JSONResponse(res)


@app.post("/api/asr/pc_mic_stop")
async def api_asr_pc_mic_stop(request: Request) -> JSONResponse:
    mode = (request.query_params.get("mode") or "transcribe").strip()
    append = request.query_params.get("append", "0") in ("1", "true", "yes")
    res = await asyncio.to_thread(companion_asr.stop_pc_mic_and_transcribe, mode, append)
    return JSONResponse(res)


# ==========================================================
# 人脸面孔中枢 API (Face Hub)
# ==========================================================
@app.get("/api/face/info")
async def api_face_info() -> JSONResponse:
    return JSONResponse(companion_face.get_vault_info())


@app.post("/api/face/enroll_camera")
async def api_face_enroll_camera(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    name = (body.get("name") or "主人").strip()
    is_master = bool(body.get("is_master", name == "主人"))

    cap_res = await asyncio.to_thread(companion_vision.capture_webcam)
    if not cap_res.get("success"):
        return JSONResponse({"success": False, "error": cap_res.get("error", "摄像头抓拍失败")})

    raw_b64 = cap_res.get("raw_b64", "")
    if not raw_b64:
        return JSONResponse({"success": False, "error": "抓拍画面为空"})

    img_bytes = base64.b64decode(raw_b64)
    res = await asyncio.to_thread(companion_face.enroll_from_image_bytes, img_bytes, name, is_master)
    if res.get("success"):
        companion_mobile.broadcast({"type": "face_vault_updated"})
    return JSONResponse(res)


@app.post("/api/face/enroll_upload")
async def api_face_enroll_upload(request: Request) -> JSONResponse:
    content_type = request.headers.get("content-type", "")
    name = "熟悉朋友"
    is_master = False
    img_bytes = b""
    if "multipart/form-data" in content_type:
        form = await request.form()
        file_obj = form.get("file")
        name = (form.get("name") or "熟悉朋友").strip()
        is_master = form.get("is_master") in ("1", "true", "yes")
        if file_obj and hasattr(file_obj, "read"):
            img_bytes = await file_obj.read()
    else:
        body = await request.json()
        name = (body.get("name") or "熟悉朋友").strip()
        is_master = bool(body.get("is_master", False))
        raw_b64 = body.get("image_b64") or body.get("image_data_uri") or ""
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        if raw_b64:
            img_bytes = base64.b64decode(raw_b64)

    if not img_bytes:
        return JSONResponse({"success": False, "error": "请提供有效的人脸照片数据"})

    res = await asyncio.to_thread(companion_face.enroll_from_image_bytes, img_bytes, name, is_master)
    if res.get("success"):
        companion_mobile.broadcast({"type": "face_vault_updated"})
    return JSONResponse(res)


@app.post("/api/face/delete")
async def api_face_delete(request: Request) -> JSONResponse:
    body = await request.json()
    face_id = (body.get("face_id") or "").strip()
    if not face_id:
        return JSONResponse({"success": False, "error": "缺少 face_id 参数"})
    res = companion_face.delete_face(face_id)
    if res.get("success"):
        companion_mobile.broadcast({"type": "face_vault_updated"})
    return JSONResponse(res)


@app.post("/api/face/switch_target")
async def api_face_switch_target(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        target_idx = body.get("target_index")
    except Exception:
        target_idx = None

    if target_idx is not None:
        cap_res = await asyncio.to_thread(companion_vision.capture_webcam, target_index=int(target_idx))
        if cap_res.get("success") and cap_res.get("face_analysis"):
            companion_mobile.broadcast({
                "type": "face_target_switched",
                "face_analysis": cap_res["face_analysis"],
                "image_data_uri": cap_res.get("image_data_uri"),
            })
        return JSONResponse(cap_res)
    else:
        res = companion_face.switch_to_next_target()
        if res.get("success"):
            cap_res = await asyncio.to_thread(companion_vision.capture_webcam, target_index=res.get("locked_index"))
            if cap_res.get("success"):
                res["image_data_uri"] = cap_res.get("image_data_uri")
                res["face_analysis"] = cap_res.get("face_analysis")
                companion_mobile.broadcast({
                    "type": "face_target_switched",
                    "face_analysis": cap_res.get("face_analysis"),
                    "image_data_uri": cap_res.get("image_data_uri"),
                })
        return JSONResponse(res)


@app.get("/api/mobile/events")
async def api_mobile_events(request: Request) -> StreamingResponse:
    role = (request.query_params.get("role") or "mobile").strip()
    client_id = (request.query_params.get("client_id") or f"{role}_{int(time.time()*1000)}").strip()
    q = companion_mobile.register_client(client_id, role)

    async def sse_gen():
        try:
            # 初始握手包
            info = companion_mobile.get_connection_info()
            yield f"data: {json.dumps({'type': 'presence', 'pc_clients': info['pc_clients'], 'mobile_clients': info['mobile_clients'], 'active_pet_id': info['active_pet_id']})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = q.get_nowait()
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    await asyncio.sleep(0.08)
        finally:
            companion_mobile.unregister_client(client_id)

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


@app.get("/api/face/video_feed")
async def api_face_video_feed(request: Request) -> StreamingResponse:
    """实时视线哨兵 MJPEG 视频流输出，带全息青色对焦框与极简状态角标"""
    q: queue.Queue[bytes] = queue.Queue(maxsize=2)
    companion_sentinel.add_frame_subscriber(q)

    async def frame_streamer():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    frame_bytes = q.get_nowait()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                except queue.Empty:
                    await asyncio.sleep(0.04)
        finally:
            companion_sentinel.remove_frame_subscriber(q)

    return StreamingResponse(
        frame_streamer(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/face/sentinel/status")
async def api_face_sentinel_status() -> JSONResponse:
    return JSONResponse(companion_sentinel.get_status())


@app.post("/api/face/sentinel/toggle")
async def api_face_sentinel_toggle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        target_state = body.get("enabled")
    except Exception:
        target_state = None
    new_state = companion_sentinel.toggle(target_state)
    companion_mobile.broadcast({
        "type": "sentinel_state_changed",
        "enabled": new_state,
    })
    return JSONResponse(companion_sentinel.get_status())


@app.post("/api/face/sentinel/cooldown_toggle")
@app.post("/api/face/sentinel/greeting_toggle")
async def api_face_sentinel_greeting_toggle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        target_state = body.get("enabled")
        if target_state is None:
            target_state = body.get("greeting_enabled")
    except Exception:
        target_state = None
    new_state = companion_sentinel.toggle_greeting(target_state)
    companion_mobile.broadcast({
        "type": "sentinel_greeting_changed",
        "greeting_enabled": new_state,
        "cooldown_enabled": new_state,
    })
    return JSONResponse(companion_sentinel.get_status())


@app.get("/api/face/sentinel/greetings")
async def api_face_sentinel_greetings() -> JSONResponse:
    return JSONResponse({"greetings": companion_sentinel.get_recent_greetings()})


@app.get("/api/system/time")
async def api_system_time() -> JSONResponse:
    """获取当前电脑本地系统硬件实时时间"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekdays_short = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    h = now.hour
    period = "凌晨" if h < 6 else ("早晨" if h < 9 else ("上午" if h < 12 else ("中午" if h < 14 else ("下午" if h < 18 else ("晚上" if h < 23 else "深夜")))))
    return JSONResponse({
        "timestamp": time.time(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
        "time_str": now.strftime("%H:%M:%S"),
        "date_str": now.strftime("%Y-%m-%d"),
        "weekday": weekdays[now.weekday()],
        "weekday_short": weekdays_short[now.weekday()],
        "period": period,
        "display_str": f"{now.strftime('%Y-%m-%d')} {weekdays_short[now.weekday()]} {now.strftime('%H:%M:%S')}",
    })


@app.get("/api/face/sentinel/guard_config")
async def api_face_sentinel_guard_config() -> JSONResponse:
    cfg = companion_sentinel.get_guard_config()
    return JSONResponse(cfg)


@app.post("/api/face/sentinel/guard_config")
async def api_face_sentinel_update_guard_config(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_cfg = companion_sentinel.update_guard_config(
        enabled=body.get("enabled"),
        start_time=body.get("start_time"),
        end_time=body.get("end_time"),
        alert_count=body.get("alert_count"),
        alert_duration_s=body.get("alert_duration_s"),
    )
    companion_mobile.broadcast({
        "type": "sentinel_guard_config_changed",
        "guard_config": new_cfg,
    })
    return JSONResponse(new_cfg)


@app.post("/api/face/sentinel/guard_toggle")
async def api_face_sentinel_guard_toggle(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        target_state = body.get("enabled")
    except Exception:
        target_state = None
    new_state = companion_sentinel.toggle_guard(target_state)
    new_cfg = companion_sentinel.get_guard_config()
    companion_mobile.broadcast({
        "type": "sentinel_guard_config_changed",
        "guard_config": new_cfg,
    })
    return JSONResponse(new_cfg)


@app.post("/api/face/sentinel/test_alert")
async def api_face_sentinel_test_alert() -> JSONResponse:
    snap_uri = companion_sentinel.capture_snapshot_data_uri()
    handle_sentinel_guard_alert(score=0.79, face_analysis=None, snapshot_data_uri=snap_uri)
    return JSONResponse({"status": "alert_triggered", "has_snapshot": bool(snap_uri)})


# ==========================================
# 智能早晨唤醒与知识库日程备忘中枢 API
# ==========================================
@app.get("/api/alarms")
async def api_get_alarms() -> JSONResponse:
    """获取所有已配置闹钟与当前响铃运行状态"""
    return JSONResponse({
        "success": True,
        "status": companion_alarm.get_status(),
        "alarms": companion_alarm.get_alarms(),
    })


@app.post("/api/alarms")
async def api_add_or_update_alarm(request: Request) -> JSONResponse:
    """添加或编辑闹钟（支持早晨唤醒晨报与知识库内容备忘）"""
    try:
        body = await request.json()
    except Exception:
        body = {}

    alarm_id = body.get("id")
    time_str = body.get("time", "06:00")
    label = body.get("label", "")
    alarm_type = body.get("type", "morning")
    content = body.get("content", "")
    repeat = body.get("repeat", "daily")
    enabled = body.get("enabled", True)

    try:
        if alarm_id:
            updated = companion_alarm.update_alarm(alarm_id, {
                "time": time_str,
                "label": label,
                "type": alarm_type,
                "content": content,
                "repeat": repeat,
                "enabled": enabled,
            })
            if not updated:
                return JSONResponse({"success": False, "error": "Alarm not found"}, status_code=404)
            res_alarm = updated
        else:
            res_alarm = companion_alarm.add_alarm(
                time_str=time_str,
                label=label,
                alarm_type=alarm_type,
                content=content,
                repeat=repeat,
                enabled=enabled,
            )

        companion_mobile.broadcast({
            "type": "alarms_updated",
            "alarms": companion_alarm.get_alarms(),
        })
        return JSONResponse({"success": True, "alarm": res_alarm})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=400)


@app.post("/api/alarms/toggle")
async def api_toggle_alarm(request: Request) -> JSONResponse:
    """一键开启或关闭指定闹钟"""
    try:
        body = await request.json()
    except Exception:
        body = {}

    alarm_id = body.get("id", "")
    enabled = body.get("enabled")
    res = companion_alarm.toggle_alarm(alarm_id, enabled)
    if not res:
        return JSONResponse({"success": False, "error": "Alarm not found"}, status_code=404)

    companion_mobile.broadcast({
        "type": "alarms_updated",
        "alarms": companion_alarm.get_alarms(),
    })
    return JSONResponse({"success": True, "alarm": res})


@app.delete("/api/alarms/{alarm_id}")
async def api_delete_alarm(alarm_id: str) -> JSONResponse:
    """删除指定闹钟"""
    success = companion_alarm.delete_alarm(alarm_id)
    if success:
        companion_mobile.broadcast({
            "type": "alarms_updated",
            "alarms": companion_alarm.get_alarms(),
        })
        return JSONResponse({"success": True})
    return JSONResponse({"success": False, "error": "Alarm not found"}, status_code=404)


@app.post("/api/alarms/stop")
async def api_stop_active_alarm(request: Request) -> JSONResponse:
    """手动关停正在响铃的闹钟（关停后立即自动触发今日天气/新闻晨报或知识库备忘语音播报）"""
    try:
        body = await request.json()
        reason = body.get("reason", "manual")
    except Exception:
        reason = "manual"

    stopped = companion_alarm.stop_ringing(reason=reason)
    return JSONResponse({
        "success": True,
        "was_ringing": stopped,
        "status": companion_alarm.get_status(),
    })


@app.post("/api/alarms/test/{alarm_id}")
async def api_test_alarm_trigger(alarm_id: str) -> JSONResponse:
    """即刻测试触发指定闹钟（清脆叮叮叮敲钟铃声 + 5分钟倒计时 + 关停后全息播报）"""
    success = companion_alarm.trigger_alarm_instant(alarm_id)
    if success:
        return JSONResponse({"success": True, "status": "alarm_triggered"})
    return JSONResponse({"success": False, "error": "Alarm not found"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, log_level="info")
