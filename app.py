from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
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
from pathlib import Path
from typing import Any, Generator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from companion_asr import _normalize_and_detect_pet, companion_asr
from companion_memory import companion_memory
from companion_mobile import companion_mobile
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

    sys_prompt = (
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
    if vision_ctx:
        sys_prompt += (
            "\n【当前回复指引】：主人要求你抓拍电脑摄像头照片或截取当前电脑页面发给他。系统刚刚已经把高清图片实时显示在了主人的聊天窗口里！"
            "请严格结合上方【画面实时感知信息】，用自然亲切活泼的口语（1到3句话）告诉主人图片已经发过去啦，并顺带点评一下画面内容（绝对禁止说“我发不了图片”或“我是AI看不到画面”）！"
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
            })
            yield f"data: {json.dumps({'type': 'vision_captured', 'mode': vision_ctx.get('mode'), 'chip_title': vision_ctx.get('chip_title'), 'image_data_uri': vision_ctx.get('image_data_uri', '')})}\n\n"

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=WEB_PORT, log_level="info")
