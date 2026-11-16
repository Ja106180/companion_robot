"""
全息伴生舱 · 离线语音识别与四层抗噪防回声中枢 (Companion ASR Engine)
====================================================================
纯 CPU 离线流水线（0 MB 显卡显存占用，不与 Qwen 3B / Breeze TTS 2 争抢 VRAM）：
1. GTCRN 神经网络轻量降噪 (data/asr_guards/gtcrn_simple.onnx)
2. Silero VAD 神经网络真·人声检测 (data/asr_guards/silero_vad.onnx) —— 剔除杂音/键盘/短噪
3. 阿里 3D-Speaker CAM++ 主人声纹门禁 (data/asr_guards/campplus_sv.onnx) —— 只听主人一人说话
4. SenseVoice-Small (INT8) 离线转写 + 伙伴同音字点名纠偏 + 双重防回声文本比对校验
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("CompanionASR")

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "sensevoice_int8"
MODEL_PATH = MODEL_DIR / "model.int8.onnx"
TOKENS_PATH = MODEL_DIR / "tokens.txt"

GUARD_DIR = BASE_DIR / "data" / "asr_guards"
SILERO_VAD_PATH = GUARD_DIR / "silero_vad.onnx"
GTCRN_PATH = GUARD_DIR / "gtcrn_simple.onnx"
CAMPPLUS_PATH = GUARD_DIR / "campplus_sv.onnx"
VOICEPRINT_PATH = GUARD_DIR / "master_voiceprint.json"

SAMPLE_RATE = 16000
DEFAULT_SV_THRESHOLD = 0.42

# 6 位毛绒伙伴标准名与常见语音同音词自动纠偏表
PET_HOMOPHONE_FIXES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^(?:呼叫|喂|嗨|你好呀|你好|请问|那个)?\s*(?:棉棉|绵绵|眠眠|明明)"), "棉棉", "luma"),
    (re.compile(r"^(?:呼叫|喂|嗨|你好呀|你好|请问|那个)?\s*(?:桃桃|逃逃|淘淘|陶陶)"), "桃桃", "pipi"),
    (re.compile(r"^(?:呼叫|喂|嗨|你好呀|你好|请问|那个)?\s*(?:芋圆|玉圆|裕元|芋园|预言)"), "芋圆", "taigo"),
    (re.compile(r"^(?:呼叫|喂|嗨|你好呀|你好|请问|那个)?\s*(?:芽芽|丫丫|雅雅|牙牙)"), "芽芽", "auri"),
    (re.compile(r"^(?:呼叫|喂|嗨|你好呀|你好|请问|那个)?\s*(?:暖暖|囡囡)"), "暖暖", "nuannuan"),
    (re.compile(r"^(?:呼叫|喂|嗨|你好呀|你好|请问|那个)?\s*(?:朵朵|多多|躲躲)"), "朵朵", "duoduo"),
]

PET_NAME_TO_ID: dict[str, str] = {
    "棉棉": "luma",
    "桃桃": "pipi",
    "芋圆": "taigo",
    "芽芽": "auri",
    "暖暖": "nuannuan",
    "朵朵": "duoduo",
}

# 常见单音节噪音幻觉词（< 0.9s 音频中出现大概率为呼吸/桌面震动误识）
NOISE_HALLUCINATION_RE = re.compile(
    r"^(?:yeah|yes|no|oh|ah|uh|um|hm+|h|m|a|i|o|ok|hi|hey|bye|the|and|嗯|啊|哦|呃|哎|哼|呀|呢|吧|哈)+[.。!！?？,，]*$",
    re.IGNORECASE,
)


def _strip_for_echo_compare(text: str) -> str:
    """移除标签与标点空白，保留核心汉字/字母用于文本回声比对"""
    cleaned = re.sub(r"\[[^\]]+\]|\([^)]+\)|（[^）]+）", "", text or "")
    cleaned = re.sub(r"[\s，。！？、；：“”‘’\"'.,!?;:\-…~～·]+", "", cleaned)
    return cleaned.lower()


def _normalize_and_detect_pet(raw_text: str) -> tuple[str, str | None, str | None]:
    """清理 SenseVoice 输出标签，纠偏伙伴名字同音字，并检测是否点名呼唤特定伙伴"""
    if not raw_text:
        return "", None, None

    text = re.sub(r"<\|[^|>]+\|>", "", raw_text).strip()
    if not text:
        return "", None, None

    called_pet_id: str | None = None
    called_pet_name: str | None = None

    for pattern, canonical_name, pet_id in PET_HOMOPHONE_FIXES:
        m = pattern.search(text)
        if m:
            prefix_seg = m.group(0)
            for wrong in (
                "绵绵", "眠眠", "明明", "逃逃", "淘淘", "陶陶",
                "玉圆", "裕元", "芋园", "预言",
                "丫丫", "雅雅", "牙牙", "囡囡", "多多", "躲躲",
            ):
                if wrong in prefix_seg:
                    text = text[: m.start()] + prefix_seg.replace(wrong, canonical_name) + text[m.end() :]
                    break
            called_pet_id = pet_id
            called_pet_name = canonical_name
            break

    if not called_pet_id:
        head = text[:10]
        for p_name, p_id in PET_NAME_TO_ID.items():
            if p_name in head:
                called_pet_id = p_id
                called_pet_name = p_name
                break

    return text, called_pet_id, called_pet_name


class CompanionASR:
    """线程安全的离线语音中枢：GTCRN降噪 + SileroVAD + CAM++声纹门禁 + SenseVoice"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._guard_lock = threading.Lock()
        self._recognizer: Any = None
        self._denoiser: Any = None
        self._vad_cfg: Any = None
        self._sv_extractor: Any = None

        self._ready = False
        self._error: str = ""

        # 主人声纹持久化档案
        self._sv_enabled: bool = False
        self._sv_threshold: float = DEFAULT_SV_THRESHOLD
        self._sv_embeddings: list[list[float]] = []
        self._sv_enrolled_at: str = ""
        self._load_voiceprint_file()

        # 防回声机制：近期宠物说过的文本队列 + 后端放音 busy 回调
        self._recent_pet_texts: deque[tuple[float, str]] = deque(maxlen=16)
        self._speaker_busy_cb: Callable[[], bool] | None = None

        # 电脑本地硬件麦克风录音子进程状态
        self._mic_lock = threading.Lock()
        self._mic_proc: subprocess.Popen[bytes] | None = None
        self._mic_chunks: list[bytes] = []
        self._mic_reader_thread: threading.Thread | None = None
        self._mic_start_time: float = 0.0

    def set_speaker_busy_checker(self, cb: Callable[[], bool]) -> None:
        self._speaker_busy_cb = cb

    def record_pet_speech(self, text: str) -> None:
        """记录近期灵宠播报内容，供后续防自身回声比对拦截"""
        norm = _strip_for_echo_compare(text)
        if len(norm) >= 4:
            self._recent_pet_texts.append((time.time(), norm))

    def _is_pet_echo_text(self, text: str) -> bool:
        """检测转写文本是否与近 45 秒内灵宠刚说过的句子高度重合（扬声器尾音回声）"""
        cand = _strip_for_echo_compare(text)
        if len(cand) < 4:
            return False
        now = time.time()
        # 保留 45 秒内句子进行子串和双字集合重合度对比
        cand_bigrams = {cand[i : i + 2] for i in range(len(cand) - 1)} if len(cand) >= 2 else {cand}
        for ts, pet_norm in list(self._recent_pet_texts):
            if now - ts > 45.0:
                continue
            if cand in pet_norm:
                return True
            if len(cand) >= 5 and cand_bigrams:
                pet_bigrams = {pet_norm[i : i + 2] for i in range(len(pet_norm) - 1)}
                overlap = len(cand_bigrams & pet_bigrams) / max(1, len(cand_bigrams))
                if overlap >= 0.55:
                    return True
        return False

    def _load_voiceprint_file(self) -> None:
        if not VOICEPRINT_PATH.is_file():
            return
        try:
            data = json.loads(VOICEPRINT_PATH.read_text(encoding="utf-8"))
            embs = data.get("embeddings") or []
            if isinstance(embs, list) and embs:
                self._sv_embeddings = [list(map(float, e)) for e in embs if isinstance(e, list) and len(e) >= 64]
                self._sv_enabled = bool(data.get("enabled", True)) and bool(self._sv_embeddings)
                self._sv_threshold = float(data.get("threshold", DEFAULT_SV_THRESHOLD))
                self._sv_enrolled_at = str(data.get("enrolled_at", ""))
                logger.info(
                    f"[ASR] Loaded master voiceprint ({len(self._sv_embeddings)} samples, "
                    f"enabled={self._sv_enabled}, threshold={self._sv_threshold})"
                )
        except Exception as exc:
            logger.warning(f"[ASR] Failed loading voiceprint file: {exc}")

    def _save_voiceprint_file(self) -> None:
        GUARD_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "enabled": self._sv_enabled,
            "threshold": round(self._sv_threshold, 3),
            "enrolled_at": self._sv_enrolled_at,
            "embeddings": self._sv_embeddings,
        }
        VOICEPRINT_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def ensure_loaded(self) -> bool:
        if self._ready and self._recognizer is not None:
            return True
        with self._lock:
            if self._ready and self._recognizer is not None:
                return True
            if not MODEL_PATH.exists() or not TOKENS_PATH.exists():
                self._error = f"SenseVoice 模型文件未就绪: {MODEL_PATH}"
                logger.warning(self._error)
                return False
            try:
                import sherpa_onnx

                t0 = time.time()
                logger.info("[ASR] Loading SenseVoice-Small + GTCRN + SileroVAD + CAM++ on CPU (0 MB VRAM)...")
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(MODEL_PATH),
                    tokens=str(TOKENS_PATH),
                    num_threads=4,
                    sample_rate=SAMPLE_RATE,
                    use_itn=True,
                    language="auto",
                    provider="cpu",
                )

                if GTCRN_PATH.is_file():
                    den_cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(
                        model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                            gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(GTCRN_PATH)),
                            num_threads=2,
                            provider="cpu",
                        )
                    )
                    self._denoiser = sherpa_onnx.OfflineSpeechDenoiser(den_cfg)

                if SILERO_VAD_PATH.is_file():
                    self._vad_cfg = sherpa_onnx.VadModelConfig(
                        silero_vad=sherpa_onnx.SileroVadModelConfig(
                            model=str(SILERO_VAD_PATH),
                            threshold=0.45,
                            min_silence_duration=0.25,
                            min_speech_duration=0.20,
                            window_size=512,
                        ),
                        sample_rate=SAMPLE_RATE,
                        num_threads=1,
                        provider="cpu",
                    )

                if CAMPPLUS_PATH.is_file():
                    sv_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                        model=str(CAMPPLUS_PATH),
                        num_threads=2,
                        provider="cpu",
                    )
                    self._sv_extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sv_cfg)

                self._ready = True
                self._error = ""
                logger.info(
                    f"[ASR] 4-Layer CPU Speech Pipeline ready in {(time.time() - t0)*1000:.0f} ms "
                    f"(GTCRN={self._denoiser is not None}, SileroVAD={self._vad_cfg is not None}, CAM++={self._sv_extractor is not None})."
                )
                return True
            except Exception as exc:
                self._error = f"加载语音流水线失败: {exc}"
                logger.exception(self._error)
                return False

    def status(self) -> dict[str, Any]:
        return {
            "ready": self._ready or (MODEL_PATH.exists() and TOKENS_PATH.exists()),
            "loaded": self._ready,
            "engine": "SenseVoice-Small + GTCRN + SileroVAD + CAM++ (CPU)",
            "vram_mb": 0,
            "guards": {
                "gtcrn": self._denoiser is not None or GTCRN_PATH.is_file(),
                "silero_vad": self._vad_cfg is not None or SILERO_VAD_PATH.is_file(),
                "voiceprint_model": self._sv_extractor is not None or CAMPPLUS_PATH.is_file(),
            },
            "voiceprint": {
                "enrolled": len(self._sv_embeddings) > 0,
                "sample_count": len(self._sv_embeddings),
                "enabled": self._sv_enabled and len(self._sv_embeddings) > 0,
                "threshold": round(self._sv_threshold, 2),
                "enrolled_at": self._sv_enrolled_at,
            },
            "error": self._error,
        }

    @staticmethod
    def decode_audio_to_float32(audio_bytes: bytes) -> np.ndarray:
        """使用 ffmpeg 管道将任意浏览器音频封装转为 16kHz 单声道 float32"""
        if not audio_bytes:
            return np.zeros(0, dtype=np.float32)

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, input=audio_bytes, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            err_msg = proc.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(f"音频解码失败: {err_msg or '空音频流'}")

        return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0

    def _run_denoise_and_vad(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        """运行 GTCRN 降噪 + Silero VAD 提取纯人声片段"""
        work = samples
        with self._guard_lock:
            if self._denoiser is not None:
                den_out = self._denoiser(work.tolist(), SAMPLE_RATE)
                work = np.array(den_out.samples, dtype=np.float32)

            if self._vad_cfg is not None:
                import sherpa_onnx

                vad = sherpa_onnx.VoiceActivityDetector(self._vad_cfg, buffer_size_in_seconds=35)
                win = 512
                segs: list[np.ndarray] = []
                for i in range(0, len(work), win):
                    chunk = work[i : i + win]
                    if len(chunk) < win:
                        chunk = np.pad(chunk, (0, win - len(chunk)))
                    vad.accept_waveform(chunk)
                    while not vad.empty():
                        segs.append(np.array(vad.front.samples, dtype=np.float32))
                        vad.pop()
                vad.flush()
                while not vad.empty():
                    segs.append(np.array(vad.front.samples, dtype=np.float32))
                    vad.pop()

                if not segs:
                    return np.zeros(0, dtype=np.float32), 0.0
                clean = np.concatenate(segs)
                return clean, float(len(clean)) / SAMPLE_RATE

        return work, float(len(work)) / SAMPLE_RATE

    def _compute_sv_embedding(self, samples: np.ndarray) -> list[float] | None:
        if self._sv_extractor is None or len(samples) < int(0.4 * SAMPLE_RATE):
            return None
        with self._guard_lock:
            stream = self._sv_extractor.create_stream()
            stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples.tolist())
            stream.input_finished()
            vec = np.array(self._sv_extractor.compute(stream), dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            return vec.tolist()

    def _verify_owner_voice(self, samples: np.ndarray) -> tuple[bool, float]:
        """校验音频声纹是否属于已录入的主人"""
        if not self._sv_enabled or not self._sv_embeddings or self._sv_extractor is None:
            return True, 1.0
        emb = self._compute_sv_embedding(samples)
        if emb is None:
            # 极短口令(<0.4s)不做严苛声纹拒绝，防止误伤“好/嗯”等单字指令
            return True, 1.0
        emb_np = np.array(emb, dtype=np.float32)
        scores = [float(np.dot(emb_np, np.array(ref, dtype=np.float32))) for ref in self._sv_embeddings]
        best_score = max(scores) if scores else 0.0
        return best_score >= self._sv_threshold, best_score

    def enroll_voiceprint_samples(self, samples: np.ndarray, append: bool = False) -> dict[str, Any]:
        """从音频数组中提取干净人声并注册/追加为主人声纹"""
        if not self.ensure_loaded():
            return {"success": False, "error": self._error or "语音识别组件尚未就绪"}
        if self._sv_extractor is None:
            return {"success": False, "error": "声纹模型 campplus_sv.onnx 未加载"}

        clean_speech, speech_dur = self._run_denoise_and_vad(samples)
        if speech_dur < 1.0:
            return {
                "success": False,
                "error": f"有效人声时长仅 {speech_dur:.1f} 秒太短啦，请清晰连续朗读 2~5 秒再提交",
                "speech_duration_s": round(speech_dur, 2),
            }

        # 适度音量归一化后再提声纹，提升不同距离拾音的稳定性
        peak = float(np.max(np.abs(clean_speech))) if len(clean_speech) else 0.0
        if 0.01 <= peak < 0.3:
            clean_speech = np.clip(clean_speech * min(4.0, 0.45 / peak), -1.0, 1.0)

        emb = self._compute_sv_embedding(clean_speech)
        if emb is None:
            return {"success": False, "error": "声纹特征提取失败，请重新录制"}

        if append and self._sv_embeddings:
            self._sv_embeddings = (self._sv_embeddings + [emb])[-4:]
        else:
            self._sv_embeddings = [emb]
        self._sv_enabled = True
        self._sv_enrolled_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_voiceprint_file()

        logger.info(f"[ASR] Owner voiceprint enrolled: samples={len(self._sv_embeddings)}, dur={speech_dur:.2f}s")
        return {
            "success": True,
            "sample_count": len(self._sv_embeddings),
            "speech_duration_s": round(speech_dur, 2),
            "enabled": self._sv_enabled,
            "threshold": round(self._sv_threshold, 2),
            "enrolled_at": self._sv_enrolled_at,
            "message": f"主人声纹录入成功（已保存 {len(self._sv_embeddings)} 条声纹样本，门禁已自动开启）",
        }

    def enroll_voiceprint_bytes(self, audio_bytes: bytes, append: bool = False) -> dict[str, Any]:
        try:
            samples = self.decode_audio_to_float32(audio_bytes)
        except Exception as exc:
            return {"success": False, "error": f"录音解码失败: {exc}"}
        return self.enroll_voiceprint_samples(samples, append=append)

    def set_voiceprint_config(self, enabled: bool | None = None, threshold: float | None = None) -> dict[str, Any]:
        if enabled is not None:
            if enabled and not self._sv_embeddings:
                return {"success": False, "error": "尚未录入主人声纹，请先录制 3 秒声音"}
            self._sv_enabled = bool(enabled)
        if threshold is not None:
            self._sv_threshold = max(0.20, min(0.75, float(threshold)))
        if self._sv_embeddings or VOICEPRINT_PATH.is_file():
            self._save_voiceprint_file()
        return {"success": True, "voiceprint": self.status()["voiceprint"]}

    def clear_voiceprint(self) -> dict[str, Any]:
        self._sv_embeddings = []
        self._sv_enabled = False
        self._sv_enrolled_at = ""
        if VOICEPRINT_PATH.is_file():
            VOICEPRINT_PATH.unlink()
        return {"success": True, "voiceprint": self.status()["voiceprint"]}

    def transcribe_samples(self, samples: np.ndarray, source: str = "manual") -> dict[str, Any]:
        """四层离线防噪/声纹/防回声转写流水线"""
        t0 = time.time()
        duration_s = float(len(samples)) / SAMPLE_RATE if len(samples) else 0.0

        # 第 0 层：如果是免按自动对讲（vad）触发，而电脑后端扬声器正在排队出声，直接防回声拦截
        if source == "vad" and self._speaker_busy_cb is not None and self._speaker_busy_cb():
            logger.info("[ASR] Dropped VAD trigger because backend LocalSpeakerPlayer is busy.")
            return {
                "success": False,
                "text": "",
                "rejected_reason": "speaker_echo",
                "error": "伙伴正在语音播报中，已自动防回声静默",
                "duration_s": round(duration_s, 2),
                "latency_ms": 0,
            }

        if duration_s < 0.22:
            return {
                "success": False,
                "text": "",
                "rejected_reason": "too_short",
                "error": "说话时间太短啦，请至少说半秒钟以上哦",
                "duration_s": round(duration_s, 2),
                "latency_ms": 0,
            }

        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        if peak < 0.008 or rms < 0.0012:
            return {
                "success": False,
                "text": "",
                "rejected_reason": "silence",
                "error": "没有听到清晰的声音，请靠近麦克风再试一次",
                "duration_s": round(duration_s, 2),
                "latency_ms": 0,
            }

        if not self.ensure_loaded():
            return {
                "success": False,
                "text": "",
                "error": self._error or "语音识别引擎未就绪",
                "duration_s": round(duration_s, 2),
                "latency_ms": 0,
            }

        # 第 1 + 2 层：GTCRN 神经降噪 + Silero VAD 提取真正人类说话声（过滤杂音/敲击/风扇）
        clean_speech, speech_dur = self._run_denoise_and_vad(samples)
        if speech_dur < 0.22 or len(clean_speech) < int(0.20 * SAMPLE_RATE):
            latency_ms = round((time.time() - t0) * 1000, 1)
            logger.info(f"[ASR] Filtered non-speech noise clip ({duration_s:.2f}s -> {speech_dur:.2f}s speech)")
            return {
                "success": False,
                "text": "",
                "rejected_reason": "non_speech",
                "error": "仅检测到环境杂音或短促噪音，已自动过滤",
                "duration_s": round(duration_s, 2),
                "latency_ms": latency_ms,
            }

        clean_peak = float(np.max(np.abs(clean_speech))) if len(clean_speech) else 0.0
        if 0.01 <= clean_peak < 0.25:
            clean_speech = np.clip(clean_speech * min(4.5, 0.45 / clean_peak), -1.0, 1.0)

        # 第 3 层：3D-Speaker CAM++ 主人声纹校验（旁人说话 & 宠物声线直接拦截）
        sv_ok, sv_score = self._verify_owner_voice(clean_speech)
        if not sv_ok:
            latency_ms = round((time.time() - t0) * 1000, 1)
            logger.info(
                f"[ASR] Rejected non-owner voice (sv_score={sv_score:.3f} < threshold={self._sv_threshold:.2f}, dur={speech_dur:.2f}s)"
            )
            return {
                "success": False,
                "text": "",
                "rejected_reason": "non_owner_voice",
                "sv_score": round(sv_score, 3),
                "error": f"非主人声纹（匹配度 {sv_score:.2f} < {self._sv_threshold:.2f}），已自动忽略",
                "duration_s": round(duration_s, 2),
                "latency_ms": latency_ms,
            }

        # 第 4 层：SenseVoice-Small 转写
        with self._lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, clean_speech.tolist())
            self._recognizer.decode_stream(stream)
            raw_text = (stream.result.text or "").strip()

        latency_ms = round((time.time() - t0) * 1000, 1)
        clean_text, called_pet_id, called_pet_name = _normalize_and_detect_pet(raw_text)

        if not clean_text:
            return {
                "success": False,
                "text": "",
                "rejected_reason": "empty",
                "error": "未识别到有效语音内容",
                "duration_s": round(duration_s, 2),
                "latency_ms": latency_ms,
            }

        # 过滤短噪音常幻觉出的无意义单字/英文感叹词（如 "Yeah." / "H."）
        if speech_dur < 1.0 and NOISE_HALLUCINATION_RE.match(clean_text):
            logger.info(f"[ASR] Filtered short hallucination word: {clean_text!r} ({speech_dur:.2f}s)")
            return {
                "success": False,
                "text": "",
                "rejected_reason": "noise_word",
                "error": "过滤短促误识别杂音",
                "duration_s": round(duration_s, 2),
                "latency_ms": latency_ms,
            }

        # 文本双重防回声校验（拦截音箱尾巴录回自己刚说的半句话）
        if self._is_pet_echo_text(clean_text):
            logger.info(f"[ASR] Dropped pet self-echo utterance: {clean_text!r}")
            return {
                "success": False,
                "text": "",
                "rejected_reason": "text_echo",
                "error": "检测到伙伴自身回声，已自动过滤",
                "duration_s": round(duration_s, 2),
                "latency_ms": latency_ms,
            }

        logger.info(
            f"[ASR] Accepted {speech_dur:.2f}s speech in {latency_ms}ms (sv={sv_score:.2f}): {clean_text!r} "
            f"(called_pet={called_pet_id})"
        )
        return {
            "success": True,
            "text": clean_text,
            "called_pet_id": called_pet_id,
            "called_pet_name": called_pet_name,
            "sv_score": round(sv_score, 3),
            "duration_s": round(duration_s, 2),
            "latency_ms": latency_ms,
        }

    def transcribe_bytes(self, audio_bytes: bytes, source: str = "manual") -> dict[str, Any]:
        """解码任意音频字节并完成四层离线识别"""
        try:
            samples = self.decode_audio_to_float32(audio_bytes)
        except Exception as exc:
            logger.warning(f"[ASR] Audio decode error: {exc}")
            return {"success": False, "text": "", "error": str(exc), "duration_s": 0, "latency_ms": 0}
        return self.transcribe_samples(samples, source=source)

    # ==========================================================
    # 电脑本机声卡直录兜底接口 (parecord / arecord)
    # ==========================================================
    def start_pc_mic(self) -> dict[str, Any]:
        with self._mic_lock:
            self._stop_pc_mic_unlocked()
            cmd = [
                "parecord",
                "--latency-msec=20",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--format=s16le",
                "--raw",
            ]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except FileNotFoundError:
                cmd = [
                    "arecord",
                    "-D",
                    "pulse",
                    "-f",
                    "S16_LE",
                    "-r",
                    str(SAMPLE_RATE),
                    "-c",
                    "1",
                    "-t",
                    "raw",
                    "-q",
                ]
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception as exc:
                    return {"success": False, "error": f"无法打开电脑麦克风: {exc}"}
            except Exception as exc:
                return {"success": False, "error": f"无法打开电脑麦克风: {exc}"}

            self._mic_proc = proc
            self._mic_chunks = []
            self._mic_start_time = time.time()

            def _reader() -> None:
                assert proc.stdout is not None
                while True:
                    buf = proc.stdout.read(1024)
                    if not buf:
                        break
                    self._mic_chunks.append(buf)

            t = threading.Thread(target=_reader, daemon=True, name="PC-Mic-Reader")
            t.start()
            self._mic_reader_thread = t
            return {"success": True}

    def _stop_pc_mic_unlocked(self) -> bytes:
        proc = self._mic_proc
        self._mic_proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._mic_reader_thread is not None:
            self._mic_reader_thread.join(timeout=1.0)
            self._mic_reader_thread = None
        raw_pcm = b"".join(self._mic_chunks)
        self._mic_chunks = []
        return raw_pcm

    def stop_pc_mic_and_transcribe(self, mode: str = "transcribe", append: bool = False) -> dict[str, Any]:
        with self._mic_lock:
            raw_pcm = self._stop_pc_mic_unlocked()
        if not raw_pcm:
            return {"success": False, "text": "", "error": "未录制到麦克风音频数据"}
        samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if mode == "enroll":
            return self.enroll_voiceprint_samples(samples, append=append)
        return self.transcribe_samples(samples, source="manual")


companion_asr = CompanionASR()
