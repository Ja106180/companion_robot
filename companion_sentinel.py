"""
companion_sentinel.py - 全息伴生舱 · 实时视线哨兵与主动感知中枢 (Companion Sentinel Engine)
========================================================================================
核心特性（0 本地显卡显存占用 · 毫秒级锁定 · 智能冷却防刷）：
1. 电脑摄像头常驻低功耗守护：
   - 自动管理 /dev/video0（或 /dev/video1），单路复用，避免多进程冲突。
   - 提供 15~20fps 轻量 MJPEG 实时视频流 (/api/face/video_feed)。
   - 为 companion_vision.capture_webcam 提供 0ms 内存瞬时抓拍（无须反复开关摄像头）。
2. InsightFace 离线周期抽帧感知（约 1.2s 一次，CPU 仅占 ~1-2%）：
   - 实时识别人脸，支持主人识别、熟悉朋友识别与陌生人感知。
   - 动态在实时帧上绘制全息青色对焦框与极简置信度角标。
3. 严格的防骚扰/智能冷却与离场重归防刷机制 (Anti-loop / Cooldown)：
   - 主人一直坐在电脑前：打招呼一次后保持静默守望，绝对不重复打扰。
   - 主人离开电脑前（离开视野 > 45 秒）：标记为已离场。
   - 主人重新归来/路过：冷却期满（如 180s）后再次温暖主动问候。
   - 陌生人友好问候并进入 300s 独立冷却，防止路人经过重复吵闹。
   - 麦克风说话、免提通话或伴侣说话中严格避让，绝不打断。
4. 全自动主动问候联动：
   - 主动问候语根据主人/时段温暖定制，Breeze TTS 2 驱动发声 + 电脑扬声器播放。
   - 实时向 Web 全息主屏和手机端推送聊天问候卡片与视线锁定事件。
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from companion_face import companion_face

CONFIG_PATH = Path(__file__).resolve().parent / "data" / "sentinel_config.json"

logger = logging.getLogger("CompanionSentinel")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Sentinel] %(message)s"))
    logger.addHandler(_h)


class CompanionSentinelEngine:
    """全息伴生舱 · 实时视线哨兵引擎"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._running: bool = True
        self._enabled: bool = True  # 是否启用实时监控

        # 摄像头参数
        self._camera_index: int = 0
        self._cap: cv2.VideoCapture | None = None
        self._camera_ok: bool = False

        # 图像与帧缓存
        self._latest_clean_frame: np.ndarray | None = None
        self._latest_annotated_jpeg: bytes = b""
        self._subscribers: set[queue.Queue[bytes]] = set()

        # 人脸检测与锁定状态
        self._last_face_analysis: dict[str, Any] | None = None
        self._last_detect_time: float = 0.0
        self._detect_interval_s: float = 1.2  # 约1.2s一次InsightFace检测，极低CPU占用

        # 冷却与状态机（彻底解决反复打招呼与倒计时到期打招呼逻辑）
        self._master_cooldown_s: float = 60.0  # 主人打招呼后 60 秒冷却
        self._master_absent_trigger_s: float = 15.0  # 离场 15 秒算离开，再回可立即迎候
        self._master_present: bool = False
        self._master_last_seen: float | None = None
        self._master_absent_since: float | None = None
        self._master_last_greeted: float | None = None
        self._master_no_face_count: int = 0

        self._stranger_cooldown_s: float = 90.0  # 陌生人 90 秒冷却
        self._stranger_present: bool = False
        self._stranger_last_seen: float | None = None
        self._stranger_absent_since: float | None = None
        self._stranger_last_greeted: float | None = None
        self._stranger_no_face_count: int = 0

        self._friend_cooldown_s: float = 60.0  # 好友 60 秒冷却
        self._friend_states: dict[str, dict[str, Any]] = {}
        self._recent_greetings: list[dict[str, Any]] = []
        self._greeting_enabled: bool = True  # 主动迎候开关（开启时主动打招呼并执行冷却，关闭时彻底静音不主动打扰）

        # 警惕模式配置（时间段设定、陌生人闯入警报、手机照片联动）
        self._guard_enabled: bool = False
        self._guard_start_time: str = "10:00"
        self._guard_end_time: str = "20:00"
        self._guard_alert_count: int = 3
        self._guard_alert_duration_s: int = 5
        self._guard_alerting: bool = False
        self._guard_last_alert_time: float = 0.0
        self._guard_alert_cooldown_s: float = 60.0
        self._guard_alert_callback: Callable[[float, dict[str, Any] | None, str], None] | None = None
        self._cancel_alert_event = threading.Event()
        self._load_guard_config()

        # 外部业务回调
        self._greeting_callback: Callable[[str, str, float, dict[str, Any] | None], None] | None = None
        self._speaker_busy_checker: Callable[[], bool] | None = None
        self._mic_busy_checker: Callable[[], bool] | None = None

        # 主动问候并发互斥与接续状态（严防语音未读完被掐断）
        self._greeting_in_progress: bool = False
        self._greeting_start_time: float = 0.0
        self._pending_master_followup: bool = False
        self._pending_stranger_followup: bool = False

        # 启动后台守护线程
        self._worker_thread = threading.Thread(
            target=self._capture_and_detect_loop,
            daemon=True,
            name="SentinelWorker",
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # 外部配置与回调注入
    # ------------------------------------------------------------------
    def set_greeting_in_progress(self, val: bool) -> None:
        """标记当前是否正在执行迎候语音播报（包含思考合成与硬件播放全过程）"""
        with self._lock:
            self._greeting_in_progress = val
            if val:
                self._greeting_start_time = time.time()
            else:
                self._greeting_start_time = 0.0

    def is_greeting_in_progress(self) -> bool:
        """检查当前是否有迎候正在播报（带 35 秒异常超时自动解开保护）"""
        with self._lock:
            if not self._greeting_in_progress:
                return False
            if time.time() - self._greeting_start_time > 35.0:
                self._greeting_in_progress = False
                return False
            return True

    def set_greeting_callback(
        self,
        cb: Callable[[str, str, float, dict[str, Any] | None], None],
    ) -> None:
        """注册主动打招呼执行函数: cb(identity, label, score, face_analysis)"""
        self._greeting_callback = cb

    def set_guard_alert_callback(
        self,
        cb: Callable[[float, dict[str, Any] | None, str], None],
    ) -> None:
        """注册警惕模式陌生人闯入警报执行函数: cb(score, face_analysis, snapshot_data_uri)"""
        self._guard_alert_callback = cb

    def set_busy_checkers(
        self,
        speaker_checker: Callable[[], bool] | None = None,
        mic_checker: Callable[[], bool] | None = None,
    ) -> None:
        """注册扬声器忙碌及麦克风忙碌检测，避开插嘴"""
        self._speaker_busy_checker = speaker_checker
        self._mic_busy_checker = mic_checker

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def toggle(self, state: bool | None = None) -> bool:
        """切换或设置摄像头哨兵开关状态"""
        with self._lock:
            if state is None:
                self._enabled = not self._enabled
            else:
                self._enabled = bool(state)
            cur = self._enabled

        logger.info(f"[Sentinel] Camera state changed to: {'ENABLED' if cur else 'PAUSED'}")
        return cur

    def is_greeting_enabled(self) -> bool:
        with self._lock:
            return self._greeting_enabled

    def is_cooldown_enabled(self) -> bool:
        return self.is_greeting_enabled()

    def toggle_greeting(self, state: bool | None = None) -> bool:
        """切换或设置主动打招呼开关（开启=识别人脸主动迎候+智能冷却；关闭=安静守望，完全不主动打扰）"""
        with self._lock:
            if state is None:
                self._greeting_enabled = not self._greeting_enabled
            else:
                self._greeting_enabled = bool(state)
            cur = self._greeting_enabled

        logger.info(f"[Sentinel] Greeting switch changed to: {'ENABLED' if cur else 'DISABLED (Silent Watch)'}")
        return cur

    def toggle_cooldown(self, state: bool | None = None) -> bool:
        return self.toggle_greeting(state)

    def _load_guard_config(self) -> None:
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._guard_enabled = bool(data.get("guard_enabled", False))
                self._guard_start_time = str(data.get("start_time", "10:00"))
                self._guard_end_time = str(data.get("end_time", "20:00"))
                self._guard_alert_count = int(data.get("alert_count", 3))
                self._guard_alert_duration_s = int(data.get("alert_duration_s", 5))
                logger.info(
                    f"[Sentinel] Loaded guard config: enabled={self._guard_enabled}, "
                    f"window={self._guard_start_time}-{self._guard_end_time}, "
                    f"count={self._guard_alert_count}, duration={self._guard_alert_duration_s}s"
                )
        except Exception as exc:
            logger.warning(f"[Sentinel] Failed to load guard config: {exc}")

    def _save_guard_config(self) -> None:
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "guard_enabled": self._guard_enabled,
                "start_time": self._guard_start_time,
                "end_time": self._guard_end_time,
                "alert_count": self._guard_alert_count,
                "alert_duration_s": self._guard_alert_duration_s,
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"[Sentinel] Failed to save guard config: {exc}")

    def get_guard_config(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._guard_enabled,
                "start_time": self._guard_start_time,
                "end_time": self._guard_end_time,
                "alert_count": self._guard_alert_count,
                "alert_duration_s": self._guard_alert_duration_s,
                "is_in_guard_period": self.is_in_guard_period(),
            }

    def update_guard_config(
        self,
        enabled: bool | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        alert_count: int | None = None,
        alert_duration_s: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if enabled is not None:
                self._guard_enabled = bool(enabled)
            if start_time is not None and ":" in str(start_time):
                self._guard_start_time = str(start_time).strip()
            if end_time is not None and ":" in str(end_time):
                self._guard_end_time = str(end_time).strip()
            if alert_count is not None:
                self._guard_alert_count = max(1, min(10, int(alert_count)))
            if alert_duration_s is not None:
                self._guard_alert_duration_s = max(1, min(30, int(alert_duration_s)))
            self._save_guard_config()
            return {
                "enabled": self._guard_enabled,
                "start_time": self._guard_start_time,
                "end_time": self._guard_end_time,
                "alert_count": self._guard_alert_count,
                "alert_duration_s": self._guard_alert_duration_s,
                "is_in_guard_period": self.is_in_guard_period(),
            }

    def toggle_guard(self, state: bool | None = None) -> bool:
        with self._lock:
            if state is None:
                self._guard_enabled = not self._guard_enabled
            else:
                self._guard_enabled = bool(state)
            self._save_guard_config()
            cur = self._guard_enabled
        logger.info(f"[Sentinel] Guard mode changed to: {'ENABLED' if cur else 'DISABLED'}")
        return cur

    def is_in_guard_period(self) -> bool:
        with self._lock:
            if not self._guard_enabled:
                return False
            st = self._guard_start_time
            et = self._guard_end_time

        try:
            now = datetime.now()
            cur_min = now.hour * 60 + now.minute
            sh, sm = map(int, st.split(":"))
            eh, em = map(int, et.split(":"))
            s_min = sh * 60 + sm
            e_min = eh * 60 + em
            if s_min <= e_min:
                return s_min <= cur_min < e_min
            else:
                # 跨天时间段，例如 22:00 到 07:00
                return cur_min >= s_min or cur_min < e_min
        except Exception:
            return False

    def capture_snapshot_data_uri(self) -> str:
        with self._lock:
            if self._latest_annotated_jpeg:
                b64 = base64.b64encode(self._latest_annotated_jpeg).decode("ascii")
                return f"data:image/jpeg;base64,{b64}"
            elif self._latest_clean_frame is not None:
                ret, buf = cv2.imencode(".jpg", self._latest_clean_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
        return ""

    def get_latest_clean_frame(self) -> np.ndarray | None:
        """为 companion_vision 拍照提供 0ms 极速内存帧"""
        with self._lock:
            if self._latest_clean_frame is not None:
                return self._latest_clean_frame.copy()
        return None

    def get_status(self) -> dict[str, Any]:
        """获取当前哨兵监控状态"""
        now = time.time()
        with self._lock:
            locked = None
            if self._last_face_analysis and self._last_face_analysis.get("locked_target"):
                lt = self._last_face_analysis["locked_target"]
                locked = {
                    "identity": lt.get("identity"),
                    "label": lt.get("label"),
                    "display_badge": lt.get("display_badge"),
                    "confidence": lt.get("confidence", 0.0),
                }

            # 主人冷却倒计时（仅在开启主动打招呼且处于冷却期时计算）
            master_cooldown_rem = 0.0
            if self._greeting_enabled and self._master_last_greeted is not None:
                rem = self._master_cooldown_s - (now - self._master_last_greeted)
                if rem > 0:
                    master_cooldown_rem = round(rem, 1)

            return {
                "enabled": self._enabled,
                "camera_ok": self._camera_ok,
                "greeting_enabled": self._greeting_enabled,
                "cooldown_enabled": self._greeting_enabled,
                "locked_target": locked,
                "master_present": self._master_present,
                "master_cooldown_remaining_s": master_cooldown_rem,
                "stranger_present": self._stranger_present,
                "guard_enabled": self._guard_enabled,
                "is_in_guard_period": self.is_in_guard_period(),
                "guard_alerting": self._guard_alerting,
                "guard_start_time": self._guard_start_time,
                "guard_end_time": self._guard_end_time,
                "guard_alert_count": self._guard_alert_count,
                "guard_alert_duration_s": self._guard_alert_duration_s,
            }

    def record_greeting(self, greeting_data: dict[str, Any]) -> None:
        """记录最近的主动打招呼，供前端刷新或新打开标签页拉取"""
        with self._lock:
            self._recent_greetings.append(greeting_data)
            if len(self._recent_greetings) > 10:
                self._recent_greetings = self._recent_greetings[-10:]

    def get_recent_greetings(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._recent_greetings)

    # ------------------------------------------------------------------
    # 视频帧订阅机制 (MJPEG 视频流)
    # ------------------------------------------------------------------
    def add_frame_subscriber(self, q: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.add(q)
            # 立即投递最新一帧，避免初次连接黑屏
            if self._latest_annotated_jpeg:
                try:
                    q.put_nowait(self._latest_annotated_jpeg)
                except Exception:
                    pass

    def remove_frame_subscriber(self, q: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _broadcast_frame(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                # 保持实时性：若队列满则丢弃旧帧
                if q.full():
                    try:
                        q.get_nowait()
                    except Exception:
                        pass
                q.put_nowait(jpeg_bytes)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 摄像头硬件生命周期
    # ------------------------------------------------------------------
    def _open_camera(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        try:
            cap = cv2.VideoCapture(self._camera_index)
            if not cap.isOpened():
                # 尝试副摄像头 /dev/video1
                cap = cv2.VideoCapture(1)
                if cap.isOpened():
                    self._camera_index = 1

            if not cap.isOpened():
                self._camera_ok = False
                return False

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
            self._cap = cap
            self._camera_ok = True
            logger.info(f"[Sentinel] Camera opened successfully (/dev/video{self._camera_index})")
            return True
        except Exception as exc:
            logger.warning(f"[Sentinel] Camera open error: {exc}")
            self._camera_ok = False
            return False

    def _close_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._camera_ok = False

    def _make_standby_frame(self, text: str = "实时视线已休眠") -> bytes:
        """生成带科技感待机占位图"""
        img = np.zeros((360, 480, 3), dtype=np.uint8)
        # 背景微渐变网格
        for y in range(0, 360, 30):
            cv2.line(img, (0, y), (480, y), (18, 26, 38), 1)
        for x in range(0, 480, 30):
            cv2.line(img, (x, 0), (x, 360), (18, 26, 38), 1)

        # 居中文本（使用英文保证无中文点阵缺失，或简单状态）
        cv2.putText(
            img,
            "SENTINEL STANDBY",
            (145, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 240, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            (175, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        return buf.tobytes()

    # ------------------------------------------------------------------
    # 核心后台工作主循环
    # ------------------------------------------------------------------
    def _capture_and_detect_loop(self) -> None:
        logger.info("[Sentinel] Background sentinel loop started")
        standby_bytes = self._make_standby_frame("点击小窗开启守望")

        while self._running:
            if not self._enabled:
                self._close_camera()
                self._broadcast_frame(standby_bytes)
                time.sleep(0.3)
                continue

            if not self._open_camera():
                no_cam_bytes = self._make_standby_frame("未检测到可用摄像头")
                self._broadcast_frame(no_cam_bytes)
                time.sleep(1.0)
                continue

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    time.sleep(0.04)
                    continue

                # 缓存干净原始帧
                with self._lock:
                    self._latest_clean_frame = frame.copy()

                now = time.time()
                # 周期性人脸检测比对 (约1.2s 一次)
                if now - self._last_detect_time >= self._detect_interval_s:
                    self._last_detect_time = now
                    try:
                        face_res = companion_face.analyze_frame(frame)
                        with self._lock:
                            self._last_face_analysis = face_res
                        self._process_cooldown_and_greeting(face_res, now)
                    except Exception as face_err:
                        logger.debug(f"[Sentinel] Analyze frame error: {face_err}")

                # 渲染预览画面（水平镜像翻转，符合自拍镜面直觉）
                preview = cv2.flip(frame, 1)
                h, w = preview.shape[:2]

                # 绘制全息青色对焦框
                with self._lock:
                    analysis = self._last_face_analysis

                if analysis and analysis.get("locked_target"):
                    lt = analysis["locked_target"]
                    bbox = lt.get("bbox")
                    if bbox and len(bbox) == 4:
                        # 原图坐标
                        ox1, oy1, ox2, oy2 = bbox
                        # 水平镜像后对应的坐标
                        mx1 = max(0, w - ox2)
                        mx2 = min(w, w - ox1)
                        my1 = max(0, oy1)
                        my2 = min(h, oy2)
                        self._draw_cyber_brackets(preview, mx1, my1, mx2, my2)

                # JPEG 编码推流
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 68]
                ok, buf = cv2.imencode(".jpg", preview, encode_params)
                if ok:
                    jpeg_bytes = buf.tobytes()
                    with self._lock:
                        self._latest_annotated_jpeg = jpeg_bytes
                    self._broadcast_frame(jpeg_bytes)

                # 维持 ~18-20 fps
                time.sleep(0.045)
            except Exception as loop_err:
                logger.warning(f"[Sentinel] Frame loop error: {loop_err}")
                time.sleep(0.1)

    # ------------------------------------------------------------------
    # 极简科技感青色对焦边角绘制
    # ------------------------------------------------------------------
    def _draw_cyber_brackets(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """在人脸四周绘制四角科技感瞄准标 (无文字干扰，极简现代)"""
        w = max(10, x2 - x1)
        h = max(10, y2 - y1)
        cor = max(10, int(min(w, h) * 0.22))
        cyan = (255, 240, 0)  # BGR #00f0ff
        thick = 2

        # 4角青色标
        # 左上
        cv2.line(img, (x1, y1), (x1 + cor, y1), cyan, thick)
        cv2.line(img, (x1, y1), (x1, y1 + cor), cyan, thick)
        # 右上
        cv2.line(img, (x2, y1), (x2 - cor, y1), cyan, thick)
        cv2.line(img, (x2, y1), (x2, y1 + cor), cyan, thick)
        # 左下
        cv2.line(img, (x1, y2), (x1 + cor, y2), cyan, thick)
        cv2.line(img, (x1, y2), (x1, y2 - cor), cyan, thick)
        # 右下
        cv2.line(img, (x2, y2), (x2 - cor, y2), cyan, thick)
        cv2.line(img, (x2, y2), (x2, y2 - cor), cyan, thick)

    # ------------------------------------------------------------------
    # 核心防骚扰 / 智能冷却状态机 (Anti-loop / Cooldown Engine)
    # ------------------------------------------------------------------
    def _can_speak_now(self) -> bool:
        """避麦检测：如果伴侣正在讲话、Breeze未就绪、正在执行迎候播报或用户正在麦克风说话，不插嘴"""
        if self.is_greeting_in_progress():
            return False
        if self._speaker_busy_checker:
            try:
                if self._speaker_busy_checker():
                    return False
            except Exception:
                return False
        if self._mic_busy_checker:
            try:
                if self._mic_busy_checker():
                    return False
            except Exception:
                return False
        return True

    def _process_cooldown_and_greeting(self, face_res: dict[str, Any], now: float) -> None:
        """根据当前帧检测结果，严密执行防刷打招呼判定与非打断礼貌接续"""
        locked = face_res.get("locked_target")

        # 1. 镜头前无人脸
        if not locked:
            self._master_no_face_count += 1
            if self._master_no_face_count >= 5:  # 连续 ~6 秒未见主人
                if self._master_present:
                    self._master_present = False
                    self._master_absent_since = now
                    logger.info("[Sentinel] Master stepped away (marked ABSENT)")

            self._stranger_no_face_count += 1
            if self._stranger_no_face_count >= 5:
                if self._stranger_present:
                    self._stranger_present = False

            return

        all_targets = face_res.get("targets", [])
        has_master = any(t.get("identity") == "master" for t in all_targets)
        has_stranger = any(t.get("identity") == "stranger" for t in all_targets)
        has_friend = any(t.get("identity") == "friend" for t in all_targets)

        identity = locked.get("identity", "stranger")
        label = locked.get("label", "新朋友")
        score = locked.get("confidence", 0.0)

        # 核心安全联动：若主人现身，且当前正处于警惕模式报警中，立即停止警报（主人回来了）
        if has_master and self._guard_alerting:
            logger.info("[Sentinel] Master returned during guard alert! Cancelling alert sirens.")
            self._cancel_alert_event.set()
            self._guard_alerting = False

        # 核心：时间段警惕模式判定（若无主人在场且有陌生人，在警惕时段内触发主动警报与抓拍）
        if not has_master and has_stranger and self.is_in_guard_period():
            self._stranger_no_face_count = 0
            self._stranger_present = True
            time_since_alert = (now - self._guard_last_alert_time)
            if not self._guard_alerting and time_since_alert >= self._guard_alert_cooldown_s:
                self._guard_alerting = True
                self._guard_last_alert_time = now
                self._cancel_alert_event.clear()
                snapshot_uri = self.capture_snapshot_data_uri()
                logger.warning(f"[Sentinel] >>> GUARD ALERT TRIGGERED FOR STRANGER (score={score:.2f})!")
                self._dispatch_guard_alert(score=score, face_analysis=face_res, snapshot_data_uri=snapshot_uri)
            return

        # 核心：若用户关闭了主动迎候开关，则进入【安静守望模式】，只追踪人脸，彻底静音绝不主动打招呼！
        if not self._greeting_enabled:
            if has_master:
                self._master_no_face_count = 0
                self._master_last_seen = now
                self._master_present = True
            if has_friend:
                for t in all_targets:
                    if t.get("identity") == "friend":
                        f_id = t.get("target_id") or t.get("label", "friend")
                        st = self._friend_states.setdefault(
                            f_id,
                            {"present": True, "last_greeted": None, "absent_since": None, "no_face_count": 0},
                        )
                        st["no_face_count"] = 0
                        st["present"] = True
            if has_stranger:
                self._stranger_no_face_count = 0
                self._stranger_present = True
            return

        # =========================================================================
        # 核心防掐断与礼貌顺畅接续机制 (Anti-Cutoff & Smooth Sequential Follow-up)
        # =========================================================================

        # 机制 1：当前正有迎候播报在发声/生成中（坚决不掐断，完整读完当前台词）
        if self.is_greeting_in_progress():
            # 若正向新朋友问好时，主人现身镜头前：记录待接续问候主人，等当前语音播完再自然迎接
            if has_master:
                self._master_last_seen = now
                m_time_since = (now - self._master_last_greeted) if self._master_last_greeted is not None else 999999.0
                m_away_dur = (now - self._master_absent_since) if self._master_absent_since is not None else 999999.0
                m_was_absent = not self._master_present
                should_queue_master = (
                    self._master_last_greeted is None or
                    (m_was_absent and m_away_dur >= 15.0 and m_time_since >= 30.0) or
                    (m_time_since >= self._master_cooldown_s)
                )
                if should_queue_master:
                    self._pending_master_followup = True
                    logger.info("[Sentinel] Master present while greeting is in progress; queued for natural follow-up once speech finishes without cutoff.")
                self._master_present = True
            return

        # 机制 2：此前语音已完整读完，且有此前出现并静静等待的主人（顺畅接续欢迎主人）
        if self._pending_master_followup:
            if has_master:
                if self._can_speak_now():
                    self._pending_master_followup = False
                    self._master_present = True
                    self._master_last_greeted = now
                    self._master_absent_since = None
                    m_target = next((t for t in all_targets if t.get("identity") == "master"), locked)
                    m_score = m_target.get("confidence", score)
                    logger.info(f"[Sentinel] >>> Seamlessly following up to greet master (score={m_score:.2f})")
                    face_ctx = dict(face_res)
                    face_ctx["followup_after_stranger"] = True
                    self._dispatch_greeting(identity="master", label="主人", score=m_score, face_analysis=face_ctx)
                    return
            else:
                if self._master_last_seen and (now - self._master_last_seen > 12.0):
                    self._pending_master_followup = False

        # 机制 3：画面中同时出现【主人】与【新朋友/好友】（合并温暖综合问候，同时兼顾两人）
        if has_master and (has_stranger or has_friend):
            m_time_since = (now - self._master_last_greeted) if self._master_last_greeted is not None else 999999.0
            s_time_since = (now - self._stranger_last_greeted) if self._stranger_last_greeted is not None else 999999.0
            should_both_greet = (
                self._master_last_greeted is None or
                m_time_since >= self._master_cooldown_s or
                self._stranger_last_greeted is None or
                s_time_since >= self._stranger_cooldown_s
            )
            if should_both_greet and self._can_speak_now():
                self._master_present = True
                self._master_last_greeted = now
                self._master_absent_since = None
                self._stranger_present = True
                self._stranger_last_greeted = now
                self._stranger_absent_since = None
                self._pending_master_followup = False
                combined_label = "主人与新朋友" if has_stranger else "主人与朋友"
                logger.info("[Sentinel] >>> Both master and friend/stranger present simultaneously! Dispatching unified warm greeting.")
                self._dispatch_greeting(identity="both", label=combined_label, score=score, face_analysis=face_res)
                return

        # 机制 4：常规单人目标判定（保持智能 60s/90s 冷却防骚扰机制，不重复刷屏）
        if identity == "master":
            self._master_no_face_count = 0
            self._master_last_seen = now

            was_absent = not self._master_present
            time_since_greet = (now - self._master_last_greeted) if self._master_last_greeted is not None else 999999.0
            away_dur = (now - self._master_absent_since) if self._master_absent_since is not None else 999999.0

            should_greet = False
            if self._master_last_greeted is None:
                should_greet = True
            elif was_absent and away_dur >= 15.0 and time_since_greet >= 30.0:
                should_greet = True
            elif time_since_greet >= self._master_cooldown_s:
                should_greet = True

            if should_greet:
                if self._can_speak_now():
                    self._master_present = True
                    self._master_last_greeted = now
                    self._master_absent_since = None
                    logger.info(f"[Sentinel] >>> TRIGGERING GREETING FOR MASTER (score={score:.2f})")
                    self._dispatch_greeting(identity="master", label="主人", score=score, face_analysis=face_res)
                else:
                    logger.debug("[Sentinel] Master detected but speaker/mic busy, waiting next frame")
            else:
                self._master_present = True

        elif identity == "friend":
            friend_id = locked.get("target_id") or label
            st = self._friend_states.setdefault(
                friend_id,
                {"present": False, "last_greeted": None, "absent_since": None, "no_face_count": 0},
            )
            st["no_face_count"] = 0
            was_absent = not st["present"]
            time_since_greet = (now - st["last_greeted"]) if st["last_greeted"] is not None else 999999.0
            away_dur = (now - st["absent_since"]) if st["absent_since"] is not None else 999999.0

            should_greet = False
            if st["last_greeted"] is None:
                should_greet = True
            elif was_absent and away_dur >= 15.0 and time_since_greet >= 30.0:
                should_greet = True
            elif time_since_greet >= self._friend_cooldown_s:
                should_greet = True

            if should_greet:
                if self._can_speak_now():
                    st["present"] = True
                    st["last_greeted"] = now
                    st["absent_since"] = None
                    logger.info(f"[Sentinel] >>> TRIGGERING GREETING FOR FRIEND {label} (score={score:.2f})")
                    self._dispatch_greeting(identity="friend", label=label, score=score, face_analysis=face_res)
            else:
                st["present"] = True

        else:
            self._stranger_no_face_count = 0
            was_absent = not self._stranger_present
            time_since_greet = (now - self._stranger_last_greeted) if self._stranger_last_greeted is not None else 999999.0
            away_dur = (now - self._stranger_absent_since) if self._stranger_absent_since is not None else 999999.0

            should_greet = False
            if self._stranger_last_greeted is None:
                should_greet = True
            elif was_absent and away_dur >= 15.0 and time_since_greet >= 30.0:
                should_greet = True
            elif time_since_greet >= self._stranger_cooldown_s:
                should_greet = True

            if should_greet:
                if self._can_speak_now():
                    self._stranger_present = True
                    self._stranger_last_greeted = now
                    self._stranger_absent_since = None
                    logger.info(f"[Sentinel] >>> TRIGGERING GREETING FOR STRANGER (score={score:.2f})")
                    self._dispatch_greeting(identity="stranger", label="新朋友", score=score, face_analysis=face_res)
            else:
                self._stranger_present = True

    def _dispatch_greeting(
        self,
        identity: str,
        label: str,
        score: float,
        face_analysis: dict[str, Any] | None,
    ) -> None:
        """异步触发主动问候，不阻塞摄像头采集，并立即预占迎候状态锁"""
        if self._greeting_callback is None:
            return

        self.set_greeting_in_progress(True)

        def _runner() -> None:
            try:
                self._greeting_callback(identity, label, score, face_analysis)
            except Exception as exc:
                logger.error(f"[Sentinel] Greeting callback error: {exc}")
                self.set_greeting_in_progress(False)

        threading.Thread(target=_runner, daemon=True, name="SentinelGreetingRunner").start()

    def _dispatch_guard_alert(
        self,
        score: float,
        face_analysis: dict[str, Any] | None,
        snapshot_data_uri: str,
    ) -> None:
        """异步触发警惕模式陌生人闯入警报"""
        if self._guard_alert_callback is None:
            self._guard_alerting = False
            return

        def _runner() -> None:
            try:
                self._guard_alert_callback(score, face_analysis, snapshot_data_uri)
            except Exception as exc:
                logger.error(f"[Sentinel] Guard alert callback error: {exc}")
                self._guard_alerting = False

        threading.Thread(target=_runner, daemon=True, name="SentinelGuardAlertRunner").start()


# 全局单例
companion_sentinel = CompanionSentinelEngine()
