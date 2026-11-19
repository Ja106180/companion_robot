"""
companion_alarm.py - 桌面毛绒伴生灵宠智能早晨唤醒与知识库日程备忘中枢

核心能力：
1. 本地硬件时钟精准调度：基于电脑系统时间，每秒高精度心跳检测闹钟触发
2. 早晨唤醒提醒 (Morning Wakeup)：
   - 设定时间（如早上 06:00）触发高保真清脆晶莹“叮叮叮”敲钟铃声
   - 铃声持续响铃，支持手动一键停止，或达到 5 分钟（300秒）超时自动关停
   - 关停后立即自动全息播报今日实时气象（气温、雨晴、穿衣出行建议）与今日实时热点要闻
3. 知识库日程备忘提醒 (Smart Knowledge Memo Alarm)：
   - 支持自定义设置时间与日程标签（如 14:30 项目周会、18:00 健身备忘）
   - 专属知识库内容输入框：输入提醒详细背景、核心注意事项、关键资料等
   - 响铃关停后，由当前激活的灵宠（棉棉/露玛/暖暖等）基于知识库内容进行智能个性化口语播报
4. 双端全息同步：PC 主端与手机移动端实时接收响铃倒计时、停止操作与晨报/备忘卡片
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import struct
import sys
import threading
import time
import wave
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("companion_alarm")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [AlarmAgent] %(message)s"))
    logger.addHandler(_handler)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ALARMS_FILE = os.path.join(DATA_DIR, "alarms.json")
SAMPLE_RATE = 24000
MAX_RINGING_DURATION_S = 300.0  # 5分钟自动关停阈值


def generate_crystal_chime_pcm(sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    纯 Python 算法合成 24kHz 16-bit 单声道晶莹铜钟/座钟“叮叮叮”三音和弦铃声。
    纯净正弦波叠加金属泛音与指数衰减，清脆悦耳，不依赖任何第三方音效文件。
    """
    notes = [1046.5, 1318.51, 1567.98]  # C6, E6, G6 清脆明亮高音三和弦
    chime_pcm = bytearray()

    for freq in notes:
        duration = 0.82  # 每个单音持续约 0.82 秒
        n_samples = int(sample_rate * duration)
        for i in range(n_samples):
            t = i / sample_rate
            # 钟声包络：快速起振与指数级自然衰减
            env = math.exp(-t * 4.6)
            # 基频 + 二次谐波 + 金属泛音 (2.76f 与 5.4f 产生真实铜铃清脆回响)
            sample = (
                0.62 * math.sin(2 * math.pi * freq * t)
                + 0.22 * math.sin(2 * math.pi * freq * 2.0 * t)
                + 0.12 * math.sin(2 * math.pi * freq * 2.76 * t)
                + 0.04 * math.sin(2 * math.pi * freq * 5.4 * t)
            ) * env
            val = int(max(-1.0, min(1.0, sample)) * 28000)
            chime_pcm.extend(struct.pack("<h", val))

        # 每次“叮”之间的短暂间隔
        silence_samples = int(sample_rate * 0.12)
        chime_pcm.extend(b"\x00\x00" * silence_samples)

    # 句尾自然混响空隙
    tail_silence = int(sample_rate * 0.45)
    chime_pcm.extend(b"\x00\x00" * tail_silence)

    return bytes(chime_pcm)


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


class CompanionAlarmEngine:
    def __init__(self, alarms_path: str = ALARMS_FILE):
        self._alarms_path = alarms_path
        self._lock = threading.RLock()
        self._alarms: list[dict[str, Any]] = []

        # 预先生成清脆“叮叮叮”铃声 PCM 与 Base64 WAV
        self._chime_pcm = generate_crystal_chime_pcm(SAMPLE_RATE)
        self._chime_wav_bytes = pcm_to_wav_bytes(self._chime_pcm, SAMPLE_RATE)
        self._chime_wav_b64 = base64.b64encode(self._chime_wav_bytes).decode("ascii")

        # 响铃会话状态
        self._ringing = False
        self._ringing_alarm: dict[str, Any] | None = None
        self._ringing_start_time = 0.0
        self._stop_event = threading.Event()
        self._ring_thread: threading.Thread | None = None

        # 外部联动回调
        self.speaker_player: Any | None = None
        self.broadcast_callback: Callable[[dict[str, Any]], None] | None = None
        self.trigger_briefing_callback: Callable[[dict[str, Any], str], None] | None = None

        self._load_alarms()
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info(f"CompanionAlarmEngine initialized with {len(self._alarms)} alarms.")

    @property
    def chime_wav_b64(self) -> str:
        return self._chime_wav_b64

    def _get_default_alarms(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "morning_alarm_default",
                "time": "06:00",
                "label": "早晨起床唤醒与晨报",
                "type": "morning",  # "morning" (早晨唤醒晨报) | "memo" (知识库日程备忘)
                "enabled": True,
                "content": "早起喝一杯温开水润喉，整理今日重要日程，出门前检查手机、工卡和钥匙。",
                "repeat": "daily",  # "daily" | "once"
                "last_triggered_date": "",
            },
            {
                "id": "sample_memo_alarm",
                "time": "14:30",
                "label": "下午重要项目会议备忘",
                "type": "memo",
                "enabled": False,
                "content": "下午与核心团队讨论智能灵宠项目阶段性进展，重点分享人脸视觉哨兵与晨报闹钟的落地效果，提前备好演示环境。",
                "repeat": "daily",
                "last_triggered_date": "",
            },
        ]

    def _load_alarms(self) -> None:
        with self._lock:
            if not os.path.exists(self._alarms_path):
                self._alarms = self._get_default_alarms()
                self._save_alarms_unlocked()
                return

            try:
                with open(self._alarms_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._alarms = data
                    elif isinstance(data, dict) and "alarms" in data:
                        self._alarms = data["alarms"]
                    else:
                        self._alarms = self._get_default_alarms()
            except Exception as exc:
                logger.warning(f"Failed to load alarms from {self._alarms_path}: {exc}. Using defaults.")
                self._alarms = self._get_default_alarms()

    def _save_alarms_unlocked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._alarms_path), exist_ok=True)
            with open(self._alarms_path, "w", encoding="utf-8") as f:
                json.dump(self._alarms, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save alarms to {self._alarms_path}: {exc}")

    def get_alarms(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(a) for a in self._alarms]

    def get_alarm_by_id(self, alarm_id: str) -> dict[str, Any] | None:
        with self._lock:
            for a in self._alarms:
                if a.get("id") == alarm_id:
                    return dict(a)
            return None

    def add_alarm(
        self,
        time_str: str,
        label: str = "",
        alarm_type: str = "morning",
        content: str = "",
        repeat: str = "daily",
        enabled: bool = True,
    ) -> dict[str, Any]:
        """新增闹钟"""
        # 校验并格式化时间 HH:MM
        clean_time = time_str.strip()
        parts = clean_time.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid time format: {time_str}, expected HH:MM")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid time values: {time_str}")
        formatted_time = f"{hour:02d}:{minute:02d}"

        alarm_id = f"alarm_{int(time.time() * 1000)}"
        if not label.strip():
            label = "早晨起床唤醒" if alarm_type == "morning" else "日程知识备忘"

        new_alarm = {
            "id": alarm_id,
            "time": formatted_time,
            "label": label.strip(),
            "type": alarm_type if alarm_type in ("morning", "memo") else "memo",
            "enabled": bool(enabled),
            "content": content.strip(),
            "repeat": "daily" if repeat == "daily" else "once",
            "last_triggered_date": "",
        }

        with self._lock:
            self._alarms.append(new_alarm)
            self._save_alarms_unlocked()
            logger.info(f"Added alarm: {new_alarm['id']} at {new_alarm['time']} ({new_alarm['label']})")
            return dict(new_alarm)

    def update_alarm(self, alarm_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """更新已有闹钟"""
        with self._lock:
            target = None
            for a in self._alarms:
                if a.get("id") == alarm_id:
                    target = a
                    break
            if not target:
                return None

            if "time" in updates:
                clean_time = str(updates["time"]).strip()
                parts = clean_time.split(":")
                if len(parts) == 2:
                    h, m = int(parts[0]), int(parts[1])
                    if 0 <= h <= 23 and 0 <= m <= 59:
                        target["time"] = f"{h:02d}:{m:02d}"

            if "label" in updates:
                target["label"] = str(updates["label"]).strip()
            if "type" in updates and updates["type"] in ("morning", "memo"):
                target["type"] = updates["type"]
            if "content" in updates:
                target["content"] = str(updates["content"]).strip()
            if "repeat" in updates:
                target["repeat"] = "daily" if updates["repeat"] == "daily" else "once"
            if "enabled" in updates:
                target["enabled"] = bool(updates["enabled"])

            self._save_alarms_unlocked()
            logger.info(f"Updated alarm: {alarm_id}")
            return dict(target)

    def toggle_alarm(self, alarm_id: str, enabled: bool | None = None) -> dict[str, Any] | None:
        """切换闹钟开启/关闭"""
        with self._lock:
            for a in self._alarms:
                if a.get("id") == alarm_id:
                    if enabled is None:
                        a["enabled"] = not a.get("enabled", True)
                    else:
                        a["enabled"] = bool(enabled)
                    self._save_alarms_unlocked()
                    logger.info(f"Toggled alarm {alarm_id} -> {a['enabled']}")
                    return dict(a)
            return None

    def delete_alarm(self, alarm_id: str) -> bool:
        """删除闹钟"""
        with self._lock:
            initial_len = len(self._alarms)
            self._alarms = [a for a in self._alarms if a.get("id") != alarm_id]
            if len(self._alarms) < initial_len:
                self._save_alarms_unlocked()
                logger.info(f"Deleted alarm {alarm_id}")
                return True
            return False

    def get_status(self) -> dict[str, Any]:
        """获取闹钟系统整体状态"""
        with self._lock:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            remaining_s = 0.0
            if self._ringing:
                elapsed = time.time() - self._ringing_start_time
                remaining_s = max(0.0, MAX_RINGING_DURATION_S - elapsed)

            return {
                "ringing": self._ringing,
                "ringing_alarm": dict(self._ringing_alarm) if self._ringing_alarm else None,
                "ringing_remaining_s": round(remaining_s, 1),
                "server_time": current_time,
                "server_date": now.strftime("%Y-%m-%d"),
                "total_alarms": len(self._alarms),
                "active_alarms_count": sum(1 for a in self._alarms if a.get("enabled")),
            }

    def trigger_alarm_instant(self, alarm_id: str) -> bool:
        """即刻触发指定闹钟（用于测试或用户调试）"""
        alarm = self.get_alarm_by_id(alarm_id)
        if not alarm:
            return False
        if self._ringing:
            self._stop_event.set()
            time.sleep(0.15)
        self.start_ringing(alarm)
        return True

    def start_ringing(self, alarm: dict[str, Any]) -> None:
        """开始执行响铃会话：播放清脆叮叮叮铃声，广播事件，并在 5 分钟超时或手动关停后触发播报"""
        with self._lock:
            if self._ringing:
                self._stop_event.set()
                time.sleep(0.15)

            self._ringing = True
            self._ringing_alarm = dict(alarm)
            self._ringing_start_time = time.time()
            self._stop_event.clear()

        logger.info(f"[Alarm] Alarm started ringing: {alarm.get('label')} ({alarm.get('time')})")

        # 广播闹钟开始响铃事件至全网（PC 端与手机端同步呈现高能闪烁振铃卡片）
        if self.broadcast_callback:
            try:
                self.broadcast_callback(
                    {
                        "type": "alarm_ringing_started",
                        "alarm": dict(alarm),
                        "chime_wav_b64": self._chime_wav_b64,
                        "start_timestamp": self._ringing_start_time,
                        "max_duration_s": MAX_RINGING_DURATION_S,
                    }
                )
            except Exception as exc:
                logger.warning(f"Error broadcasting alarm ringing started: {exc}")

        # 启动后台响铃循环线程
        self._ring_thread = threading.Thread(
            target=self._ring_loop, args=(dict(alarm),), daemon=True
        )
        self._ring_thread.start()

    def _ring_loop(self, alarm: dict[str, Any]) -> None:
        """响铃循环：每隔约 3.5 秒敲响一次 C6-E6-G6 清脆铜钟，直到手动停止或满 5 分钟超时"""
        chime_pcm = self._chime_pcm
        chime_dur_s = len(chime_pcm) / 2 / SAMPLE_RATE

        stop_reason = "manual"
        try:
            while not self._stop_event.is_set():
                elapsed = time.time() - self._ringing_start_time
                if elapsed >= MAX_RINGING_DURATION_S:
                    stop_reason = "timeout"
                    logger.info("[Alarm] 5 minutes timeout reached, auto-stopping alarm chime!")
                    break

                # 本地喇叭播放晶莹叮叮叮铜钟铃声
                if self.speaker_player:
                    try:
                        gen_id = self.speaker_player.stop()
                        self.speaker_player.play_pcm(chime_pcm, gen_id)
                    except Exception as exc:
                        logger.warning(f"Error playing alarm chime on speaker: {exc}")

                # 等待这一段叮叮叮铃声播放完毕（支持随时被手动关停打断）
                end_t = time.time() + chime_dur_s + 0.3
                while time.time() < end_t:
                    if self._stop_event.is_set():
                        stop_reason = "manual"
                        break
                    time.sleep(0.08)

                if self._stop_event.is_set():
                    stop_reason = "manual"
                    break

                # 叮与叮之间的节奏空隙（约 1.2 秒），支持快速响应关停
                gap_t = time.time() + 1.2
                while time.time() < gap_t:
                    if self._stop_event.is_set():
                        stop_reason = "manual"
                        break
                    time.sleep(0.08)

        finally:
            with self._lock:
                self._ringing = False
                self._ringing_alarm = None

            if self.speaker_player:
                try:
                    self.speaker_player.stop()
                except Exception:
                    pass

            logger.info(f"[Alarm] Alarm ringing stopped (reason: {stop_reason}). Preparing voice briefing...")

            # 广播停止响铃事件
            if self.broadcast_callback:
                try:
                    self.broadcast_callback(
                        {
                            "type": "alarm_ringing_stopped",
                            "alarm_id": alarm.get("id"),
                            "reason": stop_reason,
                        }
                    )
                except Exception as exc:
                    logger.warning(f"Error broadcasting alarm ringing stopped: {exc}")

            # 关停后立即自动触发今日天气/新闻晨报或知识库备忘语音播报
            if self.trigger_briefing_callback:
                try:
                    self.trigger_briefing_callback(alarm, stop_reason)
                except Exception as exc:
                    logger.error(f"Error executing alarm briefing callback: {exc}")

    def stop_ringing(self, reason: str = "manual") -> bool:
        """手动一键停止正在响铃的闹钟"""
        with self._lock:
            if not self._ringing:
                return False
            self._stop_event.set()
            return True

    def _scheduler_loop(self) -> None:
        """后台高精度秒级调度线程：检测当前电脑时间并比对闹钟"""
        logger.info("Alarm scheduler background loop started.")
        while self._running:
            try:
                now = datetime.now()
                current_hm = now.strftime("%H:%M")
                current_date = now.strftime("%Y-%m-%d")

                with self._lock:
                    if not self._ringing:
                        for alarm in self._alarms:
                            if not alarm.get("enabled"):
                                continue
                            if alarm.get("time") == current_hm:
                                if alarm.get("last_triggered_date") != current_date:
                                    # 命中触发时间！
                                    alarm["last_triggered_date"] = current_date
                                    if alarm.get("repeat") == "once":
                                        alarm["enabled"] = False
                                    self._save_alarms_unlocked()

                                    logger.info(
                                        f"[Alarm] Matched alarm {alarm.get('label')} at {current_hm}! Starting ringing..."
                                    )
                                    # 启动响铃
                                    threading.Thread(
                                        target=self.start_ringing,
                                        args=(dict(alarm),),
                                        daemon=True,
                                    ).start()
                                    break
            except Exception as exc:
                logger.error(f"Error in alarm scheduler loop: {exc}")

            time.sleep(1.0)


# 全局单例
companion_alarm = CompanionAlarmEngine()
