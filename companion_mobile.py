"""
全息伴生舱 · 手机端无线遥控与公网穿透同步中枢 (Companion Mobile & Tunnel Hub)
=============================================================================
功能特性：
1. 自动探测本机真实局域网 Wi-Fi IP（过滤 127.0.0.1 及 198.18.x.x 代理虚拟网卡），生成局域网直连地址。
2. 内置 Cloudflare Tunnel (`cloudflared --protocol http2`) 免费公网内网穿透守护进程，自动捕获公网 HTTPS 链接。
3. 本地极速生成高清晰度 Base64 二维码（支持局域网扫码与 4G/5G 公网穿透扫码双模式）。
4. 基于 SSE (Server-Sent Events) 实时双向事件总线：
   - 手机端发文字 / 切换灵宠 / 触发齐舞 -> 毫秒级推送至电脑端 Web 主舞台自动上屏并语音回答。
   - 电脑端灵宠流式回复与联网/记忆状态 -> 实时回传至手机端屏幕同步显示。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import qrcode

logger = logging.getLogger("CompanionMobile")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [MobileHub] %(message)s"))
    logger.addHandler(_h)

BASE_DIR = Path(__file__).resolve().parent
CLOUDFLARED_BIN = BASE_DIR / "bin" / "cloudflared"


def get_real_lan_ip() -> str:
    """
    获取本机真实的局域网 IPv4 地址（优先 192.168.x.x / 10.x.x.x / 172.16-31.x.x，
    自动排除 127.x.x.x 回环地址与 198.18.x.x 代理 TUN 虚拟网卡地址）。
    """
    candidates: list[str] = []
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=2.0)
        for token in out.split():
            ip = token.strip()
            if not re.match(r"^\d+\.\d+\.\d+\.\d+$", ip):
                continue
            if ip.startswith("127.") or ip.startswith("198.18.") or ip.startswith("198.19.") or ip.startswith("169.254."):
                continue
            if ip.startswith("172.17."):  # Docker default bridge
                continue
            candidates.append(ip)
    except Exception:
        pass

    # 优先选择 192.168.*
    for ip in candidates:
        if ip.startswith("192.168."):
            return ip
    for ip in candidates:
        if ip.startswith("10.") or ip.startswith("172."):
            return ip
    if candidates:
        return candidates[0]

    # 备用探测法
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("223.5.5.5", 80))
        ip = s.getsockname()[0]
        s.close()
        if not ip.startswith("198.18."):
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def generate_qr_data_uri(url: str) -> str:
    """将指定 URL 编码为高清 PNG Base64 Data URI 二维码"""
    if not url:
        return ""
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#050811", back_color="#ffffff")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class CompanionMobileHub:
    """管理手机端与电脑 Web 主舞台的实时同步总线及免费公网内网穿透"""

    def __init__(self, port: int = 7860) -> None:
        self.port = port
        self._lock = threading.Lock()
        # client_id -> {"role": "pc" | "mobile", "queue": queue.Queue, "connected_at": float}
        self._clients: dict[str, dict[str, Any]] = {}
        self.active_pet_id: str = "luma"
        self.history_log: list[dict[str, Any]] = []

        # Cloudflare Tunnel 状态
        self._tunnel_proc: subprocess.Popen | None = None
        self._tunnel_url: str = ""
        self._tunnel_status: str = "stopped"  # "starting" | "running" | "error" | "stopped"
        self._tunnel_error: str = ""
        self._qr_cache: dict[str, str] = {}
        self._watchdog_started: bool = False
        self._start_watchdog_if_needed()

    def _start_watchdog_if_needed(self) -> None:
        if self._watchdog_started:
            return
        self._watchdog_started = True
        t = threading.Thread(target=self._watchdog_loop, daemon=True, name="TunnelWatchdog")
        t.start()

    def _watchdog_loop(self) -> None:
        """后台健康看门狗：每 25 秒检测公网隧道连通性，若边缘失效自动重启并刷新全网二维码"""
        fail_count = 0
        while True:
            time.sleep(25)
            url = ""
            status = ""
            with self._lock:
                url = self._tunnel_url
                status = self._tunnel_status
            if status != "running" or not url:
                fail_count = 0
                continue

            is_healthy = False
            try:
                req = urllib.request.Request(
                    f"{url}/mobile",
                    headers={"User-Agent": "Mozilla/5.0 (CompanionTunnelWatchdog)"},
                )
                with urllib.request.urlopen(req, timeout=6) as resp:
                    if resp.status == 200:
                        is_healthy = True
            except Exception as e:
                logger.debug(f"[TunnelWatchdog] Ping failed: {e}")

            if is_healthy:
                fail_count = 0
            else:
                fail_count += 1
                logger.warning(f"[TunnelWatchdog] Cloudflare tunnel check failed ({fail_count}/2): {url}")
                if fail_count >= 2:
                    logger.info("[TunnelWatchdog] Cloudflare tunnel session expired. Auto-regenerating fresh tunnel...")
                    fail_count = 0
                    self.start_tunnel_async(force_restart=True)

    # ------------------------------------------------------------------
    # 公网免费内网穿透 (Cloudflare Quick Tunnel)
    # ------------------------------------------------------------------
    def start_tunnel_async(self, force_restart: bool = False) -> None:
        self._start_watchdog_if_needed()
        with self._lock:
            if self._tunnel_proc and self._tunnel_proc.poll() is None:
                if not force_restart:
                    return
                try:
                    self._tunnel_proc.terminate()
                    self._tunnel_proc.wait(timeout=1.0)
                except Exception:
                    try:
                        self._tunnel_proc.kill()
                    except Exception:
                        pass
                self._tunnel_proc = None

            self._tunnel_status = "starting"
            self._tunnel_url = ""
            self._tunnel_error = ""

        t = threading.Thread(target=self._tunnel_worker, daemon=True)
        t.start()

    def _tunnel_worker(self) -> None:
        bin_path = str(CLOUDFLARED_BIN) if CLOUDFLARED_BIN.is_file() else "cloudflared"
        cmd = [
            bin_path,
            "tunnel",
            "--protocol",
            "http2",
            "--url",
            f"http://127.0.0.1:{self.port}",
        ]
        logger.info(f"Starting Cloudflare Quick Tunnel: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._tunnel_proc = proc

            url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            assert proc.stdout is not None
            for line in proc.stdout:
                m = url_pattern.search(line)
                if m:
                    found_url = m.group(0).strip()
                    with self._lock:
                        self._tunnel_url = found_url
                        self._tunnel_status = "running"
                    logger.info(f"Cloudflare Tunnel active: {found_url}/mobile")
                    self.broadcast({
                        "type": "tunnel_ready",
                        "public_base_url": found_url,
                        "public_mobile_url": f"{found_url}/mobile",
                    })
            # 若进程退出
            with self._lock:
                if self._tunnel_proc == proc:
                    self._tunnel_status = "error"
                    self._tunnel_error = "穿透进程已退出，请点击重新生成"
        except Exception as exc:
            logger.warning(f"Cloudflare Tunnel error: {exc}")
            with self._lock:
                self._tunnel_status = "error"
                self._tunnel_error = str(exc)

    def get_connection_info(self) -> dict[str, Any]:
        lan_ip = get_real_lan_ip()
        lan_base = f"http://{lan_ip}:{self.port}"
        lan_mobile_url = f"{lan_base}/mobile"

        if lan_mobile_url not in self._qr_cache:
            self._qr_cache[lan_mobile_url] = generate_qr_data_uri(lan_mobile_url)

        with self._lock:
            pub_base = self._tunnel_url
            t_status = self._tunnel_status
            t_err = self._tunnel_error
            pc_count = sum(1 for c in self._clients.values() if c["role"] == "pc")
            mobile_count = sum(1 for c in self._clients.values() if c["role"] == "mobile")

        pub_mobile_url = f"{pub_base}/mobile" if pub_base else ""
        pub_qr = ""
        if pub_mobile_url:
            if pub_mobile_url not in self._qr_cache:
                self._qr_cache[pub_mobile_url] = generate_qr_data_uri(pub_mobile_url)
            pub_qr = self._qr_cache[pub_mobile_url]

        return {
            "success": True,
            "lan_ip": lan_ip,
            "port": self.port,
            "lan_base_url": lan_base,
            "lan_mobile_url": lan_mobile_url,
            "lan_qr": self._qr_cache[lan_mobile_url],
            "tunnel_status": t_status,
            "tunnel_error": t_err,
            "public_base_url": pub_base,
            "public_mobile_url": pub_mobile_url,
            "public_qr": pub_qr,
            "pc_clients": pc_count,
            "mobile_clients": mobile_count,
            "active_pet_id": self.active_pet_id,
        }

    # ------------------------------------------------------------------
    # SSE 实时同步事件总线
    # ------------------------------------------------------------------
    def register_client(self, client_id: str, role: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._clients[client_id] = {
                "role": role,
                "queue": q,
                "connected_at": time.time(),
            }
            pc_count = sum(1 for c in self._clients.values() if c["role"] == "pc")
            mobile_count = sum(1 for c in self._clients.values() if c["role"] == "mobile")

        logger.info(f"Client connected: {client_id} (role={role}) | PC={pc_count}, Mobile={mobile_count}")
        self.broadcast({
            "type": "presence",
            "pc_clients": pc_count,
            "mobile_clients": mobile_count,
            "active_pet_id": self.active_pet_id,
        })
        return q

    def unregister_client(self, client_id: str) -> None:
        with self._lock:
            self._clients.pop(client_id, None)
            pc_count = sum(1 for c in self._clients.values() if c["role"] == "pc")
            mobile_clients = sum(1 for c in self._clients.values() if c["role"] == "mobile")

        logger.info(f"Client disconnected: {client_id} | PC={pc_count}, Mobile={mobile_clients}")
        self.broadcast({
            "type": "presence",
            "pc_clients": pc_count,
            "mobile_clients": mobile_clients,
            "active_pet_id": self.active_pet_id,
        })

    def get_primary_pc_client_id(self) -> str | None:
        """返回最新活跃的电脑端 Web 主舞台 client_id，防止开多个标签页时重复触发语音"""
        with self._lock:
            pc_items = [
                (cid, info["connected_at"])
                for cid, info in self._clients.items()
                if info["role"] == "pc"
            ]
        if not pc_items:
            return None
        pc_items.sort(key=lambda x: x[1], reverse=True)
        return pc_items[0][0]

    def broadcast(self, event: dict[str, Any], target_role: str | None = None) -> None:
        with self._lock:
            targets = [
                info["queue"]
                for info in self._clients.values()
                if target_role is None or info["role"] == target_role
            ]
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    def append_history(self, role: str, text: str, pet_id: str = "luma", source: str = "pc") -> None:
        with self._lock:
            self.history_log.append({
                "role": role,
                "text": text,
                "pet_id": pet_id,
                "source": source,
                "time": time.strftime("%H:%M:%S"),
            })
            if len(self.history_log) > 30:
                self.history_log = self.history_log[-30:]


companion_mobile = CompanionMobileHub(port=7860)
