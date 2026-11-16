from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEB_URL = "http://127.0.0.1:7860"


def is_web_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{WEB_URL}/api/status", timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_services() -> dict:
    if not is_web_ready():
        print("🚀 正在启动 Qwen 3B × Breeze TTS 2 语音服务后台...")
        log_f = open(BASE_DIR / "webui.log", "ab")
        subprocess.Popen(
            [str(BASE_DIR / ".venv" / "bin" / "python"), str(BASE_DIR / "app.py")],
            cwd=str(BASE_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    print("⏳ 正在等待 Breeze TTS 2 语音模型载入 GPU 显存...", end="", flush=True)
    for _ in range(120):
        try:
            with urllib.request.urlopen(f"{WEB_URL}/api/status", timeout=2.0) as resp:
                data = json.loads(resp.read().decode())
                if data.get("breeze_ready"):
                    print(" 就绪！")
                    return data
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1.0)
    raise RuntimeError("Breeze TTS 2 服务启动超时，请检查 breeze_server.log")


def send_chat(
    message: str,
    history: list[dict[str, str]],
    gpu_mode: str = "concurrent",
    voice_preset: str = "female_warm",
) -> str:
    payload = {
        "message": message,
        "history": history,
        "gpu_mode": gpu_mode,
        "voice_preset": voice_preset,
        "local_speaker": True,
        "expressive_tags": True,
    }
    req = urllib.request.Request(
        f"{WEB_URL}/api/chat_stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    full_reply = ""
    print("\n🤖 助手: ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            evt = json.loads(line[6:])
            if evt.get("type") == "text":
                delta = evt.get("delta", "")
                full_reply += delta
                print(delta, end="", flush=True)
            elif evt.get("type") == "done":
                ft = evt.get("first_token_ms")
                fa = evt.get("first_audio_ms")
                ft_s = f"{int(ft)}ms" if ft else "-"
                fa_s = f"{int(fa)}ms" if fa else "-"
                print(f"\n   [🔊 语音已播报 | 首字延迟: {ft_s} | 首音延迟: {fa_s}]")
    return full_reply


def main() -> None:
    status = wait_for_services()
    print("\n" + "=" * 64)
    print("🎉 Qwen 2.5 (3B) × Breeze TTS 2 实时语音对话系统已启动！")
    print(
        f"💻 当前显卡: {status.get('name')} | 显存占用: {status.get('vram_used_mb')} / {status.get('vram_total_mb')} MB"
    )
    print(f"🌐 网页可视化界面 (浏览器可打开): {WEB_URL}")
    print("🔊 终端与网页均已开启【电脑扬声器实时发声】")
    print("💡 输入 /mode 可切换【常驻并发模式】与【显存分时轮换模式】，输入 /quit 退出")
    print("=" * 64)

    history: list[dict[str, str]] = []
    gpu_mode = "concurrent"

    # Initial spoken greeting
    greeting_q = "用一句话热情地跟主人打个招呼，告诉他你已经准备好用声音和他聊天了。"
    reply = send_chat(greeting_q, history, gpu_mode=gpu_mode)
    history.append({"role": "user", "content": greeting_q})
    history.append({"role": "assistant", "content": reply})

    while True:
        try:
            user_in = input("\n🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if not user_in:
            continue
        if user_in.lower() in ("/quit", "/exit", "exit", "quit"):
            print("👋 再见！")
            break
        if user_in.lower() == "/mode":
            gpu_mode = "timeshare" if gpu_mode == "concurrent" else "concurrent"
            mode_label = (
                "🚀 双模型常驻 GPU 流式并发"
                if gpu_mode == "concurrent"
                else "🔄 GPU 显存分时轮换 (千问跑完卸载再跑TTS)"
            )
            print(f"✅ 已切换 GPU 调度模式为: {mode_label}")
            continue

        reply = send_chat(user_in, history, gpu_mode=gpu_mode)
        if reply:
            history.append({"role": "user", "content": user_in})
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
