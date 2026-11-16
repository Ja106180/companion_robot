#!/usr/bin/env python3
"""
download_models.py - 伴生萌宠超大模型权重高速下载器
=====================================================
自动从国内高速镜像（ModelScope 魔搭社区 / 备用 HuggingFace Mirror）
多线程并发下载被 GitHub 100MB 限制的超大权重：
1. Breeze-TTS-2 Q4_K GGUF 语音合成模型 (2.53 GB)
2. SenseVoice-Small INT8 ONNX 离线语音识别模型 (239 MB)
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODELS_CONFIG = [
    {
        "name": "Breeze-TTS-2 Q4_K 语音合成模型",
        "path": BASE_DIR / "models" / "breeze-tts-2-q4_k.gguf",
        "expected_min_bytes": 2_500_000_000,
        "urls": [
            "https://modelscope.cn/models/tardigrade4ai/Breeze-TTS-2.cpp/resolve/master/breeze-tts-2-q4_k.gguf",
            "https://hf-mirror.com/HoppouAI/Breeze-TTS-2.cpp/resolve/main/breeze-tts-2-q4_k.gguf",
        ],
    },
    {
        "name": "SenseVoice-Small INT8 离线识别模型",
        "path": BASE_DIR / "data" / "sensevoice_int8" / "model.int8.onnx",
        "expected_min_bytes": 230_000_000,
        "urls": [
            "https://www.modelscope.cn/api/v1/models/pengzhendong/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/repo?Revision=master&FilePath=model.int8.onnx",
            "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx",
        ],
    },
]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
NUM_THREADS = 8


def get_file_info(url: str) -> tuple[int, str]:
    """探测文件大小及最终重定向直连 CDN"""
    req = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        final_url = resp.geturl()
        cr = resp.headers.get("Content-Range", "")
        if cr and "/" in cr:
            total = int(cr.split("/")[-1])
        else:
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl else 0
        return total, final_url


def download_file_multithread(url: str, out_path: Path, name: str) -> bool:
    """多线程分块下载大文件"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".downloading")

    print(f"\n[开始下载] {name}")
    print(f"目标路径: {out_path}")
    print("正在连接 CDN 探测资源大小...")

    try:
        total_size, final_url = get_file_info(url)
    except Exception as exc:
        print(f"获取链接信息失败: {exc}")
        return False

    print(f"文件大小: {total_size / (1024**2):.2f} MB | 已锁定高速 CDN")

    with open(temp_path, "wb") as f:
        f.truncate(total_size)

    ranges = []
    for start in range(0, total_size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        ranges.append((start, end))

    fd = os.open(str(temp_path), os.O_RDWR)
    downloaded = 0
    start_time = time.time()
    last_print = 0

    def fetch_chunk(r: tuple[int, int]) -> int:
        s, e = r
        expected = e - s + 1
        for attempt in range(5):
            try:
                target = final_url if attempt < 2 else url
                req = urllib.request.Request(
                    target,
                    headers={
                        "Range": f"bytes={s}-{e}",
                        "User-Agent": "Mozilla/5.0",
                    },
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status}")
                    pos = s
                    while pos <= e:
                        buf = resp.read(256 * 1024)
                        if not buf:
                            break
                        os.pwrite(fd, buf, pos)
                        pos += len(buf)
                    if pos - s == expected:
                        return expected
            except Exception:
                time.sleep(1 + attempt)
        raise RuntimeError(f"分块 {s}-{e} 重试多次失败")

    try:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(fetch_chunk, r) for r in ranges]
            for fut in as_completed(futures):
                nbytes = fut.result()
                downloaded += nbytes
                now = time.time()
                if now - last_print > 0.4 or downloaded == total_size:
                    last_print = now
                    elapsed = max(now - start_time, 0.1)
                    speed = (downloaded / (1024**2)) / elapsed
                    pct = downloaded * 100.0 / total_size
                    sys.stdout.write(
                        f"\r进度: {pct:5.1f}% ({downloaded / (1024**2):6.1f} / {total_size / (1024**2):.1f} MB) | 速度: {speed:5.2f} MB/s"
                    )
                    sys.stdout.flush()
    finally:
        os.close(fd)

    if temp_path.stat().st_size == total_size:
        temp_path.replace(out_path)
        print(f"\n[完成] {name} 校验通过！")
        return True
    else:
        print(f"\n[错误] 文件大小不匹配，下载未完成")
        return False


def main() -> None:
    print("=" * 65)
    print(" 全息桌面伴生萌宠 · 超大模型权重自动校验与极速下载器")
    print("=" * 65)

    all_ok = True
    for item in MODELS_CONFIG:
        path: Path = item["path"]
        name: str = item["name"]
        min_bytes: int = item["expected_min_bytes"]

        if path.is_file() and path.stat().st_size >= min_bytes:
            size_mb = path.stat().st_size / (1024**2)
            print(f"[已就绪] {name} (当前大小: {size_mb:.1f} MB)，无需重复下载。")
            continue

        success = False
        for url in item["urls"]:
            print(f"\n尝试源: {url}")
            if download_file_multithread(url, path, name):
                success = True
                break
            else:
                print("当前下载源重试失败，正在切换备用源...")

        if not success:
            all_ok = False
            print(f"[失败] 无法下载 {name}，请检查网络连接。")

    print("\n" + "=" * 65)
    if all_ok:
        print("所有必需的大模型权重均已就绪！可以直接运行 python3 app.py 启动系统。")
    else:
        print("部分权重未下载完成，请排查网络后重新运行此脚本。")
    print("=" * 65)


if __name__ == "__main__":
    main()
