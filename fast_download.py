import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = sys.argv[1]
OUT_PATH = sys.argv[2]
NUM_THREADS = 8
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB per chunk


def get_size_and_final_url(url: str) -> tuple[int, str]:
    req = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        final_url = resp.geturl()
        cr = resp.headers.get("Content-Range", "")
        total = int(cr.split("/")[-1])
        return total, final_url


def main() -> None:
    total_size, final_url = get_size_and_final_url(URL)
    print(f"Total size: {total_size / (1024**2):.2f} MB | Direct CDN resolved", flush=True)

    with open(OUT_PATH, "wb") as f:
        f.truncate(total_size)

    ranges = []
    for start in range(0, total_size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, total_size - 1)
        ranges.append((start, end))

    downloaded = 0
    start_time = time.time()
    fd = os.open(OUT_PATH, os.O_RDWR)

    def fetch_range(r: tuple[int, int]) -> int:
        s, e = r
        expected = e - s + 1
        for attempt in range(6):
            try:
                target = final_url if attempt < 3 else URL
                req = urllib.request.Request(
                    target,
                    headers={"Range": f"bytes={s}-{e}", "User-Agent": "Mozilla/5.0"},
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
            except Exception as exc:
                time.sleep(1 + attempt)
                if attempt == 5:
                    raise RuntimeError(f"Failed range {s}-{e}: {exc}") from exc
        raise RuntimeError(f"Incomplete range {s}-{e}")

    try:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
            futures = [ex.submit(fetch_range, r) for r in ranges]
            for fut in as_completed(futures):
                nbytes = fut.result()
                downloaded += nbytes
                elapsed = max(time.time() - start_time, 0.1)
                speed = (downloaded / (1024**2)) / elapsed
                pct = downloaded * 100.0 / total_size
                print(
                    f"Progress: {pct:5.1f}% ({downloaded / (1024**2):6.1f} / {total_size / (1024**2):.1f} MB) | Speed: {speed:.2f} MB/s",
                    flush=True,
                )
    finally:
        os.close(fd)

    print("Download complete!", flush=True)


if __name__ == "__main__":
    main()
