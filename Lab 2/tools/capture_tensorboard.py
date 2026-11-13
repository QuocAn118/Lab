"""Chụp TensorBoard thật từ log các run, bằng Edge headless và Playwright.

Yêu cầu tùy chọn: pip install playwright; Microsoft Edge trên Windows.
Không dựng lại biểu đồ để thay ảnh chụp giao diện.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    from playwright.sync_api import sync_playwright
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    folder = ROOT / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    with (ROOT / "reports/tv3/tensorboard_server.log").open("w", encoding="utf-8") as output:
        server = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir",
                                   str(ROOT / "reports/runs"), "--host", "127.0.0.1",
                                   "--port", str(port)], stdout=output, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError("TensorBoard stopped; see tensorboard_server.log")
                try:
                    urllib.request.urlopen(url, timeout=1).close()
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("TensorBoard did not start")
            with sync_playwright() as browser_api:
                browser = browser_api.chromium.launch(channel="msedge", headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1600, "height": 1200}, device_scale_factor=1)
                    page.goto(url + "/#scalars", wait_until="networkidle")
                    page.wait_for_timeout(12000)
                    page.get_by_text("Loss", exact=True).click()
                    page.wait_for_timeout(2000)
                    page.screenshot(path=str(folder / "tensorboard.png"), full_page=True)
                finally:
                    browser.close()
        finally:
            server.terminate()
            server.wait(timeout=15)
    print(folder / "tensorboard.png")


if __name__ == "__main__":
    main()
