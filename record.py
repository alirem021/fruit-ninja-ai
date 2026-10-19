"""ضبط فریم از بازی برای ساختن دیتاست.

اجرا (بعد از calibrate.py):
    python record.py --name me        # خودت بازی کن
    python record.py --name v1bot     # همزمان v1_opencv.py رو توی ترمینال دیگه اجرا کن

فریم‌ها توی dataset/raw/<name>/ ذخیره می‌شن. فریم‌هایی که تقریباً مثل قبلی‌ان
(منو، صفحه‌ی ثابت) ذخیره نمی‌شن.
توقف: Ctrl+C یا رسیدن به --seconds.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from screen import Screen, load_region

MIN_CHANGE = 4.0   # حداقل تفاوت میانگین پیکسل‌ها با آخرین فریم ذخیره‌شده


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="session")
    ap.add_argument("--fps", type=float, default=6, help="چند فریم در ثانیه ذخیره بشه")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    out = Path("dataset/raw") / args.name
    out.mkdir(parents=True, exist_ok=True)
    start_idx = len(list(out.glob("*.jpg")))

    screen = Screen(load_region())
    print(f"۵ ثانیه دیگه شروع می‌شه. برو سراغ بازی ({args.seconds:.0f} ثانیه ضبط می‌شه)...")
    time.sleep(5)

    last_small, saved, t_end = None, 0, time.time() + args.seconds
    try:
        while time.time() < t_end:
            t0 = time.time()
            frame = screen.grab()
            small = cv2.resize(frame, (160, 96)).astype(np.int16)
            if last_small is None or np.abs(small - last_small).mean() > MIN_CHANGE:
                name = f"{args.name}_{start_idx + saved:05d}.jpg"
                cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                last_small, saved = small, saved + 1
                if saved % 20 == 0:
                    print(f"{saved} فریم ذخیره شد")
            time.sleep(max(0, 1 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    print(f"تمام: {saved} فریم توی {out}")


if __name__ == "__main__":
    main()
