"""گرفتن تصویر از ناحیه‌ی بازی."""
import json
from pathlib import Path

import mss
import numpy as np

CONFIG = Path(__file__).parent / "config.json"


def load_region():
    if not CONFIG.exists():
        raise SystemExit("اول calibrate.py رو اجرا کن تا ناحیه‌ی بازی مشخص بشه.")
    return json.loads(CONFIG.read_text())


class Screen:
    def __init__(self, region):
        # region به واحد «پوینت» مک است (همون مختصات ماوس)
        self.region = region
        self.sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
        self.scale = None  # نسبت پیکسل تصویر به پوینت (روی Retina = 2)

    def grab(self):
        img = np.array(self.sct.grab(self.region))[:, :, :3]  # BGRA -> BGR
        if self.scale is None:
            self.scale = img.shape[1] / self.region["width"]
        return np.ascontiguousarray(img)

    def to_screen(self, x, y):
        """مختصات پیکسلی تصویر -> مختصات ماوس روی صفحه."""
        return (self.region["left"] + x / self.scale,
                self.region["top"] + y / self.scale)
