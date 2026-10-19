"""مشخص کردن ناحیه‌ی بازی روی صفحه.

ماوس رو می‌بری گوشه‌ی بالا-چپ بازی، Enter؛ بعد گوشه‌ی پایین-راست، Enter.
"""
import json

import cv2

from mouse import position
from screen import CONFIG, Screen

input("ماوس رو ببر روی گوشه‌ی «بالا-چپ» بازی و Enter بزن...")
x1, y1 = position()
input("حالا گوشه‌ی «پایین-راست» بازی و Enter...")
x2, y2 = position()

region = {"left": int(min(x1, x2)), "top": int(min(y1, y2)),
          "width": int(abs(x2 - x1)), "height": int(abs(y2 - y1))}
CONFIG.write_text(json.dumps(region, indent=2))
print("ذخیره شد:", region)

screen = Screen(region)
img = screen.grab()
cv2.imwrite("calibration.png", img)
print(f"نمونه تصویر: calibration.png  ({img.shape[1]}x{img.shape[0]} پیکسل، scale={screen.scale})")
