"""نسخه ۱: ساده‌ترین ایده، بدون هوش مصنوعی و فقط با OpenCV.

هر ثانیه یه اسکرین‌شات می‌گیره، با رنگ و حرکت میوه رو پیدا می‌کنه و می‌زنه.
(جواب نمی‌ده، ولی برای فهمیدن این‌که چرا نسخه ۲ لازم بود خوبه.)

    python v1_opencv.py              # بازی واقعی، ۱ فریم در ثانیه
    python v1_opencv.py --dry-run    # فقط تشخیص، ماوس تکون نمی‌خوره
    python v1_opencv.py --fps 10 --show

برای قطع کردن ماوس رو ببر گوشه‌ی بالا-چپ صفحه (یا Ctrl+C).
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from mouse import position, swipe
from screen import Screen, load_region

# بازه‌ی رنگ میوه‌ها در HSV (توی OpenCV: H از ۰ تا ۱۸۰، S و V از ۰ تا ۲۵۵)
# اعداد رو حدسی گذاشتم، با بازی واقعی کامل جور نیستن.
FRUIT_COLORS = {
    "red":    [((0, 130, 90), (8, 255, 255)), ((170, 130, 90), (180, 255, 255))],
    "orange": [((9, 160, 140), (20, 255, 255))],
    "yellow": [((21, 120, 140), (34, 255, 255))],
    "green":  [((35, 90, 70), (85, 255, 255))],
    "purple": [((125, 70, 50), (165, 255, 255))],
}
BOMB_MAX_V = 60          # بمب = تیره (روشنایی کم)
MOTION_THRESH = 25       # حداقل تغییر پیکسل بین دو فریم تا «حرکت» حساب بشه
MIN_AREA = 0.0015        # حداقل مساحت میوه، نسبت به کل تصویر
BOMB_SAFE_DIST = 0.12    # اگه بمب نزدیک‌تر از این (نسبت به عرض تصویر) بود، نزن
FAILSAFE = 5             # ماوس توی این فاصله از گوشه‌ی (0,0) = توقف

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def clean(mask):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def blobs(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        found.append({"center": (x + w // 2, y + h // 2), "box": (x, y, w, h), "area": area})
    return found


def detect(frame, prev):
    h, w = frame.shape[:2]
    min_area = MIN_AREA * w * h
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ۱) ماسک رنگ: هر پیکسلی که رنگش شبیه یکی از میوه‌هاست
    color = np.zeros((h, w), np.uint8)
    for ranges in FRUIT_COLORS.values():
        for lo, hi in ranges:
            color |= cv2.inRange(hsv, lo, hi)

    # ۲) ماسک حرکت: هر چیزی که نسبت به فریم قبل عوض شده
    if prev is None:
        motion = np.full((h, w), 255, np.uint8)
    else:
        # روی هر سه کانال رنگ، نه خاکستری: قرمز و قهوه‌ای روشنایی یکسانی دارن
        diff = cv2.absdiff(frame, prev).max(axis=2)
        motion = cv2.threshold(diff, MOTION_THRESH, 255, cv2.THRESH_BINARY)[1]
        motion = cv2.dilate(motion, KERNEL, iterations=2)

    # ۳) میوه = رنگ میوه داره و حرکت کرده
    fruit_mask = clean(color & motion)
    # بمب = تیره و حرکت کرده
    bomb_mask = clean(cv2.inRange(hsv[:, :, 2], 0, BOMB_MAX_V) & motion)

    return blobs(fruit_mask, min_area), blobs(bomb_mask, min_area), fruit_mask, bomb_mask


def near_bomb(pt, bombs, max_dist):
    return any(np.hypot(pt[0] - b["center"][0], pt[1] - b["center"][1]) < max_dist
               for b in bombs)


def draw(frame, fruits, bombs, fruit_mask, bomb_mask, cut, info):
    out = frame.copy()
    for f in fruits:
        x, y, w, h = f["box"]
        color = (0, 255, 0) if f in cut else (0, 200, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    for b in bombs:
        x, y, w, h = b["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(out, "BOMB", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(out, info, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    # کنار تصویر اصلی، ماسک‌ها رو هم نشون بده (سبز = میوه، قرمز = بمب)
    masks = np.zeros_like(frame)
    masks[:, :, 1] = fruit_mask
    masks[:, :, 2] = bomb_mask
    return np.hstack([out, masks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="ماوس رو تکون نده")
    ap.add_argument("--show", action="store_true", help="پنجره‌ی دیباگ زنده")
    ap.add_argument("--no-save", action="store_true", help="فریم‌ها رو ذخیره نکن")
    ap.add_argument("--seconds", type=float, default=120, help="حداکثر زمان اجرا")
    args = ap.parse_args()

    screen = Screen(load_region())
    out_dir = Path("runs/v1") / datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("۳ ثانیه وقت داری روی بازی کلیک کنی...")
    time.sleep(3)

    prev, n, cuts, start = None, 0, 0, time.time()
    try:
        while time.time() - start < args.seconds:
            t0 = time.time()
            mx, my = position()
            if mx < FAILSAFE and my < FAILSAFE:
                print("failsafe: ماوس گوشه‌ی صفحه‌ست، توقف.")
                break

            frame = screen.grab()
            fruits, bombs, fruit_mask, bomb_mask = detect(frame, prev)
            prev = frame

            w = frame.shape[1]
            safe = BOMB_SAFE_DIST * w
            cut = [f for f in fruits if not near_bomb(f["center"], bombs, safe)]
            for f in cut:
                # یک خط افقی از وسط میوه
                cx, cy = f["center"]
                half = max(f["box"][2], 0.08 * w)
                x1, y1 = screen.to_screen(cx - half, cy)
                x2, y2 = screen.to_screen(cx + half, cy)
                if not args.dry_run:
                    swipe(x1, y1, x2, y2)
            cuts += len(cut)

            took = (time.time() - t0) * 1000
            info = f"#{n} fruits={len(fruits)} bombs={len(bombs)} cut={len(cut)} {took:.0f}ms"
            print(info)
            if not args.no_save or args.show:
                vis = draw(frame, fruits, bombs, fruit_mask, bomb_mask, cut, info)
                if not args.no_save:
                    cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), vis)
                if args.show:
                    cv2.imshow("v1", cv2.resize(vis, None, fx=0.4, fy=0.4))
                    if cv2.waitKey(1) == 27:
                        break
            n += 1
            time.sleep(max(0, 1 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    print(f"تمام: {n} فریم، {cuts} برش. خروجی: {out_dir if not args.no_save else '-'}")


if __name__ == "__main__":
    main()
