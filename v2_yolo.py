"""نسخه ۲: YOLO میوه و بمب رو پیدا می‌کنه، بعد مسیرشون رو پیش‌بینی می‌کنیم و می‌زنیم.

    python v2_yolo.py --dry-run --record    # فقط نگاه می‌کنه و فیلم دیباگ می‌گیره، ماوس تکون نمی‌خوره
    python v2_yolo.py --record              # بازی واقعی

برای قطع کردن ماوس رو ببر گوشه‌ی بالا-چپ صفحه (یا Ctrl+C).
اگه Chrome روی ناحیه‌ی بازی نباشه ضربه نمی‌زنه، که به برنامه‌های دیگه چاقو نزنه.
"""
import argparse
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                    kCGWindowListOptionOnScreenOnly)
from ultralytics import YOLO

from mouse import position, swipe
from planner import plan
from screen import Screen, load_region
from tracker import Tracker

MODEL = "models/fruit_ninja_yolo11n.pt"
FAILSAFE = 5
SWIPE_STEPS, SWIPE_TIME = 8, 0.05
# میوه‌ی بریده‌شده به دو نصفه تبدیل می‌شه و ردش قطع می‌شه. پس اگه بعد از این مدت هنوز
# همون ردیابی زنده‌ست، یعنی ضربه خطا رفته و دوباره می‌زنیمش.
COOLDOWN = 0.35
FRUIT_CONF, BOMB_CONF = 0.30, 0.25  # بمب حساس‌تر: از دست دادنش گرون‌تره از آلارم اشتباه
HEARTBEAT = 5.0


def top_window_owner(x, y):
    """اسم برنامه‌ای که بالاترین پنجره‌ی معمولی در نقطه‌ی (x, y) صفحه‌ست."""
    for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID):
        if w.get("kCGWindowLayer") != 0:
            continue
        b = w["kCGWindowBounds"]
        if b["X"] <= x <= b["X"] + b["Width"] and b["Y"] <= y <= b["Y"] + b["Height"]:
            return w.get("kCGWindowOwnerName")
    return None


class Grabber(threading.Thread):
    """thread جدا که مدام آخرین فریم رو (با زمان گرفتنش) نگه می‌داره."""

    def __init__(self, region):
        super().__init__(daemon=True)
        self.region, self.latest, self.stop = region, None, False
        self.lock = threading.Lock()

    def run(self):
        screen = Screen(self.region)  # mss باید توی همین thread ساخته بشه
        self.screen = screen
        while not self.stop:
            frame = screen.grab()
            with self.lock:
                self.latest = (time.perf_counter(), frame)

    def get(self, after):
        with self.lock:
            if self.latest and self.latest[0] > after:
                return self.latest
        return None


class Swiper(threading.Thread):
    """ضربه‌ها توی thread جدا اجرا می‌شن تا حلقه‌ی دیدن متوقف نشه."""

    def __init__(self, to_screen):
        super().__init__(daemon=True)
        self.q, self.to_screen, self.count = queue.Queue(maxsize=2), to_screen, 0

    def run(self):
        while True:
            a, b = self.q.get()
            try:
                x1, y1 = self.to_screen(*a)
                x2, y2 = self.to_screen(*b)
                swipe(x1, y1, x2, y2, steps=SWIPE_STEPS, duration=SWIPE_TIME)
                self.count += 1
            except Exception as e:  # اگه این thread بی‌صدا بمیره دیگه هیچ ضربه‌ای نمی‌خوره
                print("خطا در ضربه:", repr(e))


def draw(frame, tracker, targets, bombs_pred, action, info):
    out = frame.copy()
    for tr in tracker.tracks:
        x, y = tr.pos
        col = (0, 0, 255) if tr.cls == 1 else (0, 220, 0)
        cv2.circle(out, (int(x), int(y)), int(tr.radius), col, 2)
        cv2.putText(out, f"#{tr.id}", (int(x - tr.radius), int(y - tr.radius - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for t in targets:
        cv2.drawMarker(out, (int(t["x"]), int(t["y"])), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    for bx, by, br in bombs_pred:
        cv2.circle(out, (int(bx), int(by)), int(br), (0, 0, 255), 1)
    if action:
        (x1, y1), (x2, y2), _ = action
        cv2.arrowedLine(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 4)
    cv2.rectangle(out, (0, out.shape[0] - 32), (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, info, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--fruit-conf", type=float, default=FRUIT_CONF)
    ap.add_argument("--bomb-conf", type=float, default=BOMB_CONF)
    ap.add_argument("--seconds", type=float, default=0, help="0 = تا وقتی خودت قطع کنی")
    ap.add_argument("--lead", type=float, default=0.0, help="ثانیه‌ی اضافه برای جبران تأخیر (می‌تونه منفی باشه)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--record", action="store_true", help="فیلم دیباگ ذخیره کن")
    ap.add_argument("--no-chrome-check", action="store_true")
    ap.add_argument("--delay", type=float, default=8, help="ثانیه‌ی انتظار تا بری سراغ بازی")
    args = ap.parse_args()

    region = load_region()
    W, H = region["width"], region["height"]
    model = YOLO(args.model)
    model.predict(np.zeros((H, W, 3), np.uint8), imgsz=640, device="mps", verbose=False)  # گرم‌کردن

    grabber = Grabber(region)
    grabber.start()
    while grabber.latest is None:
        time.sleep(0.01)
    swiper = Swiper(grabber.screen.to_screen)
    swiper.start()
    tracker = Tracker(W, H)

    out_dir = Path("runs/v2") / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = None
    if args.record:
        out_dir.mkdir(parents=True, exist_ok=True)
        # هر فریمِ دوم ضبط می‌شه؛ با ~۶۰ فریم در ثانیه پردازش، فیلم ۳۰ فریمی تقریباً هم‌زمان با واقعیته
        writer = cv2.VideoWriter(str(out_dir / "debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

    print(f"{args.delay:.0f} ثانیه وقت داری: برو روی Chrome، بازی (Classic) رو شروع کن...")
    time.sleep(args.delay)

    last_t, n, start, last_check, chrome_ok = 0.0, 0, time.time(), 0.0, True
    lat_ms, swipes_planned, last_beat, fruits_seen = [], 0, time.time(), set()
    try:
        while args.seconds <= 0 or time.time() - start < args.seconds:
            mx, my = position()
            if mx < FAILSAFE and my < FAILSAFE:
                print("failsafe: توقف.")
                break
            got = grabber.get(last_t)
            if got is None:
                time.sleep(0.001)
                continue
            t_cap, frame = got
            last_t = t_cap

            now = time.time()
            if not args.no_chrome_check and now - last_check > 0.25:
                cx = region["left"] + W / 2
                cy = region["top"] + H / 2
                chrome_ok = top_window_owner(cx, cy) == "Google Chrome"
                last_check = now

            res = model.predict(frame, conf=min(args.fruit_conf, args.bomb_conf),
                                imgsz=640, device="mps", verbose=False)[0]
            dets = []
            for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                                              res.boxes.conf.tolist()):
                if p >= (args.bomb_conf if int(c) == 1 else args.fruit_conf):
                    dets.append((int(c), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            tracker.update(dets, t_cap)
            fruits_seen.update(tr.id for tr in tracker.tracks if tr.cls == 0)

            # چاقو چند میلی‌ثانیه دیرتر می‌رسه، پس جای میوه رو برای اون لحظه حساب می‌کنیم:
            # از گرفتن عکس تا الان + نصف مدت ضربه + --lead
            t_now = time.perf_counter()
            lead = (t_now - t_cap) + SWIPE_TIME / 2 + args.lead
            c_prior, c_range = tracker.c_prior, tracker.c_range
            targets, bombs_pred = [], []
            for tr in tracker.tracks:
                ahead = (t_cap - tr.last_seen) + lead
                p = tr.predict(ahead, c_prior, c_range)
                if tr.cls == 1:  # بمب: هم موقعیت فعلی، هم پیش‌بینی‌شده، با حاشیه‌ی ایمنی
                    bombs_pred.append((*tr.pos, tr.radius))
                    if p:
                        bombs_pred.append((*p, tr.radius))
                elif p and t_cap > tr.cooldown_until and -tr.radius < p[0] < W + tr.radius and 0 < p[1] < H:
                    targets.append(dict(id=tr.id, x=p[0], y=p[1], r=tr.radius))

            action = None
            if targets and chrome_ok:
                action = plan(targets, bombs_pred, W, H)
                if action:
                    swipes_planned += 1
                    if not args.dry_run:
                        try:
                            swiper.q.put_nowait((action[0], action[1]))
                        except queue.Full:
                            action = None
                    if action:
                        for tr in tracker.tracks:
                            if tr.id in action[2]:
                                tr.cooldown_until = t_cap + COOLDOWN

            lat_ms.append((time.perf_counter() - t_cap) * 1000)
            n += 1
            if time.time() - last_beat >= HEARTBEAT:
                last_beat = time.time()
                print(f"[{time.time() - start:4.0f}s] {n / (time.time() - start):.0f} fps | "
                      f"میوه‌ی دیده‌شده {len(fruits_seen)} | ضربه {swiper.count} | "
                      f"{'OK' if chrome_ok else 'Chrome جلو نیست، ضربه متوقف'}")
            if writer and n % 2 == 0:
                info = (f"{'DRY ' if args.dry_run else ''}fruits={sum(t.cls == 0 for t in tracker.tracks)} "
                        f"bombs={sum(t.cls == 1 for t in tracker.tracks)} swipes={swipes_planned} "
                        f"lat={lat_ms[-1]:.0f}ms c={c_prior:.0f} {'' if chrome_ok else 'CHROME NOT ON TOP'}")
                writer.write(draw(frame, tracker, targets, bombs_pred, action, info))
    except KeyboardInterrupt:
        pass
    finally:
        grabber.stop = True
        if writer:
            writer.release()
    el = time.time() - start
    print(f"تمام: {n} فریم در {el:.0f}s ({n / el:.0f} fps)، تأخیر میانه {np.median(lat_ms):.0f}ms، "
          f"{swipes_planned} ضربه برنامه‌ریزی شد، {swiper.count} اجرا شد، c={tracker.c_prior:.0f}")
    if writer:
        print("فیلم دیباگ:", out_dir / "debug.mp4")


if __name__ == "__main__":
    main()
