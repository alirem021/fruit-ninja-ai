"""ابزار برچسب‌زدن فریم‌ها برای YOLO.

    python label.py                                  # برچسب‌زدن دستی
    python label.py --model models/fruit_ninja_yolo11n.pt   # با پیش‌برچسب مدل

موس:  کشیدن (چپ) = جعبه‌ی جدید   |   کلیک راست روی جعبه = حذفش
کلید: 1 = کلاس میوه   2 = کلاس بمب   u = برگردوندن آخرین جعبه   d = پاک کردن همه‌ی جعبه‌ها
      Enter/Space = ذخیره و بعدی   b = قبلی   x = رد کردن (ذخیره نمی‌شه)   q/Esc = خروج

قانون برچسب‌زدن:
  * فقط میوه‌ی «سالم و کامل» (و بمب) رو جعبه بزن. نصفه‌ها، لکه‌ی آب میوه، رد چاقو و
    اعداد رو جعبه نزن. تا مدل یاد بگیره اون‌ها پس‌زمینه‌ان.
  * فریم بدون میوه هم مهمه: فقط Enter بزن (فایل برچسب خالی = نمونه‌ی منفی).
  * جعبه رو تنگ دور خود میوه بکش.
"""
import argparse
import random
from pathlib import Path

import cv2

CLASSES = ["fruit", "bomb"]
COLORS = [(0, 200, 0), (0, 0, 255)]  # BGR
RAW = Path("dataset/raw")
LABELS = Path("dataset/labels")
MIN_BOX = 10  # پیکسل


def label_path(img):
    return LABELS / f"{img.stem}.txt"


def read_boxes(path, w, h):
    boxes = []
    for line in path.read_text().splitlines():
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        boxes.append([int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return boxes


def write_boxes(path, boxes, w, h):
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    LABELS.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


class Labeler:
    def __init__(self, scale):
        self.scale = scale
        self.boxes, self.cls = [], 0
        self.drag_start = self.drag_now = None

    def on_mouse(self, event, x, y, flags, _):
        x, y = x / self.scale, y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = self.drag_now = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start:
            self.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start:
            x1, y1 = self.drag_start
            x1, x2 = sorted((x1, x))
            y1, y2 = sorted((y1, y))
            if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX:
                self.boxes.append([self.cls, x1, y1, x2, y2])
            self.drag_start = self.drag_now = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            hits = [b for b in self.boxes if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
            if hits:  # کوچک‌ترین جعبه‌ای که زیر نشانگره
                self.boxes.remove(min(hits, key=lambda b: (b[3] - b[1]) * (b[4] - b[2])))

    def render(self, img, hud):
        out = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        s = self.scale
        for c, x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(out, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)), COLORS[c], 2)
            cv2.putText(out, CLASSES[c], (int(x1 * s), int(y1 * s) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[c], 2)
        if self.drag_start:
            (x1, y1), (x2, y2) = self.drag_start, self.drag_now
            cv2.rectangle(out, (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s)),
                          COLORS[self.cls], 1)
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(out, hud, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[self.cls], 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="مدل YOLO برای پیش‌برچسب")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--scale", type=float, default=1.0, help="اندازه‌ی نمایش")
    args = ap.parse_args()

    imgs = sorted(RAW.glob("*/*.jpg"))
    if not imgs:
        raise SystemExit("اول record.py رو اجرا کن.")
    random.Random(0).shuffle(imgs)  # ترتیب ثابت و تصادفی تا تنوع زودتر دیده بشه
    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model)

    i = next((k for k, p in enumerate(imgs) if not label_path(p).exists()), 0)
    lab = Labeler(args.scale)
    win = "label"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lab.on_mouse)

    def load(idx):
        img = cv2.imread(str(imgs[idx]))
        h, w = img.shape[:2]
        lp = label_path(imgs[idx])
        if lp.exists():
            lab.boxes = read_boxes(lp, w, h)
        elif model:
            r = model.predict(img, conf=args.conf, imgsz=640, verbose=False)[0]
            lab.boxes = [[int(c), *map(float, b)] for c, b in
                         zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
        else:
            lab.boxes = []
        return img, w, h

    img, w, h = load(i)
    while True:
        done = sum(label_path(p).exists() for p in imgs)
        hud = (f"[{CLASSES[lab.cls]}] 1=fruit 2=bomb | frame {i + 1}/{len(imgs)} "
               f"| labeled {done} | boxes {len(lab.boxes)}")
        cv2.imshow(win, lab.render(img, hud))
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue
        if k in (ord("q"), 27):
            break
        elif k == ord("1"):
            lab.cls = 0
        elif k == ord("2"):
            lab.cls = 1
        elif k == ord("u") and lab.boxes:
            lab.boxes.pop()
        elif k == ord("d"):
            lab.boxes.clear()
        elif k in (13, 32):  # Enter / Space
            write_boxes(label_path(imgs[i]), lab.boxes, w, h)
            i = min(i + 1, len(imgs) - 1)
            img, w, h = load(i)
        elif k == ord("x"):
            i = min(i + 1, len(imgs) - 1)
            img, w, h = load(i)
        elif k == ord("b"):
            i = max(i - 1, 0)
            img, w, h = load(i)
    cv2.destroyAllWindows()
    print(f"{sum(label_path(p).exists() for p in imgs)} فریم برچسب خورده.")


if __name__ == "__main__":
    main()
