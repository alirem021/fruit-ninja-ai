"""آموزش YOLO روی دیتاست.

    python train.py                  # آموزش کامل
    python train.py --epochs 30 --name quick   # آموزش سریع برای پیش‌برچسب
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--name", default="v2")
    ap.add_argument("--base", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data="dataset/yolo/data.yaml",
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device="mps",
        workers=4,
        project="runs/train",
        name=args.name,
        exist_ok=True,
        patience=20,
        flipud=0.0,     # بازی همیشه سر و ته نیست
        fliplr=0.5,
        degrees=0.0,
        hsv_h=0.01,     # رنگ میوه مهمه، زیاد عوضش نکن
    )
    print(f"مدل: runs/train/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
