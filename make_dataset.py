"""تبدیل فریم‌های برچسب‌خورده به دیتاست YOLO (train/val).

فریم‌های پشت‌سرهم تقریباً شبیه همن. اگه تصادفی تقسیم کنیم، فریم val شبیه فریم train می‌شه
و عدد mAP الکی خوب درمیاد. برای همین «بلوک‌های ۲۵تایی» رو کامل به val یا train می‌دیم.
"""
import random
import shutil
from pathlib import Path

from label import LABELS, RAW, CLASSES

OUT = Path("dataset/yolo")
BLOCK = 25
VAL_EVERY = 6  # تقریباً ۱۶٪ بلوک‌ها برای val


def main():
    imgs = sorted(p for p in RAW.glob("*/*.jpg") if (LABELS / f"{p.stem}.txt").exists())
    if not imgs:
        raise SystemExit("هنوز فریمی برچسب نخورده.")
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)

    blocks = [imgs[i:i + BLOCK] for i in range(0, len(imgs), BLOCK)]
    val_ids = set(random.Random(0).sample(range(len(blocks)), max(1, len(blocks) // VAL_EVERY)))
    counts = {"train": [0, 0, 0], "val": [0, 0, 0]}  # فریم، میوه، بمب
    for bi, block in enumerate(blocks):
        split = "val" if bi in val_ids else "train"
        for p in block:
            shutil.copy(p, OUT / "images" / split / p.name)
            lp = LABELS / f"{p.stem}.txt"
            shutil.copy(lp, OUT / "labels" / split / lp.name)
            counts[split][0] += 1
            for line in lp.read_text().splitlines():
                counts[split][1 + int(line.split()[0])] += 1

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    (OUT / "data.yaml").write_text(f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    for s, (f, a, b) in counts.items():
        print(f"{s}: {f} فریم، {a} میوه، {b} بمب")
    print("data.yaml:", OUT / "data.yaml")


if __name__ == "__main__":
    main()
