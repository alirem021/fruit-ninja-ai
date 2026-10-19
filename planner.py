"""برنامه‌ریز ضربه: از کجا بکشیم که بیشترین میوه رو ببریم و به بمب نخوریم."""
import math

ANGLES = [0, 25, -25, 50, -50, 90]  # درجه نسبت به افق


def dist_point_segment(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0 if L2 == 0 else max(0, min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def plan(fruits, bombs, W, H):
    """fruits: [{id,x,y,r}] موقعیت پیش‌بینی‌شده؛ bombs: [(x,y,r), ...] (موقعیت فعلی و پیش‌بینی‌شده).
    خروجی: (a, b, ids) یا None. a و b نقطه‌ی شروع و پایان ضربه (پیکسل تصویر)."""
    best, best_score = None, -1e9
    margin = 0.03 * W
    for seed in sorted(fruits, key=lambda f: -f["y"])[:3]:  # اول اونی که زودتر از صفحه خارج می‌شه
        for ang in ANGLES:
            rad = math.radians(ang)
            dx, dy = math.cos(rad), math.sin(rad)
            members = []
            for f in fruits:
                rx, ry = f["x"] - seed["x"], f["y"] - seed["y"]
                proj = rx * dx + ry * dy
                perp = abs(rx * dy - ry * dx)
                if perp <= 0.8 * f["r"] and abs(proj) <= 0.4 * W:
                    members.append((proj, f))
            projs = [p for p, _ in members]
            pad = max(0.9 * seed["r"], 0.04 * W)
            lo, hi = min(projs) - pad, max(projs) + pad
            need = max(2.6 * seed["r"], 0.12 * W)  # حداقل طول ضربه
            if hi - lo < need:
                mid = (hi + lo) / 2
                lo, hi = mid - need / 2, mid + need / 2
            a = (seed["x"] + dx * lo, seed["y"] + dy * lo)
            b = (seed["x"] + dx * hi, seed["y"] + dy * hi)
            if any(dist_point_segment((bx, by), a, b) < br + margin for bx, by, br in bombs):
                continue
            score = 10 * len(members) - abs(ang) * 0.02
            if score > best_score:
                best_score = score
                best = (a, b, [f["id"] for _, f in members])
    return best
