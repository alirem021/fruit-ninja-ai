"""ردیابی میوه‌ها و پیش‌بینی مسیر پرتابی.

هر میوه‌ای که پرتاب می‌شه با فیزیک ساده حرکت می‌کنه:
    x(t) = x0 + vx*t              (سرعت افقی ثابت)
    y(t) = y0 + vy*t + c*t^2      (c = نصف شتاب گرانش، به پیکسل بر ثانیه‌به‌توان۲، y رو به پایینه)
"""
from collections import deque

import numpy as np


class Track:
    _next_id = 0

    def __init__(self, cls, t, x, y, w, h):
        self.id = Track._next_id
        Track._next_id += 1
        self.cls = cls
        self.hist = deque(maxlen=12)  # (t, x, y)
        self.size = (w, h)
        self.last_seen = t
        self.cooldown_until = 0.0  # بعد از ضربه تا این زمان دوباره هدف نمی‌گیریمش
        self.add(t, x, y, w, h)

    def add(self, t, x, y, w, h):
        self.hist.append((t, x, y))
        self.size = (w, h)
        self.last_seen = t

    @property
    def radius(self):
        return max(self.size) / 2

    @property
    def pos(self):
        return self.hist[-1][1], self.hist[-1][2]

    def fit(self, c_prior, c_range):
        """برازش مسیر. برمی‌گردونه (px, py, c_learned) که چندجمله‌ای‌ها روی tau = t - t_last‌ان.
        c_learned فقط وقتی داده‌ی کافی بود پر می‌شه (برای یاد گرفتن گرانش بازی)."""
        n = len(self.hist)
        if n < 2:
            return None
        t, x, y = np.array(self.hist).T
        tau = t - t[-1]
        if tau[0] > -0.04:  # بازه‌ی زمانی خیلی کوتاه، سرعت قابل‌اعتماد نیست
            return None
        px = np.polyfit(tau, x, 1)
        c_learned = None
        if n >= 6:
            py = np.polyfit(tau, y, 2)
            if c_range[0] <= py[0] <= c_range[1]:
                c_learned = py[0]
            else:  # برازش نامعقول (نویز)، با گرانش پیش‌فرض دوباره بزن
                py = np.concatenate([[c_prior], np.polyfit(tau, y - c_prior * tau ** 2, 1)])
        else:
            py = np.concatenate([[c_prior], np.polyfit(tau, y - c_prior * tau ** 2, 1)])
        return px, py, c_learned

    def predict(self, ahead, c_prior, c_range):
        """موقعیت میوه «ahead» ثانیه بعد از آخرین دیده شدنش."""
        f = self.fit(c_prior, c_range)
        if f is None:
            return None
        px, py, _ = f
        return float(np.polyval(px, ahead)), float(np.polyval(py, ahead))


class Tracker:
    def __init__(self, width, height, max_gap=0.25):
        self.W, self.H = width, height
        self.tracks = []
        self.max_gap = max_gap
        self.c_range = (0.3 * height, 5.0 * height)
        self.c_default = 1.0 * height  # حدس اولیه تا وقتی گرانش بازی یاد گرفته نشده
        self.c_samples = deque(maxlen=60)

    @property
    def c_prior(self):
        if len(self.c_samples) >= 8:
            return float(np.median(self.c_samples))
        return self.c_default

    def update(self, dets, t):
        """dets: لیست (cls, x, y, w, h). تطبیق حریصانه با نزدیک‌ترین مسیر هم‌کلاس."""
        gate = 0.12 * self.W
        pairs = []
        for di, (cls, x, y, w, h) in enumerate(dets):
            for tr in self.tracks:
                if tr.cls != cls:
                    continue
                dt = max(t - tr.last_seen, 1e-3)
                pred = tr.predict(dt, self.c_prior, self.c_range) or tr.pos
                d = np.hypot(x - pred[0], y - pred[1])
                if d < gate + 0.3 * tr.radius:
                    pairs.append((d, di, tr))
        used_d, used_t = set(), set()
        for d, di, tr in sorted(pairs, key=lambda p: p[0]):
            if di in used_d or tr.id in used_t:
                continue
            used_d.add(di)
            used_t.add(tr.id)
            tr.add(t, *dets[di][1:])
            f = tr.fit(self.c_prior, self.c_range)
            if f and f[2] is not None:
                self.c_samples.append(f[2])
        for di, (cls, x, y, w, h) in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(cls, t, x, y, w, h))
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_gap]
