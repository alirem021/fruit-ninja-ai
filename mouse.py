"""کنترل ماوس روی مک با Quartz (خیلی سریع‌تر از pyautogui)."""
import time

from Quartz import (CGEventCreate, CGEventCreateMouseEvent, CGEventGetLocation,
                    CGEventPost, kCGEventLeftMouseDown, kCGEventLeftMouseDragged,
                    kCGEventLeftMouseUp, kCGEventMouseMoved, kCGHIDEventTap,
                    kCGMouseButtonLeft)


def _post(event_type, x, y):
    CGEventPost(kCGHIDEventTap,
                CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft))


def position():
    loc = CGEventGetLocation(CGEventCreate(None))
    return loc.x, loc.y


def swipe(x1, y1, x2, y2, steps=12, duration=0.06):
    """کلیک، کشیدن از (x1,y1) تا (x2,y2)، رها کردن = یک ضربه‌ی چاقو."""
    _post(kCGEventMouseMoved, x1, y1)
    _post(kCGEventLeftMouseDown, x1, y1)
    for i in range(1, steps + 1):
        t = i / steps
        _post(kCGEventLeftMouseDragged, x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
        time.sleep(duration / steps)
    _post(kCGEventLeftMouseUp, x2, y2)
