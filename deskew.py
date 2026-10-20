"""기울어진 스캔 바로 세우기 — 축선·격자선(길고 곧은 무채색 선)이 가장 곧게 서는 각도를 찾는다.

가로선은 y − x·tanθ, 세로선은 x + y·tanθ 로 펴서 줄 높이 히스토그램이 가장 뾰족한 θ 를 고른다(가로·세로 합).
긴 선이 없는 그림(축선 없는 산점 등)·이미 곧은 그림은 0 을 돌려준다. 보정 범위는 ±5°.
"""
import numpy as np
from PIL import Image

import digitize as D

MIN_ANGLE = 0.2   # 이보다 작게 기운 것은 그대로 (보정이 오히려 흐리게 함)
MAX_ANGLE = 5.0


def _points(a, maxn=80000):
    g = D.gray(a)
    a16 = a.astype(np.int16)
    m = (g < 170) & ((a16.max(axis=2) - a16.min(axis=2)) < 60)
    ys, xs = np.nonzero(m)
    if len(xs) > maxn:
        idx = np.random.default_rng(0).choice(len(xs), maxn, replace=False)
        xs, ys = xs[idx], ys[idx]
    return xs.astype(float), ys.astype(float)


def _sharp(xs, ys, t, H, W):
    """θ(tan t) 로 편 가로·세로 줄 히스토그램의 뾰족함 + 가장 긴 줄 길이"""
    yh = np.round(ys - xs * t).astype(int)
    xv = np.round(xs + ys * t).astype(int)
    hy = np.bincount(yh - yh.min())
    hx = np.bincount(xv - xv.min())
    return float((hy.astype(float) ** 2).sum() / max(1, W) + (hx.astype(float) ** 2).sum() / max(1, H)), int(hy.max()), int(hx.max())


def estimate(a):
    """→ (돌릴 각도°, 근거 dict). rotate(a, 각도) 로 바로 선다 (PIL 기준 반시계 +). 내용이 기운 각도는 −각도."""
    H, W = a.shape[:2]
    xs, ys = _points(a)
    if len(xs) < 200:
        return 0.0, {"why": "선이 거의 없음"}
    base, bh, bv = _sharp(xs, ys, 0.0, H, W)

    def scan(lo, hi, step):
        best = None
        for d in np.arange(lo, hi + step / 2, step):
            s, h, v = _sharp(xs, ys, np.tan(np.deg2rad(d)), H, W)
            if best is None or s > best[1]:
                best = (float(d), s, h, v)
        return best
    d, s, h, v = scan(-MAX_ANGLE, MAX_ANGLE, 0.1)
    d, s, h, v = scan(d - 0.1, d + 0.1, 0.01)
    info = {"gain": round(s / max(base, 1e-9), 3), "line_h": h, "line_v": v, "line_h0": bh, "line_v0": bv}
    # 근거: 편 뒤 긴 선(그림 폭·높이의 30% 넘게)이 생기고, 뾰족함이 뚜렷이 커져야 함
    longest = max(h / W, v / H)
    if abs(d) < MIN_ANGLE or longest < 0.4 or s < 1.5 * base:
        return 0.0, info
    return round(d, 2), info


def rotate(a, angle):
    """estimate 가 준 각도만큼 돌려 바로 세움 (흰 바탕, 모서리 잘림 없게 키움)"""
    im = Image.fromarray(a)
    out = im.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    return np.asarray(out)
