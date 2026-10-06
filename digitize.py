"""그래프 디지타이저 엔진 — 좌표는 전부 픽셀 처리 + 축 보정으로 결정론적으로 계산한다 (numpy + Pillow).

보정(calib): {"x": {"p1": [px, py], "p2": [px, py], "v1": 값, "v2": 값, "log": bool}, "y": {...}}
  x 값은 X1→X2 방향으로의 사영, y 값은 Y1→Y2 방향으로의 사영 (축이 서로 수직이면 조금 기운 스캔도 맞음).
  로그축은 log10 공간에서 선형.
"""
import io
import math

import numpy as np
from PIL import Image


# ── 이미지 ───────────────────────────────────────────────────────────────
def load(raw_or_path):
    im = Image.open(io.BytesIO(raw_or_path) if isinstance(raw_or_path, (bytes, bytearray)) else raw_or_path)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.uint8)


def gray(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def runs(b):
    """1차원 bool → [(시작, 끝)] (끝 포함)"""
    d = np.diff(np.concatenate(([0], np.asarray(b, np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1) - 1).tolist()))


def merge(rs, gap):
    out = []
    for s, e in rs:
        if out and s - out[-1][1] - 1 <= gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


# ── 보정 ────────────────────────────────────────────────────────────────
def _f(v, log):
    v = float(v)
    if log:
        if v <= 0:
            raise ValueError("로그축 값은 0보다 커야 합니다")
        return math.log10(v)
    return v


def matrix(calib):
    """픽셀(px, py, 1) → (x', y') 2×3 아핀 행렬 (x', y' 는 로그축이면 log10)"""
    rows = []
    for k in ("x", "y"):
        a = calib[k]
        p1, p2 = np.asarray(a["p1"], float), np.asarray(a["p2"], float)
        u = p2 - p1
        n2 = float(u @ u)
        if n2 < 1e-9:
            raise ValueError(f"{k}축 두 점이 같은 위치입니다")
        f1, f2 = _f(a["v1"], a.get("log")), _f(a["v2"], a.get("log"))
        if f1 == f2:
            raise ValueError(f"{k}축 두 값이 같습니다")
        s = (f2 - f1) / n2
        rows.append([s * u[0], s * u[1], f1 - s * (u @ p1)])
    return np.asarray(rows)


def to_data(calib, px, py):
    M = matrix(calib)
    px, py = np.asarray(px, float), np.asarray(py, float)
    xs = M[0, 0] * px + M[0, 1] * py + M[0, 2]
    ys = M[1, 0] * px + M[1, 1] * py + M[1, 2]
    if calib["x"].get("log"):
        xs = 10 ** xs
    if calib["y"].get("log"):
        ys = 10 ** ys
    return xs, ys


def to_pixel(calib, x, y):
    M = np.vstack([matrix(calib), [0, 0, 1]])
    Mi = np.linalg.inv(M)
    x, y = np.asarray(x, float), np.asarray(y, float)
    xp = np.log10(x) if calib["x"].get("log") else x
    yp = np.log10(y) if calib["y"].get("log") else y
    return Mi[0, 0] * xp + Mi[0, 1] * yp + Mi[0, 2], Mi[1, 0] * xp + Mi[1, 1] * yp + Mi[1, 2]


# ── 축·눈금·눈금 글자 검출 ────────────────────────────────────────────────
def _line_groups(dark, axis):
    """axis=0: 가로선(행마다 최장 연속), 1: 세로선. → [(중심, 두께, 길이, 시작, 끝)]"""
    m = dark if axis == 0 else dark.T
    best = []
    for i in range(m.shape[0]):
        rs = runs(m[i])
        if rs:
            s, e = max(rs, key=lambda r: r[1] - r[0])
            best.append((i, e - s + 1, s, e))
        else:
            best.append((i, 0, 0, 0))
    L = m.shape[1]
    long_ = [b for b in best if b[1] >= 0.25 * L]
    groups = []
    for b in long_:
        if groups and b[0] == groups[-1][-1][0] + 1 and abs(b[2] - groups[-1][-1][2]) < 0.05 * L:
            groups[-1].append(b)
        else:
            groups.append([b])
    out = []
    for g in groups:
        ln = max(b[1] for b in g)
        core = [b for b in g if b[1] >= 0.8 * ln]
        out.append(((core[0][0] + core[-1][0]) / 2 + 0.5, len(core), ln, min(b[2] for b in core), max(b[3] for b in core),
                    core[0][0], core[-1][0]))
    return out


def detect_axes(a, hint=None):
    """x축(아래 가로선)·y축(왼쪽 세로선)·그림 영역 검출. 실패하면 None.
    hint={"x0","y0","x1","y1"}: 그래프 고르기에서 이미 아는 축 상자 — 그 가까이의 선만 축으로 본다(칸막이 그래프·위쪽 범례 틀)."""
    H, W = a.shape[:2]
    g = gray(a)
    dark = g < 140
    hs = _line_groups(dark, 0)
    vs = _line_groups(dark, 1)
    if hint:
        hh = [h for h in hs if abs(h[0] - hint["y1"]) <= 4]
        vv = [v for v in vs if abs(v[0] - hint["x0"]) <= 4]
        if hh and vv:
            xa = max(hh, key=lambda h: h[2])
            ya = max(vv, key=lambda v: v[2])
            c0, c1 = min(h[3] for h in hh), max(h[4] for h in hh)  # 칸막이로 끊긴 아래 축선을 이어서
            box = {"x0": ya[0], "x1": float(hint["x1"]), "y0": float(hint["y0"]), "y1": xa[0]}
            return {"xaxis": {"y": xa[0], "r0": xa[5], "r1": xa[6], "c0": c0, "c1": c1},
                    "yaxis": {"x": ya[0], "c0": ya[5], "c1": ya[6], "r0": ya[3], "r1": ya[4]},
                    "box": box, "frame": True, "synth_y": False}
    synth_y = False
    if not hs or not vs:  # 축선이 없는 그래프(엑셀형: 세로축선 없음·연한 격자만) → 연한 선까지
        light = g < 235
        lhs = _line_groups(light, 0)
        hs = hs or lhs
        if not vs:
            vs = [v for v in _line_groups(light, 1) if v[2] >= 0.6 * H * 0.25]
        if hs and not vs:  # 세로축선 대신 가로선(격자 포함)들의 왼쪽 끝
            hmax = max(h[2] for h in lhs + hs)
            long_ = [h for h in lhs + hs if h[2] >= 0.6 * hmax]
            c0 = int(np.median([h[3] for h in long_]))
            top, bot = min(h[0] for h in long_), max(h[0] for h in long_)
            vs = [(c0 + 0.5, 1, int(bot - top), int(top), int(bot), c0, c0)]
            synth_y = True
    if not hs or not vs:
        return None
    hmax = max(h[2] for h in hs)
    vmax = max(v[2] for v in vs)
    xa = max((h for h in hs if h[2] >= 0.6 * hmax), key=lambda h: h[0])   # 가장 아래 긴 가로선
    ya = min((v for v in vs if v[2] >= 0.6 * vmax), key=lambda v: v[0])   # 가장 왼쪽 긴 세로선
    top = [h for h in hs if h[2] >= 0.6 * hmax and h[0] < xa[0] - 0.2 * H]
    right = [v for v in vs if v[2] >= 0.6 * vmax and v[0] > ya[0] + 0.2 * W]
    box = {"x0": ya[0], "x1": max(right, key=lambda v: v[0])[0] if right else xa[4] + 0.5,
           "y0": min(top, key=lambda h: h[0])[0] if top else ya[3] + 0.5, "y1": xa[0]}
    return {"xaxis": {"y": xa[0], "r0": xa[5], "r1": xa[6], "c0": xa[3], "c1": xa[4]},
            "yaxis": {"x": ya[0], "c0": ya[5], "c1": ya[6], "r0": ya[3], "r1": ya[4]},
            "box": box, "frame": bool(top and right), "synth_y": synth_y}


def _ticks_1d(lengths, maxlen, maxw=4):
    """열(또는 행)마다 축에서 바깥으로 이어진 어두운 길이 → 눈금 [(중심, 길이)]. maxw: 눈금 최대 굵기(축선 굵기에 비례)"""
    cand = (lengths >= 2) & (lengths <= maxlen)
    out = []
    for s, e in runs(cand):
        if e - s + 1 > maxw:
            continue
        w = lengths[s:e + 1].astype(float)
        out.append(((np.arange(s, e + 1) * w).sum() / w.sum() + 0.5, int(w.max())))
    return out


def _run_len(sub):
    """sub[0] 부터 이어진 True 길이 (열마다)"""
    return np.cumprod(sub, axis=0).sum(axis=0)


def detect_ticks(a, ax):
    """x축·y축 눈금 픽셀 위치. 바깥 눈금 → 안쪽 눈금 → 격자선 순으로 찾는다."""
    H, W = a.shape[:2]
    g = gray(a)
    dark = g < 150
    L = max(12, int(0.03 * max(H, W)))
    xa, ya, box = ax["xaxis"], ax["yaxis"], ax["box"]
    wx = max(4, int(2.5 * (xa["r1"] - xa["r0"] + 1)))  # 눈금 굵기 ≈ 축선 굵기 (고해상도·흐린 그림)
    wy = max(4, int(2.5 * (ya["c1"] - ya["c0"] + 1)))
    c0, c1 = int(round(box["x0"])), int(round(box["x1"]))
    r0, r1 = int(round(box["y0"])), int(round(box["y1"]))

    def pick(out_, in_, grid):
        for cand, kind in ((out_, "out"), (in_, "in"), (grid, "grid")):
            if len(cand) >= 2:
                return cand, kind
        return [], "none"

    # x 눈금: 축선 아래(바깥)·위(안쪽)
    below = dark[xa["r1"] + 1: xa["r1"] + 1 + L, c0:c1 + 1]
    above = dark[max(0, xa["r0"] - L): xa["r0"], c0:c1 + 1][::-1]
    xo = [(c0 + p, n) for p, n in _ticks_1d(_run_len(below), L - 1, wx)] if below.size else []
    xi = [(c0 + p, n) for p, n in _ticks_1d(_run_len(above), L - 1, wx)] if above.size else []
    inner = g[r0 + 3:r1 - 2, c0 + 3:c1 - 2] < 235
    gx = []
    if inner.size:
        frac = inner.mean(axis=0)
        gx = [(c0 + 3 + (s + e) / 2 + 0.5, 0) for s, e in runs(frac > 0.5) if e - s < wx]
    xt, xk = pick(xo, xi, gx)

    left = dark[r0:r1 + 1, max(0, ya["c0"] - L): ya["c0"]][:, ::-1].T
    right = dark[r0:r1 + 1, ya["c1"] + 1: ya["c1"] + 1 + L].T
    yo = [(r0 + p, n) for p, n in _ticks_1d(_run_len(left), L - 1, wy)] if left.size else []
    yi = [(r0 + p, n) for p, n in _ticks_1d(_run_len(right), L - 1, wy)] if right.size else []
    gy = []
    if inner.size:
        frac = inner.mean(axis=1)
        gy = [(r0 + 3 + (s + e) / 2 + 0.5, 0) for s, e in runs(frac > 0.5) if e - s < wy]
    yt, yk = pick(yo, yi, gy)
    # 축 끝(모서리)에 붙은 가짜 눈금 제거: 다른 축선 위
    xt = [t for t in xt if abs(t[0] - ya["x"]) > 2 or xk == "out"]
    yt = [t for t in yt if abs(t[0] - xa["y"]) > 2 or yk == "out"]

    def major(ts):
        if not ts:
            return []
        mx = max(t[1] for t in ts)
        return [t[0] for t in ts if t[1] >= 0.75 * mx] if mx else [t[0] for t in ts]
    return {"x": [t[0] for t in xt], "x_major": major(xt), "x_kind": xk, "x_len": max([t[1] for t in xt] or [0]),
            "y": [t[0] for t in yt], "y_major": major(yt), "y_kind": yk, "y_len": max([t[1] for t in yt] or [0])}


def label_blobs(a, ax, ticks):
    """눈금 글자 덩어리의 중심 픽셀 — x: 축 아래 첫 글줄의 가로 덩어리, y: 축 왼쪽 첫 글기둥의 세로 덩어리."""
    H, W = a.shape[:2]
    dark = gray(a) < 160
    xa, ya, box = ax["xaxis"], ax["yaxis"], ax["box"]
    out = {"x": [], "y": [], "x_boxes": [], "y_boxes": [], "x_band": None, "y_band": None}
    # x
    tl = ticks["x_len"] if ticks["x_kind"] == "out" else 0
    top = xa["r1"] + 1 + tl + 1
    mx = int(0.08 * W)
    c0, c1 = max(0, int(box["x0"]) - mx), min(W, int(box["x1"]) + mx)
    band = dark[top:min(H, top + int(0.3 * H)), c0:c1]
    rr = merge(runs(band.any(axis=1)), 2)
    if rr:
        s, e = rr[0]
        h = e - s + 1
        cols = band[s:e + 1].any(axis=0)
        bl = merge(runs(cols), max(3, int(0.45 * h)))
        out["x"] = [(c0 + (bs + be) / 2 + 0.5) for bs, be in bl]
        out["x_boxes"] = [[c0 + bs, top + s, c0 + be + 1, top + e + 1] for bs, be in bl]
        out["x_band"] = [c0, top + s, c1, top + e]
    # y
    tl = ticks["y_len"] if ticks["y_kind"] == "out" else 0
    rgt = ya["c0"] - 1 - tl - 1
    my = int(0.05 * H)
    r0, r1 = max(0, int(box["y0"]) - my), min(H, int(box["y1"]) + my)
    lo = max(0, rgt - int(0.3 * W))
    band = dark[r0:r1, lo:rgt + 1]
    if band.size:
        # 글기둥은 그림 영역 높이 안에서만, 가로로 길게 이어진 줄(범례 틀 등)은 빼고 (위·아래 글자·선이 기둥을 잇지 않게)
        inb = band[max(0, int(box["y0"]) + 3 - r0):max(1, int(box["y1"]) - 2 - r0)]
        inb = inb[inb.mean(axis=1) < 0.6] if inb.size else inb
        cr = merge(runs((inb if inb.size else band).any(axis=0)), 3)
        if cr:
            s, e = cr[-1]  # 축에 가장 가까운 글기둥
            rows = band[:, s:e + 1].any(axis=1)
            bl = merge(runs(rows), 1)
            out["y"] = [(r0 + (bs + be) / 2 + 0.5) for bs, be in bl]
            out["y_boxes"] = [[lo + s, r0 + bs, lo + e + 1, r0 + be + 1] for bs, be in bl]
            out["y_band"] = [lo + s, r0, lo + e, r1]
    return out


def _fit(pairs):
    p = np.array([q[1] for q in pairs], float)
    v = np.array([q[0] for q in pairs], float)
    A = np.vstack([p, np.ones_like(p)]).T
    (k, b), *_ = np.linalg.lstsq(A, v, rcond=None)
    if abs(k) < 1e-15:
        return k, b, float("inf")
    resid_px = (A @ [k, b] - v) / k  # 픽셀 단위 잔차
    return k, b, float(np.sqrt(np.mean(resid_px ** 2))) if len(pairs) > 2 else 0.0


def _robust(pairs):
    """잔차가 큰 짝(모서리에서 겹친 글자·잘못 읽은 값)을 하나씩 빼며 다시 맞춘다 — 최대 1/3, 최소 3쌍은 남김"""
    pairs = list(pairs)
    k, b, r = _fit(pairs)
    dropped = []
    while r > 1.0 and len(pairs) > 3 and len(dropped) < len(pairs) // 3 + 1 and math.isfinite(r):
        res = [abs((k * p + b - v) / k) for v, p in pairs]
        j = int(np.argmax(res))
        trial = pairs[:j] + pairs[j + 1:]
        k2, b2, r2 = _fit(trial)
        if not (r2 < r * 0.7):
            break
        dropped.append(pairs[j])
        pairs, k, b, r = trial, k2, b2, r2
    return pairs, k, b, r, dropped


def ticks_look_log(ticks):
    """작은 눈금 간격이 2,3,…,9 처럼 점점 좁아지는(또는 넓어지는) 줄이 있으면 로그축. 판단 못 하면 None"""
    t = np.sort(np.asarray(ticks, float))
    if len(t) < 6:
        return None
    d = np.diff(t)
    q = d[1:] / np.maximum(d[:-1], 1e-9)
    shrink = (q > 0.6) & (q < 0.95)
    grow = (q > 1.05) & (q < 1.7)
    for m in (shrink, grow):
        if any(e - s_ + 1 >= 4 for s_, e in runs(m)):
            return True
    return False if np.all(np.abs(q - 1) < 0.12) else None


def pair_axis(values, positions, ticks, majors=None, edges=(), log=None, reverse=False, fixed=False, hint=None, text_h=None):
    """VLM 이 읽은 눈금 값(순서대로)과 검출한 글자 위치를 짝짓고, 가까운 큰 눈금(또는 축 끝)으로 끌어 붙인 뒤 최소제곱 직선.
    fixed=True 면 values[i] ↔ positions[i] (None 은 건너뜀). 아니면 개수가 다를 때 어긋남(offset)을 찾는데,
    등간격 눈금에서는 한 칸 밀린 짝도 똑같이 맞으므로 ambiguous=True 로 알린다. 튀는 짝은 빼고 맞춘다(dropped).
    log=None 이면 선형·로그 중 잔차가 작은 쪽 (짝이 둘뿐이라 못 가르면 hint: 눈금 모양·VLM 판단).
    → dict(p1, p2, v1, v2, log, resid_px, n, pairs, ambiguous, dropped) 또는 None"""
    pos = sorted(positions, reverse=reverse)
    vals = [None if v is None else float(v) for v in values] if fixed else [float(v) for v in values if v is not None]
    if len([v for v in vals if v is not None]) < 2 or len(pos) < 2:
        return None
    spacing = float(np.median(np.abs(np.diff(sorted(pos)))))
    tk = np.asarray(sorted(ticks), float)
    big = np.asarray(sorted(set(list(majors if majors is not None else ticks) + list(edges))), float)

    def snap(p):
        if len(big):
            j = int(np.argmin(np.abs(big - p)))
            if abs(big[j] - p) <= max(3.0, min(0.3 * spacing, 0.6 * text_h if text_h else 1e9)):
                return float(big[j])
        if len(tk):
            j = int(np.argmin(np.abs(tk - p)))
            if abs(tk[j] - p) <= 2.5:
                return float(tk[j])
        return float(p)
    cands = []
    n, m = len(vals), len(pos)
    offs = [0] if fixed else range(-(n - 2), m - 1)
    for lg in ([log] if log is not None else [False, True]):
        if lg and min(v for v in vals if v is not None) <= 0:
            continue
        fv = [None if v is None else (math.log10(v) if lg else v) for v in vals]
        for off in offs:
            pairs = [(fv[i], snap(pos[i + off])) for i in range(n) if 0 <= i + off < m and fv[i] is not None]
            if len(pairs) < 2:
                continue
            n0 = len(pairs)  # 빼기 전 짝 수 — 글자를 더 많이 설명하는 짝짓기가 우선
            pairs, k, b, r, dropped = _robust(pairs)
            if not math.isfinite(r) or k == 0:
                continue
            cands.append(((r >= 1.5, -n0, -len(pairs), round(r, 3), hint is not None and lg != hint), lg, pairs, k, b, r, off, dropped))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    _, lg, pairs, k, b, r, off, dropped = cands[0]
    amb = any(c[1] == lg and c[6] != off and c[0][1] == cands[0][0][1] and c[0][2] == cands[0][0][2] and c[5] < max(1.5, r + 0.5)
              for c in cands[1:])
    pairs.sort(key=lambda q: q[1] * (-1 if reverse else 1))
    f1, f2 = pairs[0][0], pairs[-1][0]
    un = (lambda v: 10 ** v) if lg else (lambda v: v)
    return {"p1": (f1 - b) / k, "p2": (f2 - b) / k, "v1": un(f1), "v2": un(f2),
            "log": lg, "resid_px": r, "n": len(pairs), "ambiguous": bool(amb and not fixed),
            "pairs": [[un(v), p] for v, p in pairs], "dropped": [[un(v), p] for v, p in dropped]}


def auto_calibrate(a, xvals, yvals, xlog=None, ylog=None, each=None, hints=None, box_hint=None):
    """축·눈금 검출 + (VLM 이 읽은) 눈금 값 → 보정 제안. 값이 없으면 검출 결과만.
    each={"x": [...], "y": [...]} 는 글자 덩어리 하나하나를 따로 읽은 값(1:1, y 는 아래→위). 둘 다 있으면 더 잘 맞는 쪽.
    hints={"x": bool} 는 VLM 의 로그축 판단 — 눈금 모양(작은 눈금 간격)으로 판단되면 그쪽이 우선."""
    ax = detect_axes(a, box_hint)
    if not ax:
        return {"ok": False, "error": "축선을 찾지 못했습니다 — 수동으로 두 점씩 찍어 주세요"}
    tk = detect_ticks(a, ax)
    lb = label_blobs(a, ax, tk)
    res = {"ok": True, "axes": ax, "ticks": tk, "labels": lb, "box": ax["box"], "calib": None, "warn": []}
    xa, ya, bx = ax["xaxis"], ax["yaxis"], ax["box"]
    each = each or {}
    fits = {}
    for k, vals, lg in (("x", xvals, xlog), ("y", yvals, ylog)):
        pos = lb[k] or tk[k + "_major"]
        edges = (bx["x0"], bx["x1"]) if k == "x" else (bx["y0"], bx["y1"])
        shape = ticks_look_log(tk[k])
        hint = shape if shape is not None else (hints or {}).get(k)
        res.setdefault("log_hint", {})[k] = {"ticks": shape, "vlm": (hints or {}).get(k)}
        bxs = lb.get(k + "_boxes") or []
        th = float(np.median([b[3] - b[1] for b in bxs])) if bxs else None  # 글자 높이 → 끌어 붙일 거리 한도
        args = dict(ticks=tk[k], majors=tk[k + "_major"], edges=edges, log=lg, reverse=(k == "y"), hint=hint, text_h=th)
        opts = []
        if vals:
            c = pair_axis(vals, pos, **args)
            if c:
                c["source"] = "list"
                opts.append(c)
        if each.get(k) and lb[k] and len(each[k]) == len(lb[k]):
            c = pair_axis(each[k], lb[k], fixed=True, **args)
            if c:
                c["source"] = "each"
                opts.append(c)
        mj = tk[k + "_major"]
        if vals and len(mj) == len(vals) >= 3:  # 큰 눈금 수 = 읽은 값 수 → 눈금에 바로 (붙은 글자 '8001000' 처럼 덩어리가 모자랄 때)
            c = pair_axis(vals, mj, fixed=True, **args)
            if c:
                c["source"] = "ticks"
                opts.append(c)
        if opts:
            fits[k] = min(opts, key=lambda c: (c["resid_px"] >= 1.5, c["ambiguous"], -c["n"], c["resid_px"]))
    cx, cy = fits.get("x"), fits.get("y")
    if cx and cy:
        res["calib"] = {"x": {"p1": [cx["p1"], xa["y"]], "p2": [cx["p2"], xa["y"]], "v1": cx["v1"], "v2": cx["v2"], "log": cx["log"]},
                        "y": {"p1": [ya["x"], cy["p1"]], "p2": [ya["x"], cy["p2"]], "v1": cy["v1"], "v2": cy["v2"], "log": cy["log"]}}
        res["fit"] = {"x": cx, "y": cy}
        for k, c in (("x", cx), ("y", cy)):
            if c["resid_px"] > 1.5:
                res["warn"].append(f"{k}축 눈금 값과 위치가 잘 안 맞습니다 (잔차 {c['resid_px']:.1f}px) — 확인하세요")
            if c["ambiguous"]:
                res["warn"].append(f"{k}축: 읽은 눈금 값과 글자 {len(lb[k])}개의 개수가 달라 한 칸 어긋났을 수 있습니다 "
                                   f"— {k.upper()}1·{k.upper()}2 위치의 값을 꼭 확인하세요")
            if c["dropped"]:
                res["warn"].append(f"{k}축: 안 맞는 눈금 {', '.join(f'{v:g}' for v, _ in c['dropped'])} 은(는) 빼고 맞췄습니다 (모서리 글자 겹침 등)")
    return res


# ── 색·마스크 ─────────────────────────────────────────────────────────────
def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(c):
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in c[:3])


def _roi(box, shape, pad=0):
    H, W = shape[:2]
    x0, y0 = max(0, int(math.floor(box["x0"])) + 2 - pad), max(0, int(math.floor(box["y0"])) + 2 - pad)
    x1, y1 = min(W, int(math.ceil(box["x1"])) - 1 + pad), min(H, int(math.ceil(box["y1"])) - 1 + pad)
    return x0, y0, x1, y1


def mask_for(a, color, tol, box, excludes=(), remove_lines=True):
    """그림 영역 안에서 색 ± 허용오차(RGB 거리) 픽셀. 축선(경계 2px)·제외 영역·긴 직선(격자선)은 뺀다.
    색이 있는(검정·회색 아닌) 계열은 축선과 헷갈릴 일이 없으니 영역을 8px 넓혀 축에 걸친 표식도 온전히 잡는다."""
    colored = max(color) - min(color) > 60
    x0, y0, x1, y1 = _roi(box, a.shape, 8 if colored else 0)
    sub = a[y0:y1, x0:x1].astype(np.int32)
    c = np.asarray(color, np.int32)
    m = ((sub - c) ** 2).sum(axis=2) <= tol * tol
    for ex in excludes or ():
        ex0, ey0 = int(min(ex[0], ex[2])) - x0, int(min(ex[1], ex[3])) - y0
        ex1, ey1 = int(math.ceil(max(ex[0], ex[2]))) - x0, int(math.ceil(max(ex[1], ex[3]))) - y0
        m[max(0, ey0):max(0, ey1 + 1), max(0, ex0):max(0, ex1 + 1)] = False
    if remove_lines and m.size:
        h, w = m.shape
        m[m.mean(axis=1) > 0.6, :] = False  # 가로 격자선·같은 색 직선
        m[:, m.mean(axis=0) > 0.6] = False
        for sl in (np.s_[:4, :], np.s_[-4:, :]):  # 테두리 가까이 남은 축선 조각(굵은·흐린 축선)
            sub = m[sl]
            sub[sub.mean(axis=1) > 0.25, :] = False
        for sl in (np.s_[:, :4], np.s_[:, -4:]):
            sub = m[sl]
            sub[:, sub.mean(axis=0) > 0.25] = False
        _clear_edge_ticks(m)
    return m, (x0, y0)


def _clear_edge_ticks(m, L=None):
    """테두리(또는 3px 안)에서 시작하는 짧고 가는 수직 조각(안쪽 눈금)을 지운다 — 네 변 모두. 길이 한도는 그림 크기에 비례"""
    L = L or max(14, int(0.025 * max(m.shape)))
    if min(m.shape) < 3 * L:
        return
    for view in (m, m[::-1], m.T, m.T[::-1]):  # 위·아래·왼쪽·오른쪽 (view 는 m 을 공유)
        band = view[:L + 4]
        w = band.shape[1]
        first = np.full(w, -1)
        for k in range(3, -1, -1):
            first[band[k]] = k
        ln = np.zeros(w, int)
        for k in range(4):
            cols = first == k
            if cols.any():
                ln[cols] = np.cumprod(band[k:, cols], axis=0).sum(axis=0)
        cand = (first >= 0) & (ln >= 2) & (ln < L)
        for s_, e in runs(cand):
            tip = int((first[s_:e + 1] + ln[s_:e + 1]).max())
            if e - s_ < max(3, L // 5) and not view[tip + 1:tip + 3, max(0, s_ - 1):e + 2].any():
                for c in range(s_, e + 1):
                    view[first[c]:first[c] + ln[c], c] = False


def suggest_colors(a, box, k=6):
    """그림 영역 안의 계열 후보 색(채도 높은 색 + 거의 검정) — 안티앨리어싱 번짐(흰색과 섞인 색)은 뺀다."""
    x0, y0, x1, y1 = _roi(box, a.shape)
    img = a[y0:y1, x0:x1]
    if not img.size:
        return []
    ink = gray(img) < 235
    lines = np.zeros(ink.shape, bool)  # 격자선·테두리(가로·세로로 길게 이어진 줄)는 후보에서 뺌
    lines[ink.mean(axis=1) > 0.5, :] = True
    lines[:, ink.mean(axis=0) > 0.5] = True
    sub = img[~lines].astype(np.int32)
    mx, mn = sub.max(1), sub.min(1)
    cand = sub[((mx - mn) > 60) | (mx < 80) | (((mx - mn) < 20) & (mx > 90) & (mx < 170))]  # 채도 높은 색·검정·중간 회색
    if not len(cand):
        return []
    q = cand // 16
    keys, inv, cnt = np.unique(q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2], return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)
    clusters = []  # [색, 개수]
    for i in order:
        if cnt[i] < 8:
            break
        col = cand[inv.ravel() == i].mean(axis=0)
        for cl in clusters:
            if np.sqrt(((cl[0] - col) ** 2).sum()) < 70:
                cl[1] += int(cnt[i])
                break
        else:
            clusters.append([col, int(cnt[i])])
    clusters.sort(key=lambda c: -c[1])
    keep = []
    white = np.array([255.0, 255, 255])
    for col, n in clusters:
        if n < max(25, 0.01 * clusters[0][1]):
            continue
        blend = False
        for kc, kn in keep:  # 흰색과 섞인(번진) 색이면 버림 — 단 그 색만큼 많으면(회색 계열 등) 따로 둠
            if n >= 0.25 * kn and col.max() - col.min() < 20 and col.max() > 90:  # 중간 회색 계열은 검정의 번짐과 구분: 충분히 많을 때만
                continue
            d = white - kc
            t = np.clip(((col - kc) @ d) / (d @ d + 1e-9), 0, 1)
            if t > 0.15 and np.sqrt(((kc + t * d - col) ** 2).sum()) < 30:
                blend = True
                break
        if not blend:
            keep.append((col, n))
        if len(keep) >= k:
            break
    return [{"color": rgb2hex(c), "count": n} for c, n in keep]


COLOR_WORDS = {"red": "#d62728", "빨강": "#d62728", "빨간": "#d62728", "blue": "#1f77b4", "파랑": "#1f77b4", "파란": "#1f77b4",
               "green": "#2ca02c", "초록": "#2ca02c", "녹색": "#2ca02c", "orange": "#ff7f0e", "주황": "#ff7f0e",
               "black": "#000000", "검정": "#000000", "검은": "#000000", "purple": "#9467bd", "보라": "#9467bd",
               "magenta": "#e377c2", "pink": "#e377c2", "분홍": "#e377c2", "brown": "#8c564b", "갈색": "#8c564b",
               "gray": "#7f7f7f", "grey": "#7f7f7f", "회색": "#7f7f7f", "cyan": "#17becf", "하늘": "#17becf",
               "olive": "#bcbd22", "yellow": "#e6c700", "노랑": "#e6c700", "navy": "#000080", "teal": "#008080"}


def color_word(s):
    s = (s or "").strip().lower()
    if s.startswith("#") and len(s) == 7:
        return s
    for w in sorted(COLOR_WORDS, key=len, reverse=True):
        if w in s:
            return COLOR_WORDS[w]
    return None


def match_legend(legend, suggestions):
    """범례(이름·색 단어) ↔ 검출한 색 후보 짝짓기 (가까운 색 순, 중복 없이)"""
    out, used = [], set()
    for it in legend:
        want = color_word(it.get("color"))
        best = None
        if want:
            w = np.array(hex2rgb(want), float)
            for i, s in enumerate(suggestions):
                if i in used:
                    continue
                d = np.sqrt(((np.array(hex2rgb(s["color"]), float) - w) ** 2).sum())
                if best is None or d < best[0]:
                    best = (d, i)
        if best and best[0] < 160:
            used.add(best[1])
            out.append({"name": it.get("name") or "", "color": suggestions[best[1]]["color"], "style": it.get("style") or "",
                        "marker": it.get("marker") or "", "axis": it.get("axis") or ""})
        else:
            out.append({"name": it.get("name") or "", "color": want or "", "style": it.get("style") or "",
                        "marker": it.get("marker") or "", "axis": it.get("axis") or ""})
    for i, s in enumerate(suggestions):
        if i not in used and not legend:
            out.append({"name": f"계열 {len(out) + 1}", "color": s["color"], "style": ""})
    return out


# ── 추출 ────────────────────────────────────────────────────────────────
def extract_line(m, off):
    """열마다 마스크 덩어리 중심 → 곡선 픽셀 점. 한 열에 덩어리가 여럿이면 앞 점에 가까운 것(범례·글자 무시)."""
    x0, y0 = off
    h, w = m.shape
    cols = []
    for c in range(w):
        col = m[:, c]
        if not col.any():
            continue
        rs = runs(col)
        cols.append((c, [((s + e) / 2, e - s + 1) for s, e in rs]))
    if not cols:
        return []
    # 시작점: 덩어리 하나뿐인 열이 이어지는 가장 긴 구간에서 출발해 양쪽으로 추적
    single = [len(r) == 1 for _, r in cols]
    best, cur = (0, 0), None
    for i, s in enumerate(single + [False]):
        if s and cur is None:
            cur = i
        elif not s and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    seed = (best[0] + best[1]) // 2 if best[1] > best[0] else 0
    chosen = [None] * len(cols)

    def track(idx_iter, prev):
        hist = [prev] if prev is not None else []
        for i in idx_iter:
            c, rs = cols[i]
            if len(hist) >= 2:
                (ca, ya), (cb, yb) = hist[-2], hist[-1]
                pred = yb + (yb - ya) / max(1, cb - ca) * (c - cb) if abs(cb - ca) < 6 else yb
            elif hist:
                pred = hist[-1][1]
            else:
                pred = None
            if pred is None:
                y = max(rs, key=lambda r: r[1])[0]
            else:
                y = min(rs, key=lambda r: abs(r[0] - pred))[0]
                if hist and abs(c - hist[-1][0]) <= 3 and abs(y - hist[-1][1]) > max(0.25 * h, 40) and len(rs) == 1:
                    continue  # 갑작스러운 큰 점프(글자·범례 조각) — 건너뜀
            chosen[i] = y
            hist.append((c, y))
            hist = hist[-3:]
    rs0 = cols[seed][1]
    chosen[seed] = max(rs0, key=lambda r: r[1])[0]
    track(range(seed + 1, len(cols)), (cols[seed][0], chosen[seed]))
    track(range(seed - 1, -1, -1), (cols[seed][0], chosen[seed]))
    seq = [(c, y) for (c, _), y in zip(cols, chosen) if y is not None]
    # 큰 점프로 갈린 짧은 조각(범례 견본선·글자)은 버린다
    jump = max(0.15 * h, 20)
    pieces = [[seq[0]]] if seq else []
    for p in seq[1:]:
        q = pieces[-1][-1]
        if abs(p[1] - q[1]) > jump + 2 * (p[0] - q[0]):
            pieces.append([p])
        else:
            pieces[-1].append(p)
    if len(pieces) > 1:
        top = max(len(p) for p in pieces)
        pieces = [p for p in pieces if len(p) >= max(5, 0.08 * len(seq)) or len(p) == top]
    return [[x0 + c + 0.5, y0 + y + 0.5] for p in pieces for c, y in p]


def find_legend_boxes(a, box):
    """그림 영역 안의 테두리 있는 사각형(범례 상자) → [[x0, y0, x1, y1]]. 둥근 모서리 허용."""
    x0, y0, x1, y1 = _roi(box, a.shape)
    x0, y0, x1, y1 = x0 + 2, y0 + 2, x1 - 2, y1 - 2
    if x1 - x0 < 40 or y1 - y0 < 40:
        return []
    ink = gray(a[y0:y1, x0:x1]) < 235
    h, w = ink.shape
    hseg = [(r, s, e) for r in range(h) for s, e in runs(ink[r]) if 25 <= e - s + 1 <= 0.85 * w]
    vseg = [(c, s, e) for c in range(w) for s, e in runs(ink[:, c]) if 12 <= e - s + 1 <= 0.85 * h]
    out = []
    minw = max(50, int(0.06 * w))
    for r, s, e in hseg:
        if e - s + 1 < minw:
            continue
        for r2, s2, e2 in hseg:
            if r2 - r < 18 or abs(s2 - s) > 2 or abs(e2 - e) > 2:
                continue
            t = min(7, max(2, int(0.12 * min(e - s, r2 - r))))  # 둥근 모서리 허용폭 — 동그라미 표식은 안 걸리게
            lv = any(abs(c - s) <= t and vs <= r + t and ve >= r2 - t for c, vs, ve in vseg)
            rv = any(abs(c - e) <= t and vs <= r + t and ve >= r2 - t for c, vs, ve in vseg)
            if lv and rv and (e - s) * (r2 - r) < 0.4 * w * h and ink[r + 2:r2 - 1, s + 2:e - 1].mean() < 0.5:  # 범례 상자는 속이 대개 비어 있음 (꽉 찬 막대가 아님)
                bx = [x0 + s - t, y0 + r - 2, x0 + e + t + 1, y0 + r2 + 3]
                if not any(abs(bx[0] - o[0]) < 4 and abs(bx[1] - o[1]) < 4 for o in out):
                    out.append(bx)
                break
    return out


def _shift_and(m, r, op):
    """정사각 (2r+1) 창으로 팽창(op=or)·침식(op=and)"""
    out = m.copy()
    h, w = m.shape
    pad = np.pad(m, r, constant_values=(op == "and") and False)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sl = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            out = (out & sl) if op == "and" else (out | sl)
    return out


def _components(m, join=0):
    """연결 성분 → [(픽셀 수, 중심x, 중심y, 폭, 높이, 왼쪽, 위)]. join>0 이면 그만큼 떨어진 조각(선에 끊긴 빈 표식)을 한 덩어리로."""
    grp = _shift_and(m, join, "or") if join else m
    try:
        from scipy import ndimage
        lab, n = ndimage.label(grp, structure=np.ones((3, 3)))
    except ImportError:
        lab, n = _label_py(grp)
    if not n:
        return []
    lab = np.where(m, lab, 0)
    ys, xs = np.nonzero(lab)
    ls = lab[ys, xs]
    cnt = np.bincount(ls, minlength=n + 1)
    sx = np.bincount(ls, xs, minlength=n + 1)
    sy = np.bincount(ls, ys, minlength=n + 1)
    out = []
    order = np.argsort(ls, kind="stable")
    bounds = np.searchsorted(ls[order], np.arange(1, n + 2))
    for i in range(1, n + 1):
        if not cnt[i]:
            continue
        idx = order[bounds[i - 1]:bounds[i]]
        out.append((int(cnt[i]), sx[i] / cnt[i], sy[i] / cnt[i], int(xs[idx].max() - xs[idx].min() + 1), int(ys[idx].max() - ys[idx].min() + 1),
                    int(xs[idx].min()), int(ys[idx].min())))
    return out


def _label_py(m):
    """scipy 없을 때 8-연결 성분 번호 (파이썬 BFS)"""
    lab = np.zeros(m.shape, np.int32)
    h, w = m.shape
    n = 0
    for y0, x0 in zip(*np.nonzero(m)):
        if lab[y0, x0]:
            continue
        n += 1
        lab[y0, x0] = n
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for yy in (y - 1, y, y + 1):
                for xx in (x - 1, x, x + 1):
                    if 0 <= yy < h and 0 <= xx < w and m[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = n
                        stack.append((yy, xx))
    return lab, n


def _merge_split(comps):
    """다른 선이 가로질러 둘로 갈린 빈 표식 조각을 합친다: 합친 상자가 보통 표식 크기의 1.3배 안이면 한 표식
    (서로 다른 두 표식이면 합친 상자가 표식 두 개 크기라 합쳐지지 않음)"""
    blob = [c for c in comps if not _linelike(c)]
    if len(blob) < 3:
        return comps
    size = float(np.median([max(c[3], c[4]) for c in blob]))
    out = list(comps)
    i = 0
    while i < len(out):
        a = out[i]
        for j in range(i + 1, len(out)):
            b = out[j]
            if _linelike(a) or _linelike(b):
                continue
            x0, y0 = min(a[5], b[5]), min(a[6], b[6])
            x1, y1 = max(a[5] + a[3], b[5] + b[3]), max(a[6] + a[4], b[6] + b[4])
            if x1 - x0 <= 1.3 * size and y1 - y0 <= 1.3 * size:
                n = a[0] + b[0]
                out[i] = (n, (a[1] * a[0] + b[1] * b[0]) / n, (a[2] * a[0] + b[2] * b[0]) / n, x1 - x0, y1 - y0, x0, y0)
                del out[j]
                break
        else:
            i += 1
    return out


def _linelike(c):
    """가늘고 긴(가로세로비) 또는 크고 성긴(대각선·곡선) 덩어리"""
    n, w, h = c[0], c[3], c[4]
    return max(w, h) > 3 * min(w, h) + 2 or (max(w, h) > 30 and n < 0.2 * w * h)


def _boxes(m):
    """작은 연결 성분의 바운딩 박스 [(x0, y0, x1, y1)] (글자 판정용)"""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m, structure=np.ones((3, 3)))
        return [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in ndimage.find_objects(lab) if s]
    except ImportError:
        lab, n = _label_py(m)
        out = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            out.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        return out


def _text_filter(pts_c, dark, size, ink=None, same=None, why=None):
    """글자(가까이 줄지어 선 작은 덩어리)와 범례 견본(오른쪽에 글자가 붙은 표식)을 골라낸다 → 버릴 번호 집합"""
    bx = [b for b in _boxes(dark) if 4 <= (b[3] - b[1]) <= 3 * size + 8 and (b[2] - b[0]) <= 12 * size + 40]  # 흐린 그림은 낱말이 한 덩어리
    if not bx:
        return set()
    B = np.asarray(bx, float)
    Bh = B[:, 3] - B[:, 1]
    ci = [b for b in _boxes(ink) if 4 <= (b[3] - b[1]) <= 3 * size + 8] if ink is not None else []
    Ci = np.asarray(ci, float) if ci else None  # 색 있는 견본까지 (범례 줄 맞춤 확인용)

    def vov(i_y0, i_y1):
        """세로로 절반 이상 겹치는 덩어리"""
        return np.minimum(B[:, 3], i_y1) - np.maximum(B[:, 1], i_y0) >= 0.5 * np.minimum(i_y1 - i_y0, Bh)
    drop = set()
    for i, (n, cx, cy, w, h, x0, y0) in enumerate(pts_c):
        x1, y1 = x0 + w, y0 + h
        gap = max(3.0, 0.3 * h)
        own = (B[:, 0] >= x0 - 1) & (B[:, 1] >= y0 - 1) & (B[:, 2] <= x1 + 1) & (B[:, 3] <= y1 + 1)  # 자기 조각
        # 글자: 좌우로 아주 가까운(gap 이하) 같은 줄 덩어리가 있음
        lr = vov(y0, y1) & (((B[:, 0] - x1 >= -1) & (B[:, 0] - x1 <= gap)) | ((x0 - B[:, 2] >= -1) & (x0 - B[:, 2] <= gap))) & ~own
        if same is not None:  # 글자 조각끼리는 같은 색 — 검은 이름표가 붙은 색 표식은 글자가 아님
            lr &= np.array([bool(same[int(b[1]):int(b[3]), int(b[0]):int(b[2])].any()) for b in B]) if lr.any() else lr
        near = bool(lr.any())
        # 범례 견본: 오른쪽 (글자 높이 3배 안)에 글자 덩어리 B1, 그 바로 옆(글자 사이 간격)에 비슷한 높이의 B2
        def text_right(bx0, by0, bx1, by1):
            hh = by1 - by0
            hits = []
            g1 = B[:, 0] - bx1
            for j in np.flatnonzero(vov(by0 - 0.3 * hh, by1 + 0.3 * hh) & (g1 >= 0) & (g1 <= np.maximum(3 * Bh + 0.5 * size, max(6 * size, 45)))
                                    & ~((B[:, 0] >= bx0 - 1) & (B[:, 2] <= bx1 + 1))):
                g2 = B[:, 0] - B[j, 2]
                r = Bh / max(Bh[j], 1)
                if (vov(B[j, 1], B[j, 3]) & (g2 >= -1) & (g2 <= Bh[j] + 1) & (r > 0.25) & (r < 2.5)).any():
                    hits.append(float(B[j, 0]))
            return min(hits) if hits else None  # 글자 시작 x (가장 왼쪽 글자)
        legend = False
        tx = text_right(x0, y0, x1, y1)
        if tx is not None:
            # 범례는 견본이 세로로 줄지어 있다: 바로 위·아래에 비슷한 크기의 견본(다른 색 포함)도 오른쪽에 글자를 달고 있어야 함
            # (글자 이름표가 붙은 데이터 점은 남김). 견본이 하나뿐인 범례는 테두리 상자·제외 영역으로.
            C = Ci if Ci is not None else B
            Cw, Ch = C[:, 2] - C[:, 0], C[:, 3] - C[:, 1]
            cxs = (C[:, 0] + C[:, 2]) / 2
            mine = np.flatnonzero((C[:, 0] <= cx) & (C[:, 2] >= cx) & (C[:, 1] <= cy) & (C[:, 3] >= cy))
            if len(mine):  # 견본 전체(선+표식) 크기로 비교
                j0 = mine[np.argmin(Cw[mine] * Ch[mine])]
                ox, ow, oh, oy = cxs[j0], Cw[j0], Ch[j0], C[j0, 1]
            else:
                ox, ow, oh, oy = cx, w, h, y0
            sim = ((np.abs(cxs - ox) <= 1.5) & (np.abs(Cw - ow) <= 0.5 * ow + 3) & (np.abs(Ch - oh) <= 0.5 * oh + 3)
                   & (np.abs(C[:, 1] - oy) > 0.8 * oh) & (np.abs(C[:, 1] - oy) < 2.2 * oh + 4))  # 범례 줄 간격
            # 범례 글자는 왼쪽이 줄 맞춰 있다 (그림 속 점 이름표는 대개 제각각)
            legend = any(t is not None and abs(t - tx) <= 3 for t in (text_right(*C[j]) for j in np.flatnonzero(sim)))
        if near or legend:
            drop.add(i)
            if why is not None:
                why[i] = "legend" if legend and not near else "text"
    return drop


def _correlate(img, T):
    """img 와 T 의 상호상관(같은 크기, T 중심 기준) — numpy FFT"""
    H, W = img.shape
    h, w = T.shape
    fh, fw = H + h - 1, W + w - 1
    F = np.fft.rfft2(img, (fh, fw)) * np.fft.rfft2(T[::-1, ::-1], (fh, fw))
    full = np.fft.irfft2(F, (fh, fw))
    return full[h // 2:h // 2 + H, w // 2:w // 2 + W]


def _template_hits(m, clean, lines, thr=0.62):
    """선·다른 선과 붙은 덩어리 속 표식 찾기: 깨끗한 표식들의 평균 모양(틀)으로 상호상관 → 봉우리 = 표식 중심"""
    S = int(np.median([max(c[3], c[4]) for c in clean])) + 4
    S += 1 - S % 2
    r = S // 2
    H, W = m.shape
    patches = []
    for c in clean:
        cx, cy = int(round(c[1])), int(round(c[2]))
        if r <= cx < W - r and r <= cy < H - r:
            patches.append(m[cy - r:cy + r + 1, cx - r:cx + r + 1].astype(float))
    if len(patches) < 2:
        return []
    T = np.mean(patches, axis=0)
    self = float((T * T).sum())
    hits = []
    for c in lines:
        x0, y0 = max(0, c[5] - r), max(0, c[6] - r)
        x1, y1 = min(W, c[5] + c[3] + r), min(H, c[6] + c[4] + r)
        sc = _correlate(m[y0:y1, x0:x1].astype(float), T) / self
        while True:
            j = int(np.argmax(sc))
            py, px = divmod(j, sc.shape[1])
            if sc[py, px] < thr:
                break
            ys, xs = slice(max(0, py - 1), py + 2), slice(max(0, px - 1), px + 2)
            wgt = np.clip(sc[ys, xs] - 0.9 * sc[py, px], 0, None)
            gy, gx = np.mgrid[ys, xs]
            cy = float((gy * wgt).sum() / wgt.sum()) if wgt.sum() else py
            cx = float((gx * wgt).sum() / wgt.sum()) if wgt.sum() else px
            hits.append((int(T.sum()), x0 + cx, y0 + cy, S - 4, S - 4, int(x0 + cx - r + 2), int(y0 + cy - r + 2)))
            sc[max(0, py - int(0.75 * S)):py + int(0.75 * S) + 1, max(0, px - int(0.75 * S)):px + int(0.75 * S) + 1] = 0
    return hits


def extract_points(m, off, min_area=4, max_area=None, rgb=None):
    """산점: 연결 성분 중심. 선에 끊긴 빈 표식 조각은 합치고(2px), 잡티·선·글자 조각은 뺀다.
    표식이 같은 색 선으로 이어져 있으면(선+표식 계열) 선이 사라질 만큼 침식한 뒤 표식 중심을 찾는다.
    rgb(같은 영역 그림)가 있으면 글자·범례 견본(오른쪽에 글자가 붙은 표식)도 뺀다."""
    x0, y0 = off
    comps = [c for c in _components(m, join=2) if c[0] >= min_area]
    warn = []
    if not comps:
        return [], []
    blob = [c for c in comps if not _linelike(c)]
    if len(blob) >= 3:  # 잡티(눈금 조각·번진 점) 바닥: 큰 덩어리(상위 10%)의 15% 미만은 버림
        floor = 0.15 * float(np.percentile([c[0] for c in blob], 90))
        comps = [c for c in comps if c[0] >= floor or _linelike(c)]
    blob = [c for c in comps if not _linelike(c)]
    med = float(np.median([c[0] for c in blob])) if blob else 0
    big_line = [c for c in comps if _linelike(c) and (not med or c[0] > 3 * med)]
    comps = _merge_split(comps)
    if big_line:  # 선+표식: 침식 반경을 키워 가며 선이 없어지는 첫 반경
        for r in range(1, 7):
            er = _shift_and(m, r, "and")
            cs = _merge_split([c for c in _components(er) if c[0] >= 2])
            if cs and not any(_linelike(c) and c[0] > 12 for c in cs):
                comps = cs
                blob = [c for c in comps if not _linelike(c)]
                med = float(np.median([c[0] for c in blob])) if blob else 0
                warn.append(f"선으로 이어진 표식 — {r}px 침식으로 선을 지우고 표식 중심을 찾았습니다 (빈 표식은 사라질 수 있음)")
                break
        else:
            warn.append("표식이 선과 붙어 분리하지 못했습니다 — '곡선' 방식이나 수동 점을 쓰세요")
    else:  # 다른 선(점선·맞춤선)과 붙은 표식: 깨끗한 표식으로 만든 틀로 찾기
        clean = [c for c in comps if not _linelike(c)]
        if len(clean) >= 2:
            size = float(np.median([max(c[3], c[4]) for c in clean]))
            lines = [c for c in comps if _linelike(c) and max(c[3], c[4]) >= 0.8 * size]
            hits = _template_hits(m, clean, lines) if lines else []
            if hits:
                comps = clean + hits
                warn.append(f"선과 붙은 표식 {len(hits)}개를 모양 맞추기로 찾았습니다 — 겹쳐 보기로 확인")
    keep = [c for c in comps if not (max_area and c[0] > max_area) and not _linelike(c)]  # 선 조각
    if rgb is not None and keep:
        size = float(np.median([max(c[3], c[4]) for c in keep]))
        rg_ = rgb.astype(np.int16)
        dark = ((gray(rgb) < 160) & ((rg_.max(axis=2) - rg_.min(axis=2)) < 80)) | m  # 글자(검정) + 이 계열 — 다른 색 표식은 글자로 안 봄
        _clear_edge_ticks(dark)  # 안쪽 눈금은 글자가 아님
        dark[:2] = dark[-2:] = False
        dark[:, :2] = dark[:, -2:] = False
        pc = keep
        rg = rgb.astype(np.int16)
        inkm = (gray(rgb) < 200) | ((rg.max(axis=2) - rg.min(axis=2)) > 60)
        inkm[:2] = inkm[-2:] = False
        inkm[:, :2] = inkm[:, -2:] = False
        drop = _text_filter(pc, dark, size, inkm, m)
        if drop:
            warn.append(f"글자·범례 견본으로 보이는 {len(drop)}개는 뺐습니다 — 겹쳐 보기로 확인")
            keep = [c for i, c in enumerate(keep) if i not in drop]
    pts = []
    for n, cx, cy, w, h, *_ in keep:
        if med and n > 1.6 * med:
            warn.append(f"({x0 + cx + 0.5:.0f}, {y0 + cy + 0.5:.0f}) 근처에 표식 {n / med:.1f}개쯤 겹침 — 수동으로 확인")
        pts.append([x0 + cx + 0.5, y0 + cy + 0.5])
    pts.sort()
    return pts, warn


def resample(calib, pts, how="all", n=0, step=0.0, gap_px=None):
    """곡선 픽셀 점 → 데이터 좌표. how: all(열마다) | n(점 수, 로그축은 로그 간격) | step(x 간격). 큰 틈은 보간하지 않음."""
    if len(pts) < 2:
        xs, ys = to_data(calib, [p[0] for p in pts], [p[1] for p in pts])
        return [[float(x), float(y)] for x, y in zip(xs, ys)]
    P = np.asarray(pts, float)
    xs, ys = to_data(calib, P[:, 0], P[:, 1])
    o = np.argsort(xs)
    xs, ys, px = xs[o], ys[o], P[o, 0]
    if how == "all" or not (n or step):
        return [[float(x), float(y)] for x, y in zip(xs, ys)]
    xl, yl = bool(calib["x"].get("log")), bool(calib["y"].get("log"))
    fx = np.log10(xs) if xl else xs
    fy = np.log10(ys) if yl else ys
    if how == "n" and n >= 2:
        tx = np.linspace(fx[0], fx[-1], int(n))
    else:
        st = float(step)
        if st <= 0:
            raise ValueError("x 간격은 0보다 커야 합니다")
        if xl:
            tx = np.arange(math.ceil(fx[0] / st - 1e-9) * st, fx[-1] + 1e-12, st)  # 로그축: 10^간격 배수
        else:
            lo = math.ceil(xs[0] / st - 1e-9) * st
            tx = np.arange(lo, xs[-1] + st * 1e-9, st)
    if len(tx) > 20000:
        raise ValueError("점이 너무 많습니다 — 간격을 넓혀 주세요")
    gap_px = gap_px or max(8.0, 0.05 * (px.max() - px.min()))
    ty = np.interp(tx, fx, fy)
    j = np.clip(np.searchsorted(fx, tx), 1, len(fx) - 1)
    ok = np.abs(px[j] - px[j - 1]) <= gap_px
    out = []
    for x, y, g in zip(tx, ty, ok):
        if g:
            out.append([float(10 ** x if xl else x), float(10 ** y if yl else y)])
    return out


def mask_png(m, off, shape, color):
    """마스크를 원본 크기 투명 PNG 로 (겹쳐 보기용)"""
    H, W = shape[:2]
    rgba = np.zeros((H, W, 4), np.uint8)
    x0, y0 = off
    h, w = m.shape
    c = hex2rgb(color) if isinstance(color, str) else color
    inv = (255 - c[0], 255 - c[1], 255 - c[2]) if sum(c) > 200 else (255, 0, 255)
    rgba[y0:y0 + h, x0:x0 + w][m] = (*inv, 150)
    b = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(b, "PNG", optimize=True)
    return b.getvalue()


def run_series(a, calib, box, s, excludes=(), legends=None):
    """한 계열 자동 추출 → {px: 픽셀 점, data: 데이터 점, warn, mask, off}. 범례 상자는 자동으로 뺀다(skip_legend)."""
    color = hex2rgb(s["color"]) if isinstance(s.get("color"), str) else tuple(s["color"])
    tol = float(s.get("tol") or 60)
    if s.get("how") == "markers" and s.get("mode") != "points":  # 선 위 표식 중심 = 데이터 점
        r = auto_series(a, calib, box, rgb2hex(color), legends=legends, tol=tol)
        if r["kind"] != "line+markers":
            r["warn"] = ["선 위에서 표식을 찾지 못해 50점으로 나눠 뽑았습니다"] + r["warn"]
        m, off = mask_for(a, color, tol, box, list(excludes or ()) + (legends or []), s.get("remove_lines", True))
        return {"px": r["px"], "data": r["data"], "warn": r["warn"], "mask": m, "off": off}
    ex = list(excludes or ())
    if s.get("skip_legend", True):
        ex += find_legend_boxes(a, box) if legends is None else legends
    m, off = mask_for(a, color, tol, box, ex, s.get("remove_lines", True))
    warn = []
    if not m.any():
        return {"px": [], "data": [], "warn": ["이 색의 픽셀이 영역 안에 없습니다 — 색·허용오차·영역을 확인"], "mask": m, "off": off}
    if s.get("mode") == "points":
        x0, y0 = off
        px, warn = extract_points(m, off, int(s.get("min_area") or 4), s.get("max_area") or None,
                                  a[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] if s.get("skip_text", True) else None)
        data = []
        if px:
            xs, ys = to_data(calib, [p[0] for p in px], [p[1] for p in px])
            data = [[float(x), float(y)] for x, y in zip(xs, ys)]
    else:
        px = extract_line(m, off)
        data = resample(calib, px, s.get("how") or "all", int(s.get("n") or 0), float(s.get("step") or 0))
    if m.mean() > 0.25:
        warn.append("마스크가 영역의 25% 이상 — 허용오차가 너무 크거나 배경색을 골랐습니다")
    return {"px": px, "data": data, "warn": warn, "mask": m, "off": off}


def auto_series(a, calib, box, color, legends=None, tol=60, n=50, hint=None):
    """추출 방식 자동 선택 — 화면에서 ▶ 추출을 따로 누르지 않아도 되게.
    선(열마다 이어짐) / 점(띄엄띄엄 표식) / 선+표식(표식 = 실제 데이터 점 → 표식 중심을 씀) 을 마스크 모양으로 가른다.
    hint: VLM 범례 모양('line'·'markers'·'dashed'), 참고만. → {mode, how, px, data, warn, kind}"""
    s = dict(color=color, tol=tol)
    ln = run_series(a, calib, box, dict(s, mode="line", how="all"), legends=legends)
    pt = run_series(a, calib, box, dict(s, mode="points"), legends=legends)
    lpx, mk = ln["px"], pt["px"]
    warn = []
    if not lpx and not mk:
        return {"mode": "line", "how": "n", "px": [], "data": [], "warn": ln["warn"], "kind": "none"}
    span = (max(p[0] for p in lpx) - min(p[0] for p in lpx) + 1) if lpx else 0
    dens = len(lpx) / span if span else 0  # 선이 지나는 열 비율 (점선이면 낮음, 산점이면 아주 낮음)
    m = ln["mask"]
    blob = len(mk) >= 2 and (dens < 0.35 or (hint == "markers" and dens < 0.6))
    if blob:
        out = {"mode": "points", "how": "all", "px": mk, "data": pt["data"], "warn": pt["warn"], "kind": "points"}
    else:
        on = 0
        if len(mk) >= 3 and lpx:
            P = np.asarray(lpx)
            for x, y in mk:
                j = int(np.argmin(np.abs(P[:, 0] - x)))
                if abs(P[j, 0] - x) <= 3 and abs(P[j, 1] - y) <= 6:
                    on += 1
        if len(mk) >= 3 and on >= 0.7 * len(mk):  # 선 위에 표식 → 표식이 데이터 점
            mk = sorted(mk)
            P = np.asarray(lpx)
            dx = float(np.median(np.diff([p[0] for p in mk])))
            fill = []
            for (xa, ya_), (xb, yb_) in zip(mk[:-1], mk[1:]):  # 등간격 표식 사이가 2·3칸 비면 (다른 선·범례에 가린 표식) 선 위치로
                k = int(round((xb - xa) / dx)) if dx > 4 else 1
                if 2 <= k <= 3 and abs((xb - xa) / dx - k) < 0.15:
                    for q in range(1, k):
                        x = xa + (xb - xa) * q / k
                        yl = ya_ + (yb_ - ya_) * q / k  # 이웃 표식 사이 직선
                        j = int(np.argmin(np.abs(P[:, 0] - x)))
                        y = float(P[j, 1]) if abs(P[j, 0] - x) <= 2 else yl
                        if abs(y - yl) > 0.5 * abs(yb_ - ya_) + 3:  # 선 추적이 교차점에서 다른 선으로 넘어감 → 이웃 사이 직선
                            y = yl
                        fill.append([x, y])
            pw = [w for w in pt["warn"] if "침식" not in w]
            if fill:
                pw.append(f"다른 선·글자에 가려 안 보이는 표식 {len(fill)}개는 등간격으로 보고 선 위치로 채웠습니다 — 확인하세요")
            mk = sorted(mk + fill)
            X, Y = to_data(calib, [p[0] for p in mk], [p[1] for p in mk])
            out = {"mode": "line", "how": "markers", "px": mk, "data": [[float(x), float(y)] for x, y in zip(X, Y)], "warn": pw,
                   "kind": "line+markers"}
        else:
            d = resample(calib, lpx, "n", n)
            if d:
                X, Y = to_pixel(calib, [q[0] for q in d], [q[1] for q in d])
                px = [[float(x), float(y)] for x, y in zip(X, Y)]
            else:
                px = []
            out = {"mode": "line", "how": "n", "px": px, "data": d, "warn": ln["warn"], "kind": "line"}
            if dens < 0.85 and span > 0.2 * (box["x1"] - box["x0"]):
                warn.append("점선·끊긴 선으로 보입니다 — 빈 곳은 보간했습니다")
    if m.size and m.mean() > 0.25:
        warn.append("이 색이 그림의 넓은 면적을 차지합니다 — 배경·채움 색일 수 있습니다")
    out["warn"] = out["warn"] + warn
    return out


def right_labels(a, ax):
    """오른쪽 y축(둘째 축) 눈금 글자 덩어리: 그림 영역 오른쪽 테두리 바깥 첫 글기둥 → {"y": 중심들(위→아래), "boxes"}"""
    H, W = a.shape[:2]
    dark = gray(a) < 160
    box = ax["box"]
    x1 = int(round(box["x1"]))
    lo = min(W, x1 + 2)
    # 바깥 눈금(축에 붙은 짧은 가로줄)은 건너뜀: 축에서 이어진 잉크 열
    while lo < min(W, x1 + 14) and dark[int(box["y0"]):int(box["y1"]), lo].mean() > 0.02 and dark[int(box["y0"]):int(box["y1"]), lo].sum() < 0.2 * (box["y1"] - box["y0"]):
        lo += 1
    hi = min(W, x1 + int(0.3 * W))
    r0, r1 = max(0, int(box["y0"]) - int(0.05 * H)), min(H, int(box["y1"]) + int(0.05 * H))
    band = dark[r0:r1, lo:hi]
    if not band.size:
        return {"y": [], "boxes": []}
    inb = band[max(0, int(box["y0"]) + 3 - r0):max(1, int(box["y1"]) - 2 - r0)]
    inb = inb[inb.mean(axis=1) < 0.6] if inb.size else inb
    cr = merge(runs((inb if inb.size else band).any(axis=0)), 3)
    if not cr:
        return {"y": [], "boxes": []}
    s_, e_ = cr[0]  # 축에 가장 가까운 글기둥
    rows = band[:, s_:e_ + 1].any(axis=1)
    bl = merge(runs(rows), 1)
    return {"y": [r0 + (bs + be) / 2 + 0.5 for bs, be in bl], "boxes": [[lo + s_, r0 + bs, lo + e_ + 1, r0 + be + 1] for bs, be in bl]}


def _kmeans(X, k, it=30):
    """작은 k-평균 (가장 먼 점으로 시작)"""
    c = [X[0]]
    for _ in range(1, k):
        d = np.min([((X - q) ** 2).sum(axis=1) for q in c], axis=0)
        c.append(X[int(np.argmax(d))])
    c = np.array(c, float)
    lab = np.zeros(len(X), int)
    for _ in range(it):
        lab = np.argmin([((X - q) ** 2).sum(axis=1) for q in c], axis=0)
        nc = np.array([X[lab == j].mean(axis=0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    return lab


def marker_groups(a, calib, box, color, k, legends=None, tol=60):
    """같은 색(흑백 그림)에서 표식 모양(○□△◇, 속 빈/찬)이 다른 계열 k 개를 가른다.
    선은 구멍 메우기 → 침식으로 지우고 남은 표식의 모양값(채움·속 빔·위아래 치우침·모서리)으로 k-평균.
    → {"groups": [[픽셀 점...] × k], "legend": [(그룹 번호, 견본 y) ...], "shape": [모양 이름 × k]} 또는 None"""
    try:
        from scipy import ndimage
    except ImportError:
        return None
    col = hex2rgb(color) if isinstance(color, str) else color
    neutral = max(col) - min(col) < 40 and max(col) < 120
    m, off = mask_for(a, col, 300 if neutral else tol + 30, box, legends or [])  # 표식 테두리의 안티앨리어싱까지 (속 빈 표식이 닫히게)
    if neutral:
        x0_, y0_ = off
        sub_ = a[y0_:y0_ + m.shape[0], x0_:x0_ + m.shape[1]].astype(np.int16)
        m &= ((sub_.max(axis=2) - sub_.min(axis=2)) < 60) & (gray(sub_) < 190)
    if not m.any():
        return None
    f = ndimage.binary_fill_holes(m)
    comps, r, op = [], 0, f
    for r in range(1, 6):  # 열림(침식→팽창): 가는 선은 사라지고 표식은 제 모양으로 남음
        st = np.ones((2 * r + 1, 2 * r + 1), bool)
        op = ndimage.binary_opening(f, structure=st)
        comps = [c for c in _components(op) if c[0] >= 4]
        if comps and not any(_linelike(c) and c[0] > 12 for c in comps):
            break
    comps = [c for c in comps if not _linelike(c)]
    if len(comps) < 2 * k:
        return None
    med = float(np.median([c[0] for c in comps]))
    comps = [c for c in comps if 0.3 * med <= c[0] <= 2.5 * med]
    size = float(np.median([max(c[3], c[4]) for c in comps]))
    x0, y0 = off
    rgb = a[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
    rg_ = rgb.astype(np.int16)
    # 글자만: 표식(열림 결과)과 긴 선을 뺀 작은 조각 — 흑백 그림은 선·표식·글자가 다 같은 색이라 '같은 색 이웃 = 글자' 규칙이 표식을 지움
    rest = m & ~ndimage.binary_dilation(op, iterations=2)
    lab_r, n_r = ndimage.label(rest, structure=np.ones((3, 3)))
    textm = np.zeros_like(m)
    for j, sl in enumerate(ndimage.find_objects(lab_r)):
        if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) <= 2 * size:
            piece = lab_r[sl] == j + 1
            if piece.sum() / max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= 2.0:  # 가는 획(파선 조각)은 글자로 안 봄
                textm[sl] |= piece
    dark = textm | (((gray(rgb) < 160) & ((rg_.max(axis=2) - rg_.min(axis=2)) < 80)) & ~m)
    inkm = (gray(rgb) < 200) | ((rg_.max(axis=2) - rg_.min(axis=2)) > 60)
    why = {}
    drop = _text_filter(comps, dark, size, inkm, np.zeros_like(m), why)  # 글자 이웃 규칙은 끔 (선·파선 조각이 표식 옆에 붙음) — 글자는 아래 무리·추세 거르기로
    feats, keep = [], []
    rr = r + 1
    for i, (n, cx, cy, w, h, bx, by) in enumerate(comps):
        if i in drop and why.get(i) != "legend":
            continue
        fill = n / max(1.0, w * h)  # 열림 결과로 무리 나누기: ○ 0.8, □ 1.0 (작은 △ 는 뭉개지지만 속이 차 있어 따로 모임)
        # 위아래 치우침은 열림 전(구멍 메운) 표식으로 — △ 판정용
        y0_, x0_ = max(0, by - rr), max(0, bx - rr)
        win = op[y0_:by + h + rr, x0_:bx + w + rr]
        near_ = ndimage.binary_dilation(win, iterations=rr) & f[y0_:by + h + rr, x0_:bx + w + rr]
        if near_.any():
            ys_, xs_ = np.nonzero(near_)
            reg = near_[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
        else:
            reg = op[by:by + h, bx:bx + w]
        hr = reg.shape[0]
        top, bot = reg[: hr // 2].sum(), reg[(hr + 1) // 2:].sum()
        asym = (bot - top) / max(1.0, top + bot)  # △ 아래가 무거움
        ix, iy = int(round(cx)), int(round(cy))
        q = max(1, min(w, h) // 5)
        center = m[max(0, iy - q):iy + q + 1, max(0, ix - q):ix + q + 1].mean()  # 속 빈 표식은 가운데가 빔
        feats.append([fill * 2, asym * 2, center, w / max(1, h)])
        keep.append((i, cx + x0 + 0.5, cy + y0 + 0.5))
    if len(keep) < 2 * k:
        return None
    X = np.asarray(feats, float)  # 값 범위가 비슷해서 그대로 (표준화하면 잡음이 큰 쪽으로 쏠림)
    alive = np.ones(len(X), bool)
    for _ in range(4):  # 3개 미만인 무리(글자 조각·튀는 점)는 빼고 다시
        sub = np.flatnonzero(alive)
        if len(sub) < 2 * k:
            return None
        best = None
        for seed in range(6):
            order = np.random.default_rng(seed).permutation(len(sub))
            lb = _kmeans(X[sub][order], k)
            inert = sum(((X[sub][order][lb == j] - X[sub][order][lb == j].mean(axis=0)) ** 2).sum() for j in range(k) if (lb == j).any())
            if best is None or inert < best[0]:
                full = np.empty(len(sub), int)
                full[order] = lb
                best = (inert, full)
        lab = np.full(len(X), -1)
        lab[sub] = best[1]
        small = [j for j in range(k) if 0 < (lab == j).sum() < max(3, 0.35 * len(sub) / k)]
        if not small:
            break
        alive &= ~np.isin(lab, small)
    groups = [[] for _ in range(k)]
    leg = []
    for (i, px_, py_), g in zip(keep, lab):
        if g < 0:
            continue
        if why.get(i) == "legend":
            leg.append((int(g), py_))
        else:
            groups[g].append([px_, py_])
    for gi, g_ in enumerate(groups):  # 무리 추세에서 크게 벗어난 점(범례 견본·글자)은 뺌
        if len(g_) >= 5:
            g_.sort()
            Y_ = np.array([p_[1] for p_ in g_])
            keep_g = []
            for j_, p_ in enumerate(g_):
                nb = np.r_[Y_[max(0, j_ - 2):j_], Y_[j_ + 1:j_ + 3]]
                if not len(nb) or abs(p_[1] - np.median(nb)) <= max(4 * size, 3 * np.median(np.abs(np.diff(Y_))) + size):
                    keep_g.append(p_)
            groups[gi] = keep_g
    F = np.asarray(feats, float)
    shape = []
    for g in range(k):
        fm = np.median(F[lab == g], axis=0) if (lab == g).any() else np.zeros(4)
        nm = "triangle" if abs(fm[1]) > 0.3 else ("diamond" if fm[0] < 1.3 else "square" if fm[0] > 1.85 else "circle")
        shape.append(("hollow " if fm[2] < 0.5 else "") + nm)
    return {"groups": [sorted(g_) for g_ in groups], "legend": sorted(leg, key=lambda t: t[1]), "shape": shape}
