"""선 모양(실선·파선·점선·일점쇄선) 판정과 같은 색 계열을 선 모양으로 가르기 (2026-10-20).

1. classify(m, px): 선을 따라 1px 간격으로 '칠해짐/빔'을 재고, 빈 틈이 고르게 되풀이되면 파선 계열로 본다.
   다른 선에 가린 틈(드물고 불규칙)은 틈으로 치지 않는다. 칠한 조각 길이로 파선·점선, 긴 조각과 짧은 조각이 번갈아 나오면 일점쇄선.
2. separate(a, box, color, tol, legends): 한 색 마스크에서 표식(속 찬 점·빈 원)을 떼고, 남은 선 조각을 긴 조각(실선)과
   짧고 고른 조각(파선·점선)으로 나눠 따로 추적한다. 짧은 조각이 실선 경로 위에 있으면(다른 선이 가로질러 잘린 실선) 실선에 둔다.
   표식은 채움(속 찬 점 → 실선, 빈 원 → 파선)이 갈리면 그 짝으로, 아니면 가까운 경로로 나눈다.
   → [{style, px(경로 픽셀 점), markers([[x, y, filled]...]), n_pieces}] (모양이 하나뿐이면 None)
3. marker_points(groups): 표식이 있는 경로는 표식 중심이 데이터 점. 다른 경로의 표식 x 에 이 경로의 표식이 없으면
   (같은 자리의 다른 표식 밑에 가림) 경로 높이로 채우고 '겹침(shared)' 으로 표시한다.
그림 이름·좌표에 기대는 규칙은 없다. scipy 가 없으면 아무것도 하지 않는다(예전 동작).
"""
import math

import numpy as np

import digitize as D

STYLE_KO = {"solid": "실선", "dashed": "파선", "dotted": "점선", "dashdot": "일점쇄선"}


def norm_style(s):
    s = str(s or "").lower().replace(" ", "")
    if not s:
        return None
    if "dashdot" in s or "dash-dot" in s or "일점" in s or "-." in s:
        return "dashdot"
    if "dot" in s or "점선" in s or s == ":":
        return "dotted"
    if "dash" in s or "파선" in s or s == "--":
        return "dashed"
    if s in ("line", "solid", "실선", "-"):
        return "solid"
    return None


def line_width(m):
    """마스크 선 두께(px): 덩어리가 하나뿐인 열의 세로 길이 중앙값 (가파른 곳은 길게 나오므로 아래쪽 사분위)"""
    lens = []
    step = max(1, m.shape[1] // 400)
    for c in range(0, m.shape[1], step):
        rs = D.runs(m[:, c])
        if len(rs) == 1:
            lens.append(rs[0][1] - rs[0][0] + 1)
    return max(1.5, float(np.median(lens))) if lens else 2.0


def _dense(px):
    """경로 점(열마다) → 1px 호 길이 간격 점들 (틈도 직선으로 지나감)"""
    P = np.asarray(sorted(px), float)
    if len(P) < 2:
        return P
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(math.ceil(d)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)


def onoff(m, off, px):
    """경로를 따라 칠해짐 여부 (True/False 배열)"""
    Q = _dense(px)
    if len(Q) < 2:
        return np.zeros(0, bool)
    h, w = m.shape
    on = np.zeros(len(Q), bool)
    d = np.diff(Q, axis=0)
    steep = np.r_[np.abs(d[:, 1]) > np.abs(d[:, 0]), False]
    for i, (x, y) in enumerate(Q):
        r, c = int(y - off[1]), int(x - off[0])
        if not (0 <= r < h and 0 <= c < w):
            continue
        if steep[i]:
            on[i] = m[r, max(0, c - 1):c + 2].any()
        else:
            on[i] = m[max(0, r - 1):r + 2, c].any()
    return on


def _runs(b):
    """불리언 배열 → (값, 길이) 연속 구간"""
    if not len(b):
        return []
    idx = np.flatnonzero(np.diff(b.astype(np.int8))) + 1
    st = np.r_[0, idx]
    en = np.r_[idx, len(b)]
    return [(bool(b[s]), int(e - s)) for s, e in zip(st, en)]


def classify(m, off, px, lw=None):
    """경로의 선 모양 → 'solid' | 'dashed' | 'dotted' | 'dashdot' (판단 못 하면 'solid')
    마스크를 1px 넓혀 잰다 — 가는 선의 안티앨리어싱으로 생긴 1~2px 틈은 틈이 아님"""
    lw = lw or line_width(m)
    on = onoff(D._shift_and(m, 1, "or"), off, px)
    if len(on) < 30:
        return "solid"
    rs = _runs(on)
    while rs and not rs[0][0]:
        rs.pop(0)
    while rs and not rs[-1][0]:
        rs.pop()
    gaps = [n for v, n in rs if not v]
    ons = [n for v, n in rs if v]
    # 고른 틈이 촘촘해야 파선 (100px 에 2.5개 이상) — 여러 선이 가로지르는 묶음의 실선은 틈이 드묾
    if len(gaps) < max(5, 0.025 * len(on)) or sum(gaps) < 0.08 * len(on):
        return "solid"
    g = np.asarray(gaps, float)
    gm = float(np.median(g))
    reg = g[g <= 3 * gm + 2]  # 다른 선에 가린 큰 틈은 뺌
    if len(reg) < max(5, 0.025 * len(on)) or len(reg) < 0.6 * len(g) or float(np.std(reg)) > 0.6 * float(np.mean(reg)) + 0.5:
        return "solid"
    o = np.asarray(ons, float)
    if len(o) < 5:
        return "solid"
    # 일점쇄선: 긴 조각과 짧은 조각이 번갈아
    lo, hi = np.percentile(o, 20), np.percentile(o, 80)
    if hi >= 2.5 * max(lo, 1) and hi >= 3 * lw and lo <= max(2 * lw + 3, 6):  # 짧은 쪽은 점(선 두께 남짓)
        thr = math.sqrt(max(lo, 1) * hi)
        big = o >= thr
        alt = np.mean(big[1:] != big[:-1]) if len(big) > 2 else 0
        if 0.25 <= big.mean() <= 0.75 and alt >= 0.6:
            return "dashdot"
    om = float(np.median(o))
    if om <= max(3.5 * lw, 6) and om <= 3.5 * gm + 1:  # (1px 넓힌 마스크: 칠한 조각 +2, 틈 -2)
        return "dotted"
    return "dashed"


def _disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r + 0.5


def find_markers(m, lw):
    """선보다 굵은 둥근 덩어리(속 찬 점·빈 원) → [{x, y, d, filled, sl, k}] (로컬 좌표). 3개 미만이면 []"""
    from scipy import ndimage
    # 빈 원 테두리가 안티앨리어싱으로 한두 px 끊기면 구멍이 메워지지 않음 → 3x3 닫힘 뒤 메움 (원래 마스크와 겹친 곳만이 아니라 원판 전체)
    f = ndimage.binary_fill_holes(ndimage.binary_closing(m, structure=np.ones((3, 3)), border_value=0) | m)
    r = max(1, int(round(lw / 2 + 1)))
    op = ndimage.binary_opening(f, structure=_disk(r))
    lab, n = ndimage.label(op, structure=np.ones((3, 3)))
    out = []
    for j, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        k = lab[sl] == j + 1
        h, w = k.shape
        if k.sum() < 6 or max(h, w) < max(4, 2 * lw + 1) or max(h, w) > 6 * lw + 12 or max(h, w) > 1.6 * min(h, w) + 1:
            continue
        ci, cj = sl[0].start + (h - 1) / 2, sl[1].start + (w - 1) / 2
        a_, b_ = int(round(ci)), int(round(cj))
        core = m[max(0, a_ - 1):a_ + 2, max(0, b_ - 1):b_ + 2]
        out.append({"x": cj + 0.5, "y": ci + 0.5, "d": max(h, w), "filled": bool(core.mean() > 0.5), "sl": sl, "k": k})
    if len(out) < 4:
        return []
    md = float(np.median([o["d"] for o in out]))
    return [o for o in out if 0.6 * md <= o["d"] <= 1.6 * md]


def _comps(L):
    from scipy import ndimage
    lab, n = ndimage.label(L, structure=np.ones((3, 3)))
    out = []
    for j, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        k = lab[sl] == j + 1
        out.append({"sl": sl, "k": k, "n": int(k.sum()), "w": sl[1].stop - sl[1].start, "h": sl[0].stop - sl[0].start,
                    "cx": (sl[1].start + sl[1].stop) / 2, "cy": float(np.mean(np.nonzero(k)[0])) + sl[0].start})
    return out


def _paint(shape, comps):
    M = np.zeros(shape, bool)
    for c in comps:
        M[c["sl"]] |= c["k"]
    return M


def _yat(px, x):
    P = np.asarray(px, float)
    if not len(P) or x < P[0, 0] - 1 or x > P[-1, 0] + 1:
        return None
    return float(np.interp(x, P[:, 0], P[:, 1]))


def _trace(M, off):
    px = D.extract_line(M, off) if M.any() else []
    return sorted(px)


def lattice(xs, min_pts=4):
    """표식 x 들이 고른 간격 격자에 놓였으면 (간격, 시작) — 아니면 None. 격자 밖 표식(두 선이 붙어 굵어진 곳)을 버리는 데 씀"""
    xs = list(xs)
    if len(xs) < min_pts:
        return None
    cl = []
    for x in sorted(xs):
        if cl and x - cl[-1][-1] <= 3:
            cl[-1].append(x)
        else:
            cl.append([x])
    c = [float(np.mean(v)) for v in cl]
    if len(c) < 3:
        return None
    best = None
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            sp = c[j] - c[i]
            if sp < 15:
                continue
            for k in range(1, 4):
                s_ = sp / k
                if s_ < 15:
                    break
                on = [abs((x - c[i]) / s_ - round((x - c[i]) / s_)) * s_ <= max(2.5, 0.08 * s_) for x in xs]
                score = sum(on) - 1.5 * (k - 1)  # 간격을 쪼개 억지로 맞추지 않게
                if best is None or score > best[0] + 1e-9:
                    best = (score, s_, c[i], on)
    if best is None or sum(best[3]) < max(min_pts, 0.6 * len(xs)):
        return None
    span = (max(xs) - min(xs)) / best[1]
    if span < 2.5:
        return None
    return best[1], best[2], best[3]


def separate(a, box, color, tol=60, legends=None, want=0):
    """한 색 안의 선 모양별 경로. want: 범례가 말한 모양 수(0 = 모름). → {paths, lw, markers} (모양 2개 이상) 또는 None"""
    try:
        import scipy.ndimage  # noqa: F401 — 표식·조각 나누기에 필요
    except ImportError:
        return None
    col = D.hex2rgb(color) if isinstance(color, str) else tuple(color)
    m, off = D.mask_for(a, col, tol, box, legends or [])
    ch = max(col) - min(col)
    if ch > 50:  # 색 계열: 회색 선 가장자리 번짐(채도 낮음)이 이 색으로 잡히지 않게
        sub = a[off[1]:off[1] + m.shape[0], off[0]:off[0] + m.shape[1]].astype(np.int16)
        m &= (sub.max(axis=2) - sub.min(axis=2)) >= 0.45 * ch
    if m.sum() < 40:
        return None
    lw = line_width(m)
    mk = find_markers(m, lw)
    lt = lattice([o["x"] for o in mk]) if mk else None
    if lt:  # 고른 x 격자 밖의 덩어리(같은 색 두 선이 붙어 굵어진 곳)는 표식이 아님
        mk = [o for o, on in zip(mk, lt[2]) if on]
    out = _paths(m, lw, mk, a[off[1]:off[1] + m.shape[0], off[0]:off[0] + m.shape[1]])
    if out is not None:
        W = m.shape[1]
        span = lambda p: p["px"][-1][0] - p["px"][0][0]  # noqa: E731
        if want >= 2 and len(out) > want:  # 범례가 말한 수만큼 — 길고 조각 많은 경로부터
            out = sorted(out, key=lambda p: -span(p) * (1 + min(p["n_pieces"], 40)))[:want]
        elif want <= 1:  # 범례 한 줄을 모양으로 늘릴 때는 짧은 파선 조각 경로는 버림
            out = [p for p in out if p["style"] == "solid" or span(p) >= 0.3 * W]
            if len(out) < 2:
                out = None
    if out is None:
        out = _marker_only(mk)
        if out is None:
            return None
    else:
        _assign(out, mk, (0, 0))
    for p in out:
        p["px"] = [[x + off[0], y + off[1]] for x, y in p["px"]]
        p["markers"] = sorted([[x + off[0], y + off[1], f] for x, y, f in p["markers"]])
    return {"paths": out, "lw": lw, "markers": len(mk), "lattice": (lt[0], lt[1] + off[0]) if lt else None}


def _marker_only(mk):
    """선 경로는 못 갈랐지만 표식이 속 찬 것·빈 것으로 갈림 → 표식 무리 둘(속 찬 = 실선, 빈 = 파선으로 가정)"""
    if len(mk) < 4:
        return None
    fi = [o for o in mk if o["filled"]]
    ho = [o for o in mk if not o["filled"]]
    if len(fi) < 2 or len(ho) < 2 or len(mk) < 5:
        return None
    # 두 무리가 같은 x 에서 다른 높이로 함께 있어야 (두 계열) — 채움 판정이 흔들리는 한 계열이 아님
    both = sum(1 for o in fi if any(abs(o["x"] - q["x"]) <= 3 and abs(o["y"] - q["y"]) > o["d"] for q in ho))
    if both < 1:
        return None
    for g in (fi, ho):  # 한 무리에 같은 x 의 표식이 둘이면 한 계열이 아님
        xs = sorted(o["x"] for o in g)
        if any(b - a_ <= 3 for a_, b in zip(xs, xs[1:])):
            return None
    out = []
    for st, g in (("solid", fi), ("dashed", ho)):
        P = sorted([[o["x"], o["y"]] for o in g])
        out.append({"style": st, "style_guess": True, "px": P, "n_pieces": 0, "markers": [[x, y, st == "solid"] for x, y in P], "marker_only": True})
    return out


def _near(px, x, gap=4.0):
    """경로에서 x 의 높이 — 경로 점이 x 가까이(gap px) 없으면 None (틈을 건너 보간하지 않음)"""
    P = np.asarray(px, float)
    if not len(P):
        return None
    j = int(np.searchsorted(P[:, 0], x))
    best = None
    for k in (j - 1, j):
        if 0 <= k < len(P) and abs(P[k, 0] - x) <= gap:
            if best is None or abs(P[k, 0] - x) < abs(P[best, 0] - x):
                best = k
    if best is None:
        return None
    return _yat(px, x)


def _ends(c):
    """조각의 왼쪽·오른쪽 끝 (x, y) — 로컬 좌표"""
    ys, xs = np.nonzero(c["k"])
    xs = xs + c["sl"][1].start
    ys = ys + c["sl"][0].start
    l, r = xs.min(), xs.max()
    return (float(l), float(ys[xs == l].mean())), (float(r), float(ys[xs == r].mean()))


def _chains(comps, lw, W, min_n=6):
    """짧은 조각(대시·점)을 왼쪽부터 방향이 이어지게 잇는다 → 조각 사슬 목록.
    다음 조각은 사슬 끝 몇 조각으로 잰 기울기로 내다본 높이에 가장 가까운 것 (교차점에서 다른 선의 대시로 넘어가지 않게)."""
    for c in comps:
        c["L"], c["R"] = _ends(c)
    G = 6 * lw + 10
    unused = sorted(comps, key=lambda c: c["L"][0])
    chains = []
    while unused:
        ch = [unused.pop(0)]
        while True:
            pts = [q for c in ch[-5:] for q in (c["L"], c["R"])]
            X = np.array([q[0] for q in pts])
            Y = np.array([q[1] for q in pts])
            k = float(np.polyfit(X, Y, 1)[0]) if X.max() - X.min() >= 4 else 0.0
            ex, ey = ch[-1]["R"]
            best = None
            for c in unused:
                dx = c["L"][0] - ex
                if dx > G:
                    break
                if dx < -2:
                    continue
                yp = ey + k * dx
                dy = abs(c["L"][1] - yp)
                if dy > 2 + lw + 0.35 * abs(dx) + 0.3 * abs(k) * abs(dx):
                    continue
                sc = dy + 0.1 * abs(dx)
                if c["R"][0] - c["L"][0] >= 4 and X.max() - X.min() >= 8:  # 대시 자체의 기울기가 사슬 방향과 맞아야 (교차점에서 다른 선 대시로 넘어가지 않게)
                    kc = (c["R"][1] - c["L"][1]) / (c["R"][0] - c["L"][0])
                    sc += 6.0 * abs(math.atan(kc) - math.atan(k))
                if best is None or sc < best[0]:
                    best = (sc, c)
            if best is None:
                break
            ch.append(best[1])
            unused.remove(best[1])
        chains.append(ch)
    # 다른 선에 가려(또는 교차점에서) 끊긴 사슬끼리 잇기 — 모든 (끝 → 시작) 짝을 내다본 높이 오차 순으로 (가장 잘 맞는 짝부터)
    def tail(A):
        pts = [q for c in A[-5:] for q in (c["L"], c["R"])]
        X = np.array([q[0] for q in pts])
        k = float(np.polyfit(X, [q[1] for q in pts], 1)[0]) if X.max() - X.min() >= 4 else 0.0
        return A[-1]["R"][0], A[-1]["R"][1], k

    def head(B_):
        pts = [q for c in B_[:5] for q in (c["L"], c["R"])]
        X = np.array([q[0] for q in pts])
        k = float(np.polyfit(X, [q[1] for q in pts], 1)[0]) if X.max() - X.min() >= 4 else 0.0
        return B_[0]["L"][0], B_[0]["L"][1], k
    while True:
        cand = []
        for i, A in enumerate(chains):
            if len(A) < 3:
                continue
            ex, ey, ka = tail(A)
            for j, B_ in enumerate(chains):
                if j == i or len(B_) < 3:
                    continue
                sx, sy, kb = head(B_)
                dx = sx - ex
                if not (0 < dx <= 0.12 * W):
                    continue
                e = abs(sy - (ey + ka * dx))
                e2 = abs(ey - (sy - kb * dx))  # 뒤 사슬 머리에서 거꾸로 내다본 높이도
                lim = 4 + 1.5 * lw + 0.3 * dx
                if e <= lim and e2 <= lim:
                    cand.append(((e + e2) / lim + 0.5 * abs(math.atan(ka) - math.atan(kb)), i, j))
        if not cand:
            break
        cand.sort()
        _, i, j = cand[0]
        chains[i] = chains[i] + chains[j]
        del chains[j]
    out = []
    for ch in chains:
        xs = [c["cx"] for c in ch]
        if len(ch) >= min_n and max(xs) - min(xs) >= 0.15 * W:
            out.append(ch)
    return out


def _paths(m, lw, mk, sub=None):
    """마스크(표식 뺌) → 선 모양별 경로 [{style, px, n_pieces}] (2개 이상일 때만)"""
    from scipy import ndimage
    H, W = m.shape
    L = m.copy()
    if mk:
        mm = np.zeros_like(m)
        for o in mk:
            mm[o["sl"]] |= o["k"]
        L &= ~ndimage.binary_dilation(mm, iterations=2)
    cs = [c for c in _comps(L) if c["n"] >= 2]
    if len(cs) < 6:
        return None
    ell = lambda c: max(c["w"], c["h"], c["n"] / max(lw, 1.0))  # noqa: E731
    lmax = 7 * lw + 4
    long_ = [c for c in cs if ell(c) > lmax]
    short = [c for c in cs if ell(c) <= lmax]
    if len(short) < 6:
        return None
    tolp = max(2.5, 1.2 * lw + 1.5)
    sol = _trace(_paint(m.shape, long_), (0, 0)) if long_ else []
    if len(sol) < 0.15 * W:
        sol = []
        short = short + long_
        long_ = []
    # 실선 경로 위의 짧은 조각(가로지른 다른 선에 잘린 실선·표식 가장자리)은 실선에 둠
    off_ = []
    S_ = np.asarray(sol, float) if sol else None
    for c in short:
        on = False
        if S_ is not None and _near(sol, c["cx"], 0.5 * c["w"] + 3) is not None:
            # 조각 픽셀이 실선 경로 위에 놓였나 (픽셀마다 같은 열의 경로 높이와의 거리 중앙값) — 나란히 몇 px 떨어진 파선 대시는 실선이 아님
            ys_, xs_ = np.nonzero(c["k"])
            xs_ = xs_ + c["sl"][1].start + 0.5
            ys_ = ys_ + c["sl"][0].start + 0.5
            ok_ = (xs_ >= S_[0, 0] - 2) & (xs_ <= S_[-1, 0] + 2)
            if ok_.any():
                dd = np.abs(ys_[ok_] - np.interp(xs_[ok_], S_[:, 0], S_[:, 1]))
                on = float(np.median(dd)) <= 0.5 * lw + 1.2
        if on:
            long_.append(c)
        else:
            off_.append(c)
    if len(off_) < 6:
        return None
    lens = np.asarray([ell(c) for c in off_], float)
    groups = [off_]
    q25, q75 = np.percentile(lens, 25), np.percentile(lens, 75)
    if q75 >= 2.5 * max(q25, 1.0):  # 짧은 조각이 두 무리(점선 점 + 파선 대시)
        thr = math.sqrt(max(q25, 1.0) * q75)
        g1 = [c for c in off_ if ell(c) < thr]
        g2 = [c for c in off_ if ell(c) >= thr]
        if len(g1) >= 6 and len(g2) >= 6:
            groups = [g1, g2]
    paths = []
    if long_:
        paths.append({"style": "solid", "comps": long_})
    for g in groups:
        for ch in _chains(g, lw, W):
            paths.append({"style": None, "comps": ch})
    # 사슬에 못 든 짧은 조각(일점쇄선의 점·교차점에서 잘린 대시)이 어떤 파선 경로 위에 있으면 그 경로에 붙임
    inch = {id(c) for p in paths for c in p["comps"]}
    left = [c for g in groups for c in g if id(c) not in inch]
    for p in paths:
        p["M"] = _paint(m.shape, p["comps"])
        p["px"] = _trace(p["M"], (0, 0))
    for c in left:
        best = None
        for p in paths:
            if p["style"] == "solid" or not p["px"]:
                continue
            y = _near(p["px"], c["cx"], 0.5 * c["w"] + 8)
            if y is not None and abs(y - c["cy"]) <= tolp and (best is None or abs(y - c["cy"]) < best[0]):
                best = (abs(y - c["cy"]), p)
        if best:
            best[1]["comps"].append(c)
            best[1]["grew"] = True
    for p in paths:
        if p.get("grew"):
            p["M"] = _paint(m.shape, p["comps"])
            p["px"] = _trace(p["M"], (0, 0))
    # 다른 무리(긴 대시·짧은 점)의 경로가 같은 길이면 일점쇄선 하나로 합침
    merged = True
    while merged:
        merged = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                A, B_ = paths[i], paths[j]
                if A["style"] == "solid" or B_["style"] == "solid":
                    continue
                S_, T_ = (A, B_) if len(A["px"]) <= len(B_["px"]) else (B_, A)  # 짧은 경로의 점이 긴 경로 위에 있으면 같은 선
                d = [abs(y - q) for x, y in S_["px"] for q in [_near(T_["px"], x, 8)] if q is not None]
                if len(d) >= 0.5 * len(S_["px"]) and np.median(d) <= 2 * lw + 1:
                    A["comps"] = A["comps"] + B_["comps"]
                    A["M"] = A["M"] | B_["M"]
                    A["px"] = _trace(A["M"], (0, 0))
                    A["merged"] = True
                    del paths[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for p in paths:
        if len(p["px"]) < 8 or p["px"][-1][0] - p["px"][0][0] < 0.15 * W:
            continue
        st = p["style"] or classify(p["M"], (0, 0), p["px"], lw)
        if p["style"] is None and st == "solid":
            continue  # 짧은 조각을 이었는데 틈이 고르지 않음 — 파선이 아니라 다른 선에 잘린 부스러기
        out.append({"style": st, "px": p["px"], "n_pieces": len(p["comps"]), "markers": [], "M": p["M"]})
    sols = [p for p in out if p["style"] == "solid"]
    if sub is not None and sols:  # 짧은 조각의 색이 실선 색과 다르면(다른 진한 선의 가장자리 번짐) 같은 색 계열이 아님
        def mcol(M):
            v = sub[M & m]
            return np.median(v.astype(float), axis=0) if len(v) else None
        c0 = mcol(sols[0]["M"])
        for p in out:
            if p["style"] != "solid" and c0 is not None:
                c1 = mcol(p["M"])
                p["dcol"] = float(np.sqrt(((c1 - c0) ** 2).sum())) if c1 is not None else 0.0
                if p["dcol"] > 30:
                    p["n_pieces"] = 0
    if sub is not None:  # 다른(더 진한·다른 색) 선 바로 옆을 따라가는 경로 = 그 선의 번짐(복사·스캔 그림의 테두리)
        sf = sub.astype(np.int16)
        base_c = np.median(sf[m].reshape(-1, 3), axis=0) if m.any() else None
        for p in out:
            if p["style"] == "solid" or base_c is None:
                continue
            P = p["px"][::2]
            near = 0
            for x, y in P:
                r, c = int(y), int(x)
                win = sf[max(0, r - int(lw) - 3):r + int(lw) + 4, max(0, c - 1):c + 2].reshape(-1, 3)
                v = win - base_c
                dw = 255 - base_c
                t = np.clip((v @ dw) / float(dw @ dw + 1e-9), 0, 1)
                perp = np.sqrt(((v - t[:, None] * dw) ** 2).sum(axis=1))  # 이 색과 흰 바탕 사이(번짐)에서 벗어난 정도
                ink = (win.max(axis=1) < 235) & (perp > 30)
                near += bool(ink.sum() >= 2)
            if P and near > 0.5 * len(P):
                p["n_pieces"] = 0
    keep = []
    for p in out:  # 오탐 막기: 원래 마스크 그대로도 그 경로에 고른 틈이 있어야 하고, 실선 경로와 떨어져 있어야 함
        if p["style"] != "solid":
            if p["n_pieces"] < 8 or classify(m, (0, 0), p["px"], lw) == "solid":
                continue
            if sols:
                d = [abs(y - q) for x, y in p["px"] for q in [_near(sols[0]["px"], x, 3)] if q is not None]
                if d and float(np.median(d)) < 2 * lw + 2:
                    continue
        keep.append(p)
    out = keep
    if len(out) < 2:
        return None
    for p in out:  # 파선 경로는 실선과 같은 x 에 함께 있어야 함 (실선이 가려진 틈을 메우는 조각이면 같은 선)
        if p["style"] == "solid" or not sols:
            continue
        both = sum(1 for x, _ in p["px"][::2] if _near(sols[0]["px"], x, 3) is not None)
        if both < 0.5 * len(p["px"][::2]):
            return None
    for i in range(len(out)):  # 서로 다른 경로여야 함 (같은 경로를 둘로 나눈 것이 아님)
        for j in range(i + 1, len(out)):
            A, B_ = out[i]["px"], out[j]["px"]
            d = [abs(y - q) for x, y in A for q in [_near(B_, x)] if q is not None]
            if d and np.mean(np.asarray(d) <= tolp) > 0.8:
                return None
    for p in out:
        p.pop("M", None)
    return out


def _assign(out, mk, off):
    """표식 → 경로: 채움이 갈리면(속 찬 점·빈 원 둘 다 2개 이상, 경로 둘) 속 찬 = 실선, 빈 = 다른 경로. 아니면 가까운 경로"""
    if not mk:
        return
    fill = [o["filled"] for o in mk]
    by_fill = 2 <= sum(fill) <= len(fill) - 2 and len(out) == 2 and sum(p["style"] == "solid" for p in out) == 1
    for o in mk:
        X, Y = o["x"], o["y"]
        if by_fill:
            k = 0 if o["filled"] == (out[0]["style"] == "solid") else 1
        else:
            ds = []
            for p in out:
                yy = _yat(p["px"], X)
                if yy is None and p["px"]:
                    e = min((p["px"][0], p["px"][-1]), key=lambda q: abs(q[0] - X))
                    yy = e[1] if abs(e[0] - X) <= o["d"] + 3 else None
                ds.append(abs(yy - Y) if yy is not None else 1e9)
            k = int(np.argmin(ds))
            if ds[k] > 3 * o["d"]:
                continue
        out[k]["markers"].append([X, Y, o["filled"]])


def _grid(xs):
    cl = []
    for x in sorted(xs):
        if cl and x - cl[-1][-1] <= 3:
            cl[-1].append(x)
        else:
            cl.append([x])
    return [float(np.mean(c)) for c in cl]


def _on_path(P, x, reach=10):
    """경로 P(정렬된 [[x, y]]) 의 x 높이 — 끝 너머 reach px 까지는 끝 기울기로 늘림"""
    if not len(P):
        return None
    j = int(np.argmin(np.abs(P[:, 0] - x)))
    dx = abs(P[j, 0] - x)
    if dx <= 2:
        return float(P[j, 1])
    if P[0, 0] < x < P[-1, 0]:
        return float(np.interp(x, P[:, 0], P[:, 1])) if dx <= 16 else None  # 표식 자리(표식을 떼어 낸 틈)
    if dx <= reach and len(P) >= 3:  # 경로 끝이 가린 표식 가장자리에서 시작 — 끝 기울기로 몇 px 늘림
        side = P[:3] if x < P[0, 0] else P[-3:]
        if side[-1, 0] - side[0, 0] < 1:
            return float(side[:, 1].mean())
        k = np.polyfit(side[:, 0], side[:, 1], 1)
        return float(np.polyval(k, x))
    return None


def marker_points(paths, n=50, calib=None, grid=None):
    """경로별 데이터 점: 표식이 있으면 표식 중심 + 표식 격자 x 에서 표식이 안 보이면 경로 높이
    (그 자리에 다른 경로의 표식이 있으면 그 밑에 가린 것 = shared). 표식이 없으면 경로를 calib 로 n 점 재표본.
    → 경로마다 (px, shared x 목록, kind)"""
    gx = list(grid) if grid else _grid([x for p in paths for x, y, _ in p["markers"]])
    res = []
    for p in paths:
        mk = p["markers"]
        if len(mk) >= 3 or (len(mk) >= 2 and len(gx) >= 3):
            pts = [[x, y] for x, y, _ in mk]
            shared = []
            P = np.asarray(p["px"] or [[x, y] for x, y, _ in mk], float)
            P = P[np.argsort(P[:, 0])]
            for x in gx:
                if min(abs(x - q[0]) for q in pts) <= 3:
                    continue
                y = _on_path(P, x)
                hid = [q for o in paths if o is not p for q in o["markers"] if abs(q[0] - x) <= 3]
                if y is None and p.get("marker_only") and hid:  # 선 경로 없이 표식만: 같은 자리 다른 표식 밑에 가림
                    y = hid[0][1]
                if y is None:
                    continue
                near_ = [q for q in hid if abs(q[1] - y) <= max(8, 1.5 * p.get("d", 7))]
                if near_:  # 같은 자리의 다른 표식 밑에 가림 → 그 표식 중심 (가파른 선에서 경로 높이보다 정확)
                    y = min(near_, key=lambda q: abs(q[1] - y))[1]
                    shared.append(x)
                pts.append([x, y])
            res.append((sorted(pts), shared, "line+markers"))
        else:
            if calib is not None and len(p["px"]) >= 2:
                d = D.resample(calib, p["px"], "n", n)
                X, Y = D.to_pixel(calib, [q[0] for q in d], [q[1] for q in d]) if d else ([], [])
                res.append(([[float(x), float(y)] for x, y in zip(X, Y)], [], "line"))
            else:
                res.append((p["px"], [], "line"))
    return res


def grid_pass(series, a, box):
    """표식 그림(계열 대부분이 같은 x 격자에 표식)에서 표식을 못 찾은 선 계열(다른 선·표식에 가린 표식)은
    격자 x 마다 그 색 덩어리가 있으면 선 높이를 데이터 점으로 쓴다."""
    mks = [s for s in series if s.get("kind") == "line+markers" and s.get("_path")]
    xs = [x for s in mks for x, y, _ in s["_path"]["markers"]]
    if len(mks) < 2 or not xs:
        return series
    lt = lattice(xs, min_pts=4)
    if not lt:
        return series
    gx = [x for x in _grid(xs) if any(abs(x - q) <= 3 for q, o in zip(xs, lt[2]) if o)]
    if len(gx) < 3:
        return series
    for s in series:
        p = s.get("_path")
        if not p or s.get("kind") != "line" or not s.get("color"):
            continue
        P = np.asarray(p["px"], float)
        if len(P) < 8:
            continue
        P = P[np.argsort(P[:, 0])]
        col = D.hex2rgb(s["color"])
        m0, off = D.mask_for(a, col, 60, box, remove_lines=False)
        # 표식 테두리는 1px 안티앨리어싱이라 색보다 옅음 → 이 색과 흰 바탕 사이(흰색 75% 까지) 픽셀도 표식으로
        sub = a[off[1]:off[1] + m0.shape[0], off[0]:off[0] + m0.shape[1]].astype(np.float32)
        c_ = np.asarray(col, np.float32)
        dw = 255 - c_
        v = sub - c_
        t = (v @ dw) / float(dw @ dw + 1e-9)
        perp = np.sqrt(np.maximum(0, (v * v).sum(axis=2) - t ** 2 * float(dw @ dw)))
        m = m0 | ((t >= 0) & (t <= 0.75) & (perp <= 25))
        d = int(max(5, min(15, 3.5 * line_width(m0)))) if m0.any() else 7  # 표식 크기 어림(선 두께의 3.5배)
        pts, hits, cand = [], 0, 0
        for x in gx:
            y = _on_path(P, x, reach=d + 12)
            if y is None:
                continue
            cand += 1
            r0, c0 = int(y - off[1]), int(x - off[0])
            ra = max(0, r0 - d)
            win = m[ra:r0 + d + 1, max(0, c0 - 1):c0 + 2]  # 표식 가운데 세로 띠: 원·점이면 표식 지름만큼, 선이면 선 두께만큼
            yy = np.nonzero(win.any(axis=1))[0]
            if len(yy):
                # 경로 높이를 지나는 덩어리만
                runs_ = np.split(yy, np.flatnonzero(np.diff(yy) > max(2, d - 3)) + 1)  # 빈 원의 위·아래 테두리는 한 덩어리
                run = min(runs_, key=lambda q: abs((q[0] + q[-1]) / 2 - (r0 - ra)))
                ext = run[-1] - run[0] + 1
                if 0.6 * d <= ext <= 1.6 * d + 2:
                    hits += 1
                    y = float(off[1] + ra + (run[0] + run[-1]) / 2 + 0.5)  # 그 자리 덩어리(빈 원·점) 가운데
            pts.append([x, y])
        poly = False
        if cand >= 3:  # 경로가 격자 x 사이를 곧은 줄로 잇는 꺾은선이면 (표식이 다른 선·표식에 가려도) 표식 꺾은선
            qx = np.array([q[0] for q in pts])
            qy = np.array([_on_path(P, x, reach=d + 12) for x in qx], float)
            inn = (P[:, 0] >= qx.min()) & (P[:, 0] <= qx.max())
            if inn.sum() >= 10 and not np.isnan(qy).any():
                dev = np.abs(P[inn, 1] - np.interp(P[inn, 0], qx, qy))
                poly = float(np.mean(dev <= max(2.0, line_width(m0) + 1))) >= 0.9
        if cand >= 3 and (hits >= 0.6 * cand or poly):
            s["px"] = pts
            s["kind"], s["how"] = "line+markers", "markers"
            s["warn"] = list(s.get("warn") or []) + [f"표식이 다른 선·표식에 가려 잘 안 보여, 다른 계열의 표식 x {len(pts)}곳에서 이 선의 높이를 읽었습니다 — 확인하세요"]
    return series


def style_of(a, box, color, legends=None, tol=60):
    """한 색 선 계열의 선 모양 (가르기 없이)"""
    col = D.hex2rgb(color) if isinstance(color, str) else tuple(color)
    m, off = D.mask_for(a, col, tol, box, legends or [])
    if m.sum() < 20:
        return None
    px = D.extract_line(m, off)
    if len(px) < 10:
        return None
    return classify(m, off, px)


def label(st, style_lab=None):
    return (style_lab or {}).get(st) or STYLE_KO.get(st) or str(st)


def to_series(sp, members, calib, style_lab=None, grid=None):
    """separate() 결과 → 계열 dict 목록. members: 같은 색 범례 계열들(1개면 모양마다 이름 뒤에 ' · 실선' 등을 붙여 늘림).
    범례가 여럿이면 범례가 말한 모양(style)과 같은 경로를 짝짓고, 모자라면 실선부터 차례로."""
    paths = sorted(sp["paths"], key=lambda p: ["solid", "dashed", "dashdot", "dotted"].index(p["style"]) if p["style"] in STYLE_KO else 9)
    pts = marker_points(paths, calib=calib, grid=grid)
    if len(members) == 1:
        m0 = members[0]
        base = m0.get("name") or ""
        cnt = {}
        for p in paths:
            cnt[p["style"]] = cnt.get(p["style"], 0) + 1
        seen = {}
        order = sorted(range(len(paths)), key=lambda k: (paths[k]["style"] or "", float(np.mean([q[1] for q in paths[k]["px"]])) if paths[k]["px"] else 0))
        num = {}
        for k in order:  # 같은 모양이 둘 이상이면 위에서부터 1, 2 … (예: 이론선 + 위·아래 ±5% 파선)
            st = paths[k]["style"]
            seen[st] = seen.get(st, 0) + 1
            num[k] = f" {seen[st]}" if cnt[st] > 1 else ""
        pairs = [(dict(m0, name=(f"{base} · " if base else "") + label(p["style"], style_lab) + num[k]), p, q) for k, (p, q) in enumerate(zip(paths, pts))]
    else:
        used, pairs = set(), [None] * len(members)
        for i, mb in enumerate(members):
            st = norm_style(mb.get("style"))
            k = next((k for k, p in enumerate(paths) if k not in used and p["style"] == st), None)
            if k is not None:
                used.add(k)
                pairs[i] = (dict(mb), paths[k], pts[k])
        rest = [k for k in range(len(paths)) if k not in used]
        for i in range(len(members)):
            if pairs[i] is None:
                if not rest:
                    return None
                k = rest.pop(0)
                pairs[i] = (dict(members[i]), paths[k], pts[k])
    out = []
    names = ", ".join(p[0]["name"] for p in pairs)
    for mb, p, (px, shared, kind) in pairs:
        if not px:
            continue
        w = [f"같은 색 선을 선 모양으로 갈랐습니다({names}) — 이 계열: {label(p['style'])}" + (f", 표식 {len(p['markers'])}개" if p["markers"] else "")]
        if p.get("style_guess") and p.get("marker_only"):
            w.append("선 경로는 가르지 못해 표식 채움으로 나눴습니다(속 찬 점 = 실선, 빈 원 = 파선으로 봄) — 확인하세요")
        elif p.get("style_guess"):
            w.append("선이 흐려(저해상도) 모양 판정이 애매해, 선마다 끊긴 비율 순서(실선 < 파선 < 점선)로 범례 이름과 짝지었습니다 — 확인하세요")
        if shared:
            w.append(f"같은 자리의 다른 표식에 가려 안 보이는 표식 {len(shared)}개는 이 선의 높이로 채웠습니다('겹침')")
        o = dict(mb, px=px, mode="line", how="markers" if kind == "line+markers" else "n", kind=kind, line_style=p["style"], split=True, warn=w, _path=p)
        if shared:
            o["shared"] = [[float(x) - 0.5, float(x) + 0.5] for x in shared]
        out.append(o)
    return out


def style_legend(legend, colors):
    """범례에서 '모양만 알려 주는' 검은 견본 줄(예: 실선+속 찬 점 = 3 boxes, 파선+빈 원 = 4 boxes)을 떼어 낸다.
    조건: 색 있는 계열 줄이 1개 이상, 무채색 어두운 줄이 2개 이상이고 서로 모양(선·표식)이 다르며, 그림 안에 어두운 무채색 선이 없음.
    → (계열 범례, {모양: 이름}) — 조건이 안 맞으면 (legend, {})"""
    def rgb(it):
        w = D.color_word(it.get("color"))
        return np.array(D.hex2rgb(w), float) if w else None
    neu, chro = [], []
    for it in legend or []:
        c = rgb(it)
        if c is None:
            continue
        (neu if (c.max() - c.min() < 30 and c.max() < 110) else chro if c.max() - c.min() > 50 else []).append(it)
    if len(neu) < 2 or not chro:
        return legend, {}
    keys = [(norm_style(it.get("style")) or "solid", str(it.get("marker") or "").lower()) for it in neu]
    if len(set(keys)) < len(keys):
        return legend, {}
    for c in colors or []:
        r = np.array(D.hex2rgb(c["color"]), float)
        if r.max() - r.min() < 30 and r.max() < 110:
            return legend, {}  # 그림 안에 검은 선이 있음 → 진짜 계열
    sty = {}
    for it, (st, _) in zip(neu, keys):
        if st in sty:
            return legend, {}
        sty[st] = str(it.get("name") or "").strip()
    return [it for it in legend if it not in neu], sty


def _dark_mask(a, col, tol, box, legends=None):
    """검정·진회색 선: 가는 선·저해상도는 안티앨리어싱으로 회색이 됨 → 무채색이면서 어두운 픽셀 넓게 (open_markers 와 같은 방식)"""
    if max(col) < 90 and max(col) - min(col) < 30:
        m, off = D.mask_for(a, col, max(tol, 300), box, legends or [])
        sub = a[off[1]:off[1] + m.shape[0], off[0]:off[0] + m.shape[1]].astype(np.int16)
        m &= ((sub.max(axis=2) - sub.min(axis=2)) < 40) & (D.gray(sub) < 200)
        return m, off
    return D.mask_for(a, col, tol, box, legends or [])


def multi_track(a, box, color, k=0, tol=60, legends=None):
    """같은 색·같은 모양 선 여러 개(교차함)를 방향이 이어지게 따로 추적한다 → 경로 목록(k개) 또는 None.
    열마다 덩어리를 나눠 갖되, 두 선이 만나 한 덩어리가 되면 둘 다 그 덩어리를 지나가고(교차),
    갈라진 뒤에는 각자 내다본 높이(앞 몇 열의 기울기)에 가까운 덩어리를 가진다 — 색이 같아도 교차점에서 정체가 바뀌지 않게.
    k=0 이면 덩어리가 둘 이상인 열이 많은지로 선 수를 정한다(2~4)."""
    import itertools
    col = D.hex2rgb(color) if isinstance(color, str) else tuple(color)
    m, off = _dark_mask(a, col, tol, box, legends)
    if m.sum() < 30:
        return None
    H, W = m.shape
    lw = line_width(m)
    cols = []
    for c in range(W):
        rs = [r for r in D.runs(m[:, c])]
        if rs:
            cols.append((c, [((s + e) / 2, e - s + 1) for s, e in rs]))
    if len(cols) < 0.3 * W:
        return None
    cnt = np.array([len(r) for _, r in cols])
    if k <= 0:
        vals, nums = np.unique(cnt[cnt >= 2], return_counts=True)
        if not len(vals):
            return None
        k = int(vals[np.argmax(nums)])
        if k > 4 or np.mean(cnt >= k) < 0.4:
            return None
    elif np.mean(cnt >= k) < 0.3:
        return None
    if k < 2:
        return None
    # 시작: 덩어리가 정확히 k 개인 열 중 가장 왼쪽 (k 개 덩어리가 이어지는 첫 구간)
    start = next((i for i, (c, r) in enumerate(cols) if len(r) == k and all(len(cols[j][1]) == k for j in range(i, min(len(cols), i + 5)))), None)
    if start is None:
        return None
    tracks = [[(cols[start][0], r[0])] for r in sorted(cols[start][1])]

    def pred(t, c):
        pts = [p for p in t[-12:] if c - p[0] <= 25]
        if len(pts) >= 4 and pts[-1][0] - pts[0][0] >= 3:
            kk = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
            return float(np.polyval(kk, c))
        return t[-1][1]

    def run(order, tracks):
        for i in order:
            c, rs = cols[i]
            ps = [pred(t, c) for t in tracks]
            if len(rs) >= len(tracks):
                best = None
                for perm in itertools.permutations(range(len(rs)), len(tracks)):
                    cost = sum(abs(rs[j][0] - p) for j, p in zip(perm, ps))
                    if best is None or cost < best[0]:
                        best = (cost, perm)
                for t, j, p in zip(tracks, best[1], ps):
                    if abs(rs[j][0] - p) <= max(6, 3 * lw) + 0.0:
                        t.append((c, rs[j][0]))
            else:  # 덩어리가 모자람: 만난 선들 — 가장 가까운 덩어리를 공유(교차 덩어리 가운데가 아니라 각자 예측 높이로 둬 방향을 지킴)
                for t, p in zip(tracks, ps):
                    j = min(range(len(rs)), key=lambda q: abs(rs[q][0] - p))
                    r_, w_ = rs[j]
                    if abs(r_ - p) <= w_ / 2 + max(4, 2 * lw):
                        t.append((c, p if w_ > 1.6 * lw + 1 else r_))
    run(range(start + 1, len(cols)), tracks)
    tracks = [t[::-1] for t in tracks]
    run(range(start - 1, -1, -1), tracks)
    out = []
    for t in tracks:
        t = sorted(t)
        if len(t) < 0.3 * W:
            return None
        out.append([[x + off[0] + 0.5, y + off[1] + 0.5] for x, y in t])
    for i in range(len(out)):  # 서로 다른 경로여야
        for j in range(i + 1, len(out)):
            A = np.asarray(out[i])
            B_ = np.asarray(out[j])
            yb = np.interp(A[:, 0], B_[:, 0], B_[:, 1])
            if np.median(np.abs(A[:, 1] - yb)) < 3 * lw:
                return None
    return out


def tracks_to_paths(a, box, color, tracks, legends=None, tol=60, styles=None):
    """multi_track 경로 → separate() 꼴 결과. 경로마다 선 모양을 틈으로 판정하되, 저해상도라 판정이 흐리면
    범례가 말한 모양들(styles)을 '빈 비율' 순서(실선 < 일점쇄선 < 파선 < 점선)로 짝짓는다."""
    col = D.hex2rgb(color) if isinstance(color, str) else tuple(color)
    m, off = D.mask_for(a, col, tol, box, legends or [])
    lw = line_width(m)
    out = []
    for t in tracks:
        on = onoff(m, off, t)
        out.append({"style": classify(m, off, t, lw), "px": t, "n_pieces": 0, "markers": [], "_off": 1 - float(on.mean()) if len(on) else 0.0})
    want = [norm_style(x) or "solid" for x in (styles or [])]
    if len({p["style"] for p in out}) < len(out) and len(want) == len(out) and len(set(want)) == len(want):
        rank = {"solid": 0, "dashdot": 1, "dashed": 2, "dotted": 3}
        for p, st in zip(sorted(out, key=lambda p: p["_off"]), sorted(want, key=lambda w: rank.get(w, 2))):
            p["style"] = st
            p["style_guess"] = True
    for p in out:
        p.pop("_off", None)
    if len({p["style"] for p in out}) < 2:
        return None
    return {"paths": out, "lw": lw, "markers": 0, "lattice": None}
