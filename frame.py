"""축선이 없는 그래프의 그림 영역(축 상자) 찾기 — 좌표는 AI 가 아니라 픽셀로.

1) 회색·연한 바탕 사각형(ggplot·seaborn·fivethirtyeight): 종이색과 다른 바탕이 덮은 사각형 = 그림 영역,
   그 안의 바탕보다 밝거나 어두운 긴 줄 = 격자선(눈금 위치).
2) 격자선 끝(OWID·엑셀형·축선 없는 웹 차트): 연한 무채색 긴 줄(점선·파선은 틈을 이어서)의 왼쪽·오른쪽 끝이
   그림 영역의 좌우, 세로 격자선(없으면 가장 위·아래 가로선)이 위아래. 눈금 글자 띠가 아래에 있으면 바닥을 글자 바로 위까지 내린다.
돌려주는 축 사전은 digitize.detect_axes 와 같은 꼴 + "grid": {"x": [...], "y": [...]} (격자선 = 눈금 후보), "frame_src".
"""
import numpy as np

import digitize as D


def _sat(a):
    a = a.astype(np.int16)
    return a.max(axis=2) - a.min(axis=2)


def _mode_end(vals, tol=4, prefer_hi=True):
    """끝 좌표들 중 가장 많이 겹치는 값 (±tol). 같으면 prefer_hi 쪽"""
    vals = sorted(float(v) for v in vals)
    best = None
    for v in vals:
        n = sum(abs(w - v) <= tol for w in vals)
        key = (n, v if prefer_hi else -v)
        if best is None or key > best[0]:
            best = (key, float(np.median([w for w in vals if abs(w - v) <= tol])))
    return best[1] if best else None


def _lines_from_rows(m, minlen, gap):
    """행마다 (틈 gap 이하를 이은) 가장 긴 줄 → 이어진 행 묶음 [(중심, 두께, 시작, 끝)]"""
    rows = []
    for i in range(m.shape[0]):
        rs = D.merge(D.runs(m[i]), gap) if m[i].any() else []
        if rs:
            s, e = max(rs, key=lambda r: r[1] - r[0])
            if e - s + 1 >= minlen:
                rows.append((i, s, e))
    out = []
    for r in rows:
        if out and r[0] == out[-1][-1][0] + 1 and abs(r[2] - out[-1][-1][2]) <= 6:
            out[-1].append(r)
        else:
            out.append([r])
    res = []
    for g in out:
        if len(g) > 6:  # 두꺼운 띠(채움·막대)는 격자선이 아님
            continue
        res.append(((g[0][0] + g[-1][0]) / 2 + 0.5, len(g), int(np.median([r[1] for r in g])), int(np.median([r[2] for r in g]))))
    return res


def _axes(box, src, gx=(), gy=()):
    x0, y0, x1, y1 = box
    return {"xaxis": {"y": y1, "r0": int(round(y1)), "r1": int(round(y1)), "c0": int(x0), "c1": int(x1)},
            "yaxis": {"x": x0, "c0": int(round(x0)), "c1": int(round(x0)), "r0": int(y0), "r1": int(y1)},
            "box": {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)}, "frame": False, "synth_y": True,
            "axisless": True, "frame_src": src, "grid": {"x": [float(v) for v in gx], "y": [float(v) for v in gy]}}


def panel_background(a):
    """종이색과 다른 바탕 사각형 (ggplot 회색, seaborn 푸른 회색, fivethirtyeight 연회색) → 축 사전 또는 None"""
    H, W = a.shape[:2]
    if H < 60 or W < 60:
        return None
    sat = _sat(a)
    g = D.gray(a)
    paper = np.median(np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)]), axis=0)
    cand = (g >= 200) & (g < 252) & (sat < 25)
    if cand.mean() < 0.15:
        return None
    q = (a[cand] // 4).astype(np.int32)
    keys, cnt = np.unique(q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2], return_counts=True)
    k = int(keys[np.argmax(cnt)])
    bg = np.array([k // 4096, (k // 64) % 64, k % 64], float) * 4 + 2
    if np.abs(bg - paper).max() < 8:  # 바탕이 종이와 같음 — 사각형이 아님
        return None
    m = np.abs(a.astype(np.int16) - bg.astype(np.int16)).max(axis=2) <= 5
    cols = D.merge(D.runs(m.mean(axis=0) > 0.3), 3)
    if not cols:
        return None
    c0, c1 = max(cols, key=lambda r: r[1] - r[0])
    rows = D.merge(D.runs(m[:, c0:c1 + 1].mean(axis=1) > 0.5), 3)
    if not rows:
        return None
    r0, r1 = max(rows, key=lambda r: r[1] - r[0])
    cols = D.merge(D.runs(m[r0:r1 + 1].mean(axis=0) > 0.5), 3)
    c0, c1 = max(cols, key=lambda r: r[1] - r[0])
    w, h = c1 - c0 + 1, r1 - r0 + 1
    if w < 0.25 * W or h < 0.2 * H:
        return None
    inside = m[r0:r1 + 1, c0:c1 + 1].mean()
    outside = (m.sum() - m[r0:r1 + 1, c0:c1 + 1].sum()) / max(1, H * W - w * h)
    if inside < 0.55 or outside > 0.1:
        return None
    # 안쪽 격자선: 바탕보다 밝거나 어두운 무채색 긴 줄
    sub = a[r0:r1 + 1, c0:c1 + 1].astype(np.int16)
    gs = D.gray(sub)
    line = (np.abs(gs - D.gray(bg[None, None, :])[0, 0]) > 6) & (_sat(sub) < 30)
    gy = [r0 + c for c, t, s, e in _lines_from_rows(line, 0.5 * w, 10)]
    gx = [c0 + c for c, t, s, e in _lines_from_rows(line.T, 0.5 * h, 10)]
    gy = [y for y in gy if r0 + 2 < y < r1 - 1]
    gx = [x for x in gx if c0 + 2 < x < c1 - 1]
    return _axes((c0, r0, c1 + 1, r1 + 1), "panel", gx, gy)


def grid_extent(a):
    """연한 무채색 긴 줄(점선은 틈을 이음)의 끝 → 축 사전 또는 None. 눈금 글자 옆은 글자를 피해서 잰다."""
    H, W = a.shape[:2]
    g = D.gray(a)
    sat = _sat(a)
    # 바탕(가장 흔한 밝기)과 다른 연한 무채색 = 격자선 후보 — 흰 종이면 g<247, fivethirtyeight 같은 회색 종이면 바탕보다 어둡거나 밝은 선
    hist = np.bincount(np.clip(g, 0, 255).astype(int).ravel(), minlength=256)
    bgg = int(np.argmax(hist))
    light = (np.abs(g - bgg) > 6) & (g >= 150) & (sat < 40) if bgg < 247 else (g < 247) & (g >= 150) & (sat < 40)
    text = (g < 200) & (sat < 40)  # 무채색 글자 (색 있는 데이터 선은 지나가는 틈만 이음)
    # 글자 획 옆 안티앨리어싱은 선이 아님: 어두운 픽셀 둘레 2px 를 뺀다 (가로선은 가로로, 세로선은 세로로)
    th = text.copy()
    tv = text.copy()
    for k in (1, 2):
        th[:, k:] |= text[:, :-k]
        th[:, :-k] |= text[:, k:]
        tv[k:] |= text[:-k]
        tv[:-k] |= text[k:]
    # 진한 선(축선)은 선으로 남김 — 무채색이고 길게 이어진 줄은 글자가 아님
    longd = np.zeros_like(text)
    for c, t, s, e in _lines_from_rows(text & (sat < 40), 0.3 * W, 1):
        longd[int(c - t / 2):int(c + t / 2) + 1, s:e + 1] = True
    colored = sat >= 40  # 색 있는 데이터 선·표식이 격자선을 가로지르는 곳은 선의 일부로 (틈이 끊기지 않게)
    hm = (light & ~th) | longd | colored
    vm = (light & ~tv) | colored
    hl = [l for l in _lines_from_rows(hm, 0.35 * W, 8) if l[1] <= 3]
    if len(hl) < 2:
        return None
    x0 = _mode_end([l[2] for l in hl], prefer_hi=True)
    x1 = _mode_end([l[3] for l in hl], prefer_hi=False)
    if x1 is None or x0 is None or x1 - x0 < 0.3 * W:
        return None
    hl = [l for l in hl if abs(l[2] - x0) <= 8 and abs(l[3] - x1) <= 8]
    if len(hl) < 2:
        return None
    ys = sorted(l[0] for l in hl)
    vl = [l for l in _lines_from_rows(vm[:, int(x0):int(x1) + 1].T, 0.3 * (ys[-1] - ys[0] + 1), 8) if l[1] <= 3]
    vl = [(int(x0) + c, t, s, e) for c, t, s, e in vl]
    y0, y1 = ys[0], ys[-1]
    bottom_line = True
    if len(vl) >= 2:
        t0 = _mode_end([l[2] for l in vl], prefer_hi=False)
        t1 = _mode_end([l[3] for l in vl], prefer_hi=True)
        if t0 is not None and t1 is not None and t1 - t0 > 0.5 * (y1 - y0):
            y0, y1 = min(y0, t0), max(y1, t1)
    else:
        # 세로 격자선이 없으면 바닥 = 아래 눈금 글자 띠 바로 위 (데이터가 가장 아래 격자선 밑으로 내려갈 수 있음)
        neutral = (g < 160) & (sat < 40)
        for y in range(int(y1) + 2, min(H, int(y1) + int(0.25 * H))):
            row = neutral[y, int(x0):int(x1) + 1]
            if row.any():
                bl = D.merge(D.runs(row), 3)
                if len(bl) >= 3 and bl[-1][1] - bl[0][0] > 0.5 * (x1 - x0):
                    if y - y1 > 14:  # 글자 띠가 맨 아래 선에서 떨어져 있음 (로그축 등 데이터가 그 아래로 내려감)
                        y1 = y - 3.0
                        bottom_line = False
                    break
    if y1 - y0 < 0.15 * H:
        return None
    gy = [l[0] for l in hl if y0 + 2 < l[0] < y1 - 1]
    gx = [l[0] for l in vl if x0 + 2 < l[0] < x1 - 1]
    top_line = y0
    if len(vl) < 2:  # 맨 위 격자선 위로 조금 넘친 데이터(웹 차트)도 담게 위를 띄움 — 위 글자(부제목) 바로 아래까지만
        # 겹친 색 선의 번짐(파랑+갈색 → 회색)은 글자가 아님: 색 픽셀 2px 둘레는 뺀다
        col = sat >= 40
        cd = col.copy()
        for k in (1, 2):
            cd[k:] |= col[:-k]
            cd[:-k] |= col[k:]
            cd[:, k:] |= col[:, :-k]
            cd[:, :-k] |= col[:, k:]
        neutral = (g < 200) & (sat < 40) & ~cd  # 글자 획 둘레(옅은 번짐)까지 — 색 마스크에 글자 가장자리가 섞이지 않게
        lim = int(max(0, y0 - max(6, 0.12 * (y1 - y0))))
        yt = lim
        for y in range(int(y0) - 3, lim - 1, -1):
            if neutral[y, int(x0) + 2:int(x1) - 1].any():
                yt = y + 4
                break
        roi_y0 = float(min(y0, max(yt, lim)))
    else:
        roi_y0 = float(y0)
    ax = _axes((x0, y0, x1 + 1, y1), "grid", gx, gy)
    ax["bottom_line"] = bottom_line
    ax["top_line"] = float(top_line)
    ax["roi_y0"] = roi_y0  # 추출(색 마스크) 영역의 위 끝 — 그림 영역(맨 위 격자선)보다 조금 위
    return ax


def find(a):
    """축선 없는 그래프의 축 상자: 바탕 사각형 → 격자선 끝 순서"""
    for f in (panel_background, grid_extent):
        try:
            ax = f(a)
        except Exception:
            ax = None
        if ax:
            return ax
    return None
