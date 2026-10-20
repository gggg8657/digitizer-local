"""막대그래프·여러 그래프 쪽 — 픽셀 처리로 결정론적으로 (numpy + Pillow, digitize.py 를 씀).

find_panels(a)            그림(논문 쪽)에서 축 상자(왼쪽 세로축 + 아래 가로축이 만나는 모서리)를 모두 찾는다
detect_bars(a, box, ...)  막대 하나하나: 범주축 위치, 쌓인 조각(색·무늬가 바뀌는 곳), 오차 막대 끝
fill_sig(...)             조각·범례 견본의 채움 모양: 주 색 + 잉크 비율(무늬) + 빈 막대면 테두리 색
값은 값축 보정(digitize.pair_axis)으로만 계산한다. 가로 막대는 그림을 돌려서 같은 코드로 처리한다.
"""
import math

import numpy as np

import digitize as D


# ── 여러 그래프가 있는 그림: 축 상자 찾기 ───────────────────────────────────
def _segments(dark, minlen, maxthick):
    """가로로 긴 어두운 줄 → [(중심 행, 두께, 시작 열, 끝 열)]. 세로줄은 dark.T 로."""
    rows = []
    for y in range(dark.shape[0]):
        for s, e in D.runs(dark[y]):
            if e - s + 1 >= minlen:
                rows.append((y, s, e))
    rows.sort(key=lambda r: (r[1], r[0]))
    lines = []  # 같은 시작·끝으로 이어진 행들
    used = [False] * len(rows)
    by_row = {}
    for i, r in enumerate(rows):
        by_row.setdefault(r[0], []).append(i)
    for i, (y, s, e) in enumerate(rows):
        if used[i]:
            continue
        grp = [(y, s, e)]
        used[i] = True
        yy = y
        while True:
            nxt = [j for j in by_row.get(yy + 1, ()) if not used[j] and abs(rows[j][1] - s) <= 3 and abs(rows[j][2] - e) <= 3]
            if not nxt:
                break
            used[nxt[0]] = True
            grp.append(rows[nxt[0]])
            yy += 1
        if len(grp) <= maxthick:
            lines.append(((grp[0][0] + grp[-1][0]) / 2 + 0.5, len(grp), min(g[1] for g in grp), max(g[2] for g in grp)))
    return lines


def _ink_rows(ink, gap):
    """잉크가 있는 행 → 처음 만나는 빈 띠(gap 이상)까지의 길이"""
    n = 0
    blank = 0
    seen = False
    for v in ink:
        if v:
            seen, blank = True, 0
        else:
            blank += 1
            if seen and blank >= gap:
                return n - blank + 1
        n += 1
    return n - blank


def find_panels(a, min_frac=0.06):
    """기본은 선을 그대로 보고, 가장 큰 그래프가 작게 나오면(살짝 기운 스캔에서 축선이 계단처럼 끊김) 번지게 해서 다시 찾는다."""
    import layout
    p0 = _find_panels(a, min_frac, False)
    axisless=layout.panels(a)
    if len(axisless)>sum(p['kind']=='plot' for p in p0):
        return axisless + [p for p in p0 if p['kind']!='plot']
    big = lambda ps: max([(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]) for p in ps if p["kind"] == "plot"] or [0])  # noqa: E731
    H, W = a.shape[:2]
    if big(p0) >= 0.3 * H * W:
        return p0
    p1 = _find_panels(a, min_frac, True)
    return p1 if big(p1) > 1.5 * big(p0) else p0


def _find_panels(a, min_frac=0.06, tilt=False):
    """축 상자 후보 → [{box: [x0,y0,x1,y1] (축선 안쪽 그림 영역), crop: 눈금 글자·축 이름까지 넣은 자르기 범위,
    kind: 'plot' | 'image'(지도·사진처럼 안이 빽빽함), score}]. 큰 것부터 위→아래·왼→오른 순."""
    H, W = a.shape[:2]
    g = D.gray(a)
    dark = g < 160
    minlen = max(40, int(min_frac * min(H, W)))
    th = max(6, int(0.006 * max(H, W)))
    def smear(m, ax_):  # 살짝 기운 스캔: 선이 계단처럼 끊기지 않게 수직 방향으로 3px 번지게
        out = m.copy()
        for k in (1, 2, 3):
            out[k:] |= m[:-k] if ax_ == 0 else out[k:]
            out[:-k] |= m[k:] if ax_ == 0 else out[:-k]
        return out
    if tilt:
        hs = _segments(smear(dark, 0), minlen, th + 6)
        vs = [(c, t, s, e) for c, t, s, e in _segments(smear(dark.T, 0), minlen, th + 6)]
    else:
        hs = _segments(dark, minlen, th)
        vs = [(c, t, s, e) for c, t, s, e in _segments(dark.T, minlen, th)]

    def unshadow(ls):  # 번진(바로 세운·흐린) 선의 가장자리 행이 짧게 따로 잡힌 것은 바로 옆의 긴 선에 맡김
        return [l for l in ls if not any(o is not l and abs(o[0] - l[0]) <= 2.5 and abs(o[2] - l[2]) <= 3 and o[3] - o[2] > l[3] - l[2] + 3 for o in ls)]
    hs, vs = unshadow(hs), unshadow(vs)

    satb = (a.max(axis=2).astype(np.int16) - a.min(axis=2)) >= 40

    def bridge(ls, lim, sm):  # 축선 위에 겹친 색 표식(밝은 초록 세모 등)에 몇 px 끊긴 같은 줄은 잇는다 — 틈이 색 픽셀일 때만 (빈 틈은 다른 칸)
        ls = sorted(ls, key=lambda l: (round(l[0]), l[2]))
        out_ = []

        def colored_gap(o, l):
            c_ = int(round((o[0] + l[0]) / 2 - 0.5))
            g_ = sm[c_, o[3] + 1:l[2]] if 0 <= c_ < sm.shape[0] else sm[0, 0:0]
            return g_.size == 0 or g_.mean() >= 0.5
        for l in ls:
            o = next((o for o in out_ if abs(o[0] - l[0]) <= 2 and 0 <= l[2] - o[3] - 1 <= lim and colored_gap(o, l)), None)
            if o is not None:
                out_[out_.index(o)] = ((o[0] + l[0]) / 2, max(o[1], l[1]), o[2], l[3])
            else:
                out_.append(l)
        return out_
    hs, vs = bridge(hs, max(6, int(0.006 * W)), satb), bridge(vs, max(6, int(0.006 * H)), satb.T)

    def inside_fill(ls, m):  # 칠한 막대 안의 가는 줄(맞닿은 두 색 막대의 경계 행, 막대를 지나는 오차 막대) — 위·아래가 다 칠해져 있으면 축이 아님
        out = []
        for l in ls:
            c, t, s0, e0 = l
            ra, rb = int(c - t / 2 - 3), int(c + t / 2 + 3)
            if 0 <= ra and rb < m.shape[0] and e0 > s0:
                up, dn = m[ra, s0:e0 + 1].mean(), m[rb, s0:e0 + 1].mean()
                if (m[ra, s0:e0 + 1] & m[rb, s0:e0 + 1]).mean() > 0.5 or (min(up, dn) > 0.3 and up + dn > 1.3):
                    continue
            out.append(l)
        return out
    hs, vs = inside_fill(hs, dark), inside_fill(vs, dark.T)
    # 색 있는 가로줄(세로축에서 출발하는 평평한 데이터 곡선·색 화살표)은 축 상자의 아래 축이 아님 — 그래프를 위아래로 쪼개지 않게
    sat_ = a.max(axis=2).astype(np.int16) - a.min(axis=2)

    def achro(l):  # 무채색 줄이거나, 색 픽셀이 칠한 면(막대 채움)의 가장자리면 축으로 둠 — 위아래가 비고 색만 가는 줄(데이터 곡선)은 뺌
        c, t, s0, e0 = l
        r = int(round(c - 0.5))
        seg = dark[r, s0:e0 + 1]
        if not seg.any() or float(np.median(sat_[r, s0:e0 + 1][seg])) < 80:
            return True
        col = seg & (sat_[r, s0:e0 + 1] >= 80)
        k = int(t) + max(6, int(0.01 * H))  # 굵은 색 곡선(4~6px)은 이만큼 떨어지면 색이 없음 — 막대 채움은 이어짐
        up = (sat_[max(0, r - k):r, s0:e0 + 1] >= 80).all(axis=0)  # 위(또는 아래)로 k 행 내내 색 = 칠한 면
        dn = (sat_[r + 1:min(H, r + k + 1), s0:e0 + 1] >= 80).all(axis=0)
        return float((col & (up | dn)).sum()) >= 0.5 * col.sum()
    hs = [l for l in hs if achro(l)]
    cands = []
    for vx, vt, vs0, ve in vs:
        # 왼쪽 끝이 이 세로축에 닿는 가로선들 (아래 축, 위 테두리, 위아래로 붙은 그래프의 공유 축)
        lt = 3 + max(12, int(0.02 * max(H, W)))  # 바깥 눈금이 가로선 왼쪽 끝을 늘림

        def meets(hs0, ht):
            return vx - lt - ht <= hs0 <= vx + 3 + max(vt, ht)
        touch = sorted(hy for hy, ht, hs0, he in hs if meets(hs0, ht) and vs0 - 3 - ht <= hy <= ve + 3 + ht and he - vx >= minlen)
        if not touch or abs(touch[-1] - ve) > 4 + vt:
            continue  # 세로선 아래 끝이 가로축과 만나야 축 상자
        tops = [vs0 + 0.5] + touch
        for i, hy in enumerate(touch):
            top = tops[i]
            if hy - top < minlen:
                continue
            he = max(e for y_, t_, s_, e in hs if y_ == hy and meets(s_, t_))
            cands.append([vx, top, he + 0.5, hy])
    for b in cands:  # 번지게 한 선의 가운데 → 실제 축선 가운데로 (±5px 안에서 어두운 픽셀이 가장 많은 열·행)
        r0, r1, c0, c1 = int(b[1]), int(b[3]), int(b[0]), int(b[2])
        xs = [x for x in range(max(0, int(b[0]) - 5), min(W, int(b[0]) + 6))]
        if xs and r1 > r0:
            cnt = [dark[r0:r1, x].sum() for x in xs]
            best = max(cnt)
            sel = [x for x, c_ in zip(xs, cnt) if c_ >= 0.8 * best]
            b[0] = (min(sel) + max(sel)) / 2 + 0.5
        ys = [y for y in range(max(0, int(b[3]) - 5), min(H, int(b[3]) + 6))]
        if ys and c1 > c0:
            cnt = [dark[y, c0:c1].sum() for y in ys]
            best = max(cnt)
            sel = [y for y, c_ in zip(ys, cnt) if c_ >= 0.8 * best]
            b[3] = (min(sel) + max(sel)) / 2 + 0.5
    # 같은 모서리를 가진 후보(격자선 등)는 가장 큰 것만, 다른 축 상자 안에 든 상자(범례 틀·삽입 그림)는 뺌
    cands.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    boxes, insets = [], []
    for b in cands:
        if any(abs(b[0] - o[0]) <= 6 and abs(b[3] - o[3]) <= 6 for o in boxes):
            continue
        host = next((o for o in boxes if o[0] - 3 <= b[0] and o[1] - 3 <= b[1] and b[2] <= o[2] + 3 and b[3] <= o[3] + 3), None)
        if host is not None:
            # 큰 그래프 안의 작은 축 상자: 눈금 글자가 왼쪽에 있고 충분히 크면 삽입 그래프 — 큰 그래프 추출에서 그 영역을 뺀다
            bw_, bh_ = b[2] - b[0], b[3] - b[1]
            if bw_ * bh_ >= 0.04 * (host[2] - host[0]) * (host[3] - host[1]) and bw_ < 0.8 * (host[2] - host[0]):
                lb_ = dark[int(b[1]):int(b[3]), max(0, int(b[0]) - int(0.2 * bw_) - 10):max(0, int(b[0]) - 3)]
                if lb_.size and lb_.any(axis=0).sum() >= 4 and not any(abs(b[0] - q[0]) <= 6 and abs(b[3] - q[3]) <= 6 for q in insets):
                    insets.append(b)
            continue
        boxes.append(b)
    out = []
    for b in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        inner = a[y0 + 3:y1 - 2, x0 + 3:x1 - 2]
        if not inner.size:
            continue
        ig = D.gray(inner)
        sat = inner.max(axis=2).astype(int) - inner.min(axis=2)
        busy = float(((ig < 235) | (sat > 40)).mean())
        # 왼쪽 눈금 글자: 세로축 바로 왼쪽에 잉크가 있는지
        lb = dark[y0:y1, max(0, x0 - int(0.25 * (x1 - x0)) - 10):max(0, x0 - 3)]
        has_lbl = bool(lb.size and lb.any(axis=0).sum() >= 4)
        lbg = D.gray(a[y0:y1, max(0, x0 - 40):max(0, x0 - 3)])
        inner_busy = busy
        if lbg.size and float((lbg < 235).mean()) > 0.5:  # 축 왼쪽도 빽빽함 → 지도 속 삽입 그림
            busy = max(busy, 0.6)
        kind = "image" if busy > 0.55 else "plot" if has_lbl else "other"
        why = {"image": "안이 빽빽함 (지도·사진)", "other": "왼쪽 눈금 글자가 없음 (그래프가 아닐 수 있음)"}.get(kind, "")
        out.append({"box": [float(v) for v in b], "kind": kind, "why": why, "busy": round(busy, 3), "inner": round(inner_busy, 3),
                    "score": (x1 - x0) * (y1 - y0) * (1 if kind == "plot" else 0.1)})
    # 칸막이 그래프: 같은 높이의 상자가 바짝 붙어 줄지어 있으면(왼쪽 상자의 값축을 같이 씀) 한 그래프로
    out.sort(key=lambda p: p["box"][0])
    merged = []
    for p in out:
        q = next((q for q in merged if q["kind"] == "plot" and p["inner"] <= 0.55 and abs(q["box"][1] - p["box"][1]) <= 4
                  and abs(q["box"][3] - p["box"][3]) <= 4 and -2 <= p["box"][0] - q["box"][2] <= max(12, 0.02 * W)), None)
        if q:
            q.setdefault("part_boxes", [[float(v) for v in q["box"]]]).append([float(v) for v in p["box"]])
            q["box"] = [q["box"][0], min(q["box"][1], p["box"][1]), p["box"][2], max(q["box"][3], p["box"][3])]
            q["parts"] = q.get("parts", 1) + 1
            q["why"] = f"칸 {q['parts']}개가 값축을 같이 쓰는 그래프"
            q["score"] += p["score"]
        else:
            merged.append(p)
    out = merged
    for p in out:
        p["crop"] = _crop_for(a, p["box"], [o["box"] for o in out if o is not p])
        ins = [b for b in insets if p["box"][0] - 3 <= b[0] and p["box"][1] - 3 <= b[1] and b[2] <= p["box"][2] + 3 and b[3] <= p["box"][3] + 3]
        if ins and p["kind"] == "plot":  # 삽입 그래프: 눈금 글자·축 이름까지 넓힌 영역 (추출에서 뺄 곳)
            p["insets"] = [[float(v) for v in _crop_for(a, b, [])] for b in ins]
            p["insets"] = [[max(q[0], p["box"][0]), max(q[1], p["box"][1]), min(q[2], p["box"][2]), min(q[3], p["box"][3])] for q in p["insets"]]
    out.sort(key=lambda p: (round(p["box"][1] / max(1, 0.25 * H)), p["box"][0]))
    return out


def _crop_for(a, box, others):
    """축 상자 → 눈금 글자·축 이름까지 넣은 자르기 범위. 옆 그래프 상자나 넓은 빈 띠에서 멈춘다."""
    H, W = a.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ink = D.gray(a) < 200
    lo_x = max(0, x0 - 0.7 * w)
    hi_y = min(H, y1 + 0.5 * h)
    lo_y = max(0, y0 - 0.12 * h)
    hi_x = min(W, x1 + 0.08 * w)
    for o in others:
        if o[1] < y1 and o[3] > y0:  # 세로로 겹침 → 왼쪽·오른쪽 이웃
            if o[2] <= x0:
                lo_x = max(lo_x, o[2] + 4)
            if o[0] >= x1:
                hi_x = min(hi_x, o[0] - 4)
        if o[0] < x1 and o[2] > x0:  # 가로로 겹침 → 위·아래 이웃 (이웃의 위 글자 몫 조금 남김)
            if o[1] >= y1 - 3:  # 아래 이웃 (축선을 같이 쓰면 그 선까지는 넣음)
                hi_y = min(hi_y, max(y1 + 4, o[1] - 0.04 * (o[3] - o[1])))
            if o[3] <= y0 + 3:
                lo_y = min(y0 - 4, max(lo_y, o[3] + 4))
    gap_y = max(12, int(0.07 * h))
    rows = ink[int(y1) + 3:int(hi_y), int(lo_x):int(hi_x)].any(axis=1)
    hi_y = min(hi_y, y1 + 3 + _ink_rows(rows, gap_y) + 6)
    gap_x = max(12, int(0.06 * w))
    cols = ink[int(y0):int(hi_y), int(lo_x):int(x0) - 3].any(axis=0)[::-1]
    lo_x = max(lo_x, x0 - 3 - _ink_rows(cols, gap_x) - 6)
    return [float(max(0, lo_x)), float(max(0, lo_y)), float(min(W, hi_x)), float(min(H, hi_y))]


# ── 막대 검출 (세로 막대 기준 — 가로 막대는 그림을 돌려서) ────────────────────────
def rot(a):
    """가로 막대 → 세로 막대: 반시계 90° (왼쪽 축이 아래로). 원래 (x, y) ↔ 돌린 (y, W - x)"""
    return np.ascontiguousarray(np.rot90(a, 1))


def ink_mask(a):
    """배경(흰색) 아닌 픽셀 — 연한 색 채움도 포함"""
    a = a.astype(np.int16)
    sat = a.max(axis=2) - a.min(axis=2)
    return (D.gray(a) < 228) | (sat > 45)


def _label(m):
    try:
        from scipy import ndimage
        return ndimage.label(m, structure=np.ones((3, 3)))
    except ImportError:
        return D._label_py(m)


def _contig(row, l, r):
    """row[l..r] 안 가장 긴 연속 True 길이"""
    best = 0
    for s, e in D.runs(row[l:r + 1]):
        best = max(best, e - s + 1)
    return best


def _strip_cols(l, r):
    """막대 안 표본 열: 테두리 바로 안쪽 양옆 띠 — 가운데(오차 막대 줄기·막대 안 글자 상자)는 피한다"""
    w = r - l + 1
    if w < 14:
        return np.arange(l + 1, r) if w > 2 else np.arange(l, r + 1)
    m, k = max(2, int(round(0.05 * w))), max(3, int(round(0.15 * w)))
    return np.concatenate([np.arange(l + m, l + k + 1), np.arange(r - k, r - m + 1)])


def fill_sig(px, ink, wall=None):
    """채움 모양: px (N×3) 표본 픽셀, ink (N) 잉크 여부. → {kind: solid|hatch|empty, color, ink}
    빈 막대(흰 채움)는 테두리 색(wall: 테두리 픽셀)으로 구분한다."""
    px = np.asarray(px, float).reshape(-1, 3)
    ink = np.asarray(ink, bool).reshape(-1)
    f = float(ink.mean()) if ink.size else 0.0
    if f <= 0.08:
        w = np.asarray(wall, float).reshape(-1, 3) if wall is not None and len(wall) else None
        if w is not None:
            w = w[D.gray(w) < 200] if (D.gray(w) < 200).any() else w
        col = w.mean(axis=0) if w is not None and len(w) else np.array([0.0, 0, 0])
        return {"kind": "empty", "color": D.rgb2hex(col), "ink": round(f, 3)}
    sel = px[ink]
    # 잉크 픽셀 중 진한 쪽 절반(안티앨리어싱 번짐 빼기) 평균
    gg = D.gray(sel)
    core = sel[gg <= np.median(gg)] if f < 0.95 else sel
    return {"kind": "solid" if f >= 0.95 else "hatch", "color": D.rgb2hex(core.mean(axis=0)), "ink": round(f, 3)}


def sig_dist(p, q):
    """채움 모양 거리 (0 = 같음). 색 거리 + 잉크 비율 차 + 종류 차"""
    c = np.sqrt(((np.array(D.hex2rgb(p["color"]), float) - D.hex2rgb(q["color"])) ** 2).sum())
    d = c + 220 * abs(p["ink"] - q["ink"])
    if p["kind"] != q["kind"]:
        d += 60 if "empty" in (p["kind"], q["kind"]) else 30  # 꽉 찬 색 ↔ 같은 색 무늬는 다른 계열
    return float(d)


def detect_bars(a, box, base_row, axis_rows=None, left_col=None):
    """세로 막대 검출. box = 그림 영역(x0,y0,x1,y1 — 축선 중심), base_row = 막대가 서는 축선의 첫 행(픽셀).
    → [{l, r, top_outer, top_px, segs: [{y0, y1, sig}], err: {hi_px, lo_px, sym}}] (픽셀, 위가 0)"""
    H, W = a.shape[:2]
    x0 = int(math.ceil(left_col if left_col is not None else box["x0"])) + 2
    x1 = int(math.floor(box["x1"])) - 2
    y0 = int(math.ceil(box["y0"])) + 2
    while x1 > x0 + 10 and _contig(ink_mask(a[y0:int(base_row), x1 - 1:x1])[:, 0], 0, int(base_row) - y0 - 1) > 0.9 * (int(base_row) - y0):
        x1 -= 1  # 오른쪽 테두리 번짐 열
    yb = int(base_row)  # 축선 첫 행 — 이 위가 그림
    if x1 - x0 < 10 or yb - y0 < 10:
        return []
    full = ink_mask(a[:, x0:x1])
    yb0 = yb
    while yb - 1 > y0 and yb0 - yb < 6 and (_contig(full[yb - 1], 0, x1 - x0 - 1) > 0.9 * (x1 - x0) or full[yb - 1].mean() > 0.9):  # 축선 위 번짐 행도 축선 (칸막이로 끊긴 축선 포함)
        yb -= 1  # 6행까지만 — 그 위는 그림 폭 거의 전부를 덮은 맞붙은 막대(히스토그램)일 수 있음
    while x0 < x1 - 10 and _contig(ink_mask(a[y0:yb, x0:x0 + 1])[:, 0], 0, yb - y0 - 1) > 0.9 * (yb - y0):  # 세로축선 번짐 열
        x0 += 1
    sub = a[y0:yb, x0:x1]
    ink = ink_mask(sub)
    h, w = ink.shape
    lab, n = _label(ink)
    band = lab[max(0, h - 6):h]
    ids = [i for i in np.unique(band) if i]
    g = D.gray(sub)
    bars = []
    for cid in ids:
        cols = (band == cid).any(axis=0)
        rs = D.merge(D.runs(cols), 3)
        comp = lab == cid
        if len(rs) > 1:  # 무늬만 있는 바닥(///): 사이 열도 위쪽에 무늬가 차 있으면 한 막대
            occ = comp.sum(axis=0)
            hgt = float(np.median([occ[s_:e_ + 1].max() for s_, e_ in rs]))
            m_ = [rs[0]]
            for s_, e_ in rs[1:]:
                gap_ = occ[m_[-1][1] + 1:s_]
                if gap_.size and gap_.min() >= max(3, 0.06 * hgt):
                    m_[-1] = (m_[-1][0], e_)
                else:
                    m_.append((s_, e_))
            rs = m_
        wide = [(s, e) for s, e in rs if e - s + 1 >= 5]
        thin = [(s, e) for s, e in rs if e - s + 1 < 5]
        for s, e in wide:
            bars.append((s, e, "fill"))
        # 빈 막대: 가는 벽 둘이 위에서 가로선으로 이어짐
        used = set()
        for i in range(len(thin)):
            if i in used:
                continue
            for j in range(i + 1, len(thin)):
                if j in used:
                    continue
                l, r = thin[i][0], thin[j][1]
                if r - l + 1 < 6:
                    continue
                if any(s <= l <= e or s <= r <= e or (l <= s and e <= r) for s, e in wide):
                    break
                rows = comp[:, l:r + 1]
                cover = np.array([_contig(rows[y], 0, r - l) for y in range(h)])
                if (cover >= 0.85 * (r - l + 1)).any():
                    bars.append((l, r, "open"))
                    used.update((i, j))
                    break
    if not bars:
        return []
    # 맞붙은 막대(묶음 막대) 나누기: 바닥 색이 바뀌는 곳
    split = []
    for l, r, k in bars:
        if k != "fill" or r - l < 16:
            split.append((l, r, k))
            continue
        bm = sub[max(0, h - 42):h - 2, l:r + 1].astype(float).mean(axis=0)  # 빗금 무늬도 열마다 같아지게 40행 평균
        cuts = [l]
        q = 6
        for c in range(q, r - l + 1 - q):
            L_, R_ = bm[c - q:c], bm[c:c + q]
            dl = np.sqrt(((L_.mean(axis=0) - R_.mean(axis=0)) ** 2).sum())
            noise = (np.sqrt((L_.std(axis=0) ** 2).sum()) + np.sqrt((R_.std(axis=0) ** 2).sum())) / 2  # 무늬면 창 안에서도 출렁임
            if dl > 90 and dl > 3 * noise and c + l - cuts[-1] >= 6:
                cuts.append(l + c)
        cuts.append(r + 1)
        for s_, e_ in zip(cuts[:-1], cuts[1:]):
            if e_ - s_ >= 5:
                split.append((s_, e_ - 1, k))
    split2, parts = [], []
    for l, r, k in split:
        pc = _split_touching(sub, ink, g, l, r, h) if k == "fill" and r - l >= 16 else [(l, r)]
        if len(pc) > 1:
            parts.append(((l, r, k), pc))
        else:
            split2.append((l, r, k))
    for whole, pc in parts:  # 다른 (안 나뉜) 막대가 나뉜 조각보다 훨씬 넓으면 나누지 않음 (범례·글자에 가린 막대 하나)
        unit = float(np.median([e_ - s_ + 1 for s_, e_ in pc]))
        if any(k_ == "fill" and r_ - l_ + 1 > 1.6 * unit for l_, r_, k_ in split2):
            split2.append(whole)
        else:
            split2 += [(s_, e_, whole[2]) for s_, e_ in pc]
    bars = sorted(split2)
    wmax = max(r - l + 1 for l, r, _ in bars)
    bars = [b for b in bars if b[1] - b[0] + 1 >= max(5, 0.3 * wmax)]
    out = []
    for l, r, kind in bars:
        b = _measure(sub, ink, g, l, r, h)
        if b is None or (kind == "open" and b["top_outer"] <= 3):  # 칸막이 틀(위 테두리까지 닿는 빈 '막대')
            continue
        b["l"], b["r"] = l + x0, r + x0 + 1  # 픽셀 경계 (r 은 끝 열 다음)
        b["cx"] = (b["l"] + b["r"]) / 2
        b["open"] = kind == "open"
        for key in ("top_outer", "top_px"):
            b[key] += y0
        for sgm in b["segs"]:
            sgm["y0"] += y0
            sgm["y1"] += y0
        for key in ("hi_px", "lo_px"):
            if b["err"].get(key) is not None:
                b["err"][key] += y0
        out.append(b)
    # 다른 막대를 품은 넓은 빈 '막대'(칸 틀)는 뺀다
    out = [b for b in out if not (b["open"] and any(o is not b and b["l"] <= o["l"] and o["r"] <= b["r"] for o in out))]
    return _zero_bins(out, y0 + h)


def _zero_bins(out, base):
    """히스토그램(같은 폭 막대가 대부분 맞붙음): 막대 폭의 정수배인 빈 틈 = 높이 0 인 칸 → 값 0 막대로 채운다"""
    if len(out) < 4:
        return out
    out = sorted(out, key=lambda b: b["l"])
    ws = np.array([b["r"] - b["l"] for b in out], float)
    w = float(np.median(ws))
    if (np.abs(ws - w) > max(2.0, 0.08 * w)).any():
        return out
    gaps = [q["l"] - p["r"] for p, q in zip(out[:-1], out[1:])]
    if sum(gp_ <= 2 for gp_ in gaps) < 0.6 * len(gaps):
        return out
    s0 = out[0]["segs"][0]["sig"]
    if any(len(b["segs"]) != 1 or sig_dist(b["segs"][0]["sig"], s0) > 40 for b in out):
        return out  # 히스토그램은 한 색 — 색이 다른 막대가 맞붙은 것은 묶음 막대
    gi = [i for i, gp_ in enumerate(gaps) if gp_ > 2]
    if len(gi) >= 2 and len({j - i for i, j in zip(gi[:-1], gi[1:])}) == 1 and gi[0] + 1 == gi[1] - gi[0]:
        return out  # 일정한 간격마다 빈 틈 = 같은 색 묶음 막대의 묶음 사이
    add = []
    for p, gp_ in zip(out[:-1], gaps):
        k = int(round(gp_ / w))
        if k >= 1 and abs(gp_ - k * w) <= max(2.0, 0.1 * w):
            for i in range(k):
                l_ = p["r"] + gp_ * i / k
                r_ = p["r"] + gp_ * (i + 1) / k
                sg = {"y0": float(base), "y1": float(base), "sig": dict(p["segs"][0]["sig"])}
                add.append({"l": l_, "r": r_, "cx": (l_ + r_) / 2, "open": False, "zero": True, "top_outer": base, "top_px": float(base),
                            "segs": [sg], "err": {}, "w": r_ - l_})
    return sorted(out + add, key=lambda b: b["l"])


def _top_profile(ink, l, r, h):
    """열마다 바닥에서 위로 이어진 잉크의 맨 위 행 (sub 좌표)"""
    col = ink[:h, l:r + 1]
    blank = ~col
    # 바닥에서 위로 처음 만나는 빈 행 바로 아래 = 꼭대기
    rev = blank[::-1]
    first = np.where(rev.any(axis=0), rev.argmax(axis=0), h)
    return (h - first).astype(float)


def _split_touching(sub, ink, g, l, r, h):
    """맞붙은 막대(히스토그램·간격 없는 막대) 나누기 → [(l, r)].
    (1) 막대 사이 가는 세로 테두리선, (2) 꼭대기 높이가 계단처럼 바뀌는 곳, (3) 그렇게 나눈 막대 폭의 정수배인 조각은 똑같이 나눔.
    무늬(세로 빗금)·글자 때문에 생긴 가짜 경계는 조각이 고르지 않으면 버린다."""
    w = r - l + 1
    top = _top_profile(ink, l, r, h)
    hb = int(min(40, max(4, h - np.median(top) - 2)))
    band = g[max(0, h - hb - 2):h - 2, l:r + 1].astype(float)
    cm = band.mean(axis=0)
    fill = float(np.median(cm))
    cuts = set()
    # (1) 가는 어두운 세로선 (막대 안쪽 3px 넘게 떨어진 곳)
    darkc = (cm < fill - 40) & ((band < fill - 40).mean(axis=0) > 0.8)
    for s_, e_ in D.runs(darkc):
        if e_ - s_ + 1 <= 4 and s_ >= 4 and e_ <= w - 5:
            cuts.add((s_ + e_ + 1) // 2)
    # (2) 꼭대기 계단: 양쪽 6열이 평평하고 3px 넘게 차이
    q = 6
    best = {}
    for c in range(q, w - q + 1):
        L_, R_ = top[c - q:c], top[c:c + q]
        if L_.max() - L_.min() <= 1.5 and R_.max() - R_.min() <= 1.5 and abs(np.median(L_) - np.median(R_)) >= 3:
            best[c] = abs(np.median(L_) - np.median(R_))
    for c in sorted(best):
        if any(abs(c - o) < q for o in cuts):
            continue
        cuts.add(c)
    if not cuts:
        return [(l, r)]
    edges = [0] + sorted(cuts) + [w]
    pcs = [(a_, b_ - 1) for a_, b_ in zip(edges[:-1], edges[1:])]
    ws = np.array([b_ - a_ + 1 for a_, b_ in pcs], float)
    unit = float(np.min(ws[ws >= 0.6 * np.median(ws)])) if len(ws) else w
    if ws.min() < 8 or ws.max() > 12 * unit:
        return [(l, r)]
    # 고른 폭인지: 각 조각이 단위 폭의 정수배(±15%)여야
    ks = np.round(ws / unit)
    if (np.abs(ws - ks * unit) > np.maximum(2.5, 0.15 * unit)).any():
        return [(l, r)]
    out = []
    for (a_, b_), k in zip(pcs, ks):
        k = int(k)
        for i in range(k):
            s_ = a_ + int(round(i * (b_ - a_ + 1) / k))
            e_ = a_ + int(round((i + 1) * (b_ - a_ + 1) / k)) - 1
            out.append((l + s_, l + e_))
    return out


def _measure(sub, ink, g, l, r, h):
    """막대 하나: 꼭대기·쌓인 조각·오차 막대 (sub 좌표)"""
    w = r - l + 1
    st = _strip_cols(l, r)
    wl = np.arange(l, min(r, l + 3))
    wr = np.arange(max(l, r - 2), r + 1)
    y = h - 1
    gap = 0
    top = None
    solid = None  # 폭 전체를 잇거나 양 벽이 있는 맨 위 행
    while y >= 0:
        row = ink[y]
        # 막대 안: 폭 전체를 잇는 줄(테두리·채움) / 표본 띠에 잉크(무늬) / 양 벽이 다 있음(빈 막대)
        full_ = _contig(row, l, r) >= 0.85 * w or (row[wl].any() and row[wr].any())
        if full_ or row[st].mean() >= 0.12:
            top, gap = y, 0
            if full_:
                solid = y
        else:
            gap += 1
            if gap > 2:
                break
        y -= 1
    if top is None or h - top < 1:
        return None
    if solid is not None and top < solid and not ink[top:solid, l:r + 1].any(axis=1).all():
        top = solid  # 빈 줄 너머 꼭대기에 붙은 글자(값 라벨)는 막대가 아님
    # 꼭대기 테두리: 맨 위(안티앨리어싱 한 행은 건너뜀)와 같은 색으로 이어지는 행들. 5행 넘으면 테두리 없는 채움
    rowcol = lambda yy: sub[yy, st].astype(float).mean(axis=0)
    dk = lambda yy: max(0.0, 255.0 - float(g[yy, st].mean()))
    k0 = 1 if top + 1 < h and dk(top) < 0.6 * dk(top + 1) else 0
    lc = rowcol(min(h - 1, top + k0))
    t = k0 + 1
    while top + t < h and t < 6 and np.sqrt(((rowcol(top + t) - lc) ** 2).sum()) < 60:
        t += 1
    if top + t >= h or t >= 6:  # 테두리 없는 채움: 맨 위 행이 덮인 비율만큼
        full = max(dk(min(h - 1, top + 3)), 1.0)
        top_px = top + 1 - min(1.0, dk(top) / full) if k0 else float(top)
        t = 0
    else:
        din = dk(top + t)
        wts = np.array([abs(dk(top + k) - din) for k in range(t)])
        top_px = float((np.arange(top, top + t) * wts).sum() / wts.sum()) + 0.5 if wts.sum() > 0 else top + t / 2
    t_in = top + t
    segs = _segments_of(sub, ink, g, l, r, st, t_in, h, top_px)
    err = _errbar(sub, ink, l, r, st, top, t_in, h, segs)
    return {"top_outer": top, "top_px": top_px, "segs": segs, "err": err, "w": w}


def _row_feat(sub, ink, st, y):
    px = sub[y, st].astype(float)
    m = ink[y, st]
    f = m.mean()
    col = px[m].mean(axis=0) if m.any() else np.array([255.0, 255, 255])
    return np.array([f * 255, *col])


def _period(x):
    """행마다 잉크 비율의 반복 주기(무늬 간격, 픽셀) — 자기상관 첫 봉우리. 없으면 0"""
    x = np.asarray(x, float)
    if len(x) < 24 or x.std() < 1e-6:
        return 0
    k_ = min(61, len(x) // 2 * 2 - 1)  # 조각이 바뀌는 계단은 빼고 무늬 출렁임만
    x = x - np.convolve(np.pad(x, k_ // 2, mode="edge"), np.ones(k_) / k_, "valid")
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    for k in range(3, min(60, len(x) // 2)):
        if ac[k] > 0.3 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]:
            return min(40, k + 1)
    return 0


def _segments_of(sub, ink, g, l, r, st, t_in, h, top_px):
    """꼭대기 테두리 아래 ~ 바닥: 채움이 바뀌는 곳(가로 테두리선 또는 색·무늬 변화)으로 조각을 나눈다. 아래 → 위"""
    w = r - l + 1
    ys = np.arange(t_in, h)
    if len(ys) < 1:
        return [{"y0": top_px, "y1": float(h), "sig": fill_sig(sub[h - 1:h, st], ink[h - 1:h, st])}]
    # 가로 테두리선 행: 막대 폭 거의 전부를 잇는 어두운 줄인데 위·아래 3px 은 그렇지 않음
    full = np.array([_contig(ink[y], l, r) >= 0.85 * w for y in ys])
    feat = np.array([_row_feat(sub, ink, st, y) for y in ys])
    gm = np.array([float(g[y, st].mean()) for y in range(h)])
    up = np.array([gm[max(0, y - 4)] for y in ys])
    dn = np.array([gm[min(h - 1, y + 4)] for y in ys])
    darkrow = (gm[ys] < 120) & (gm[ys] < np.minimum(up, dn) - 30)  # 위·아래보다 확 어두운 가는 줄
    line = full & darkrow
    lines = []
    for s, e in D.runs(line):
        if e - s + 1 <= 4:
            lines.append((s, e))
    keep = ~line
    win = max(int(np.clip(0.25 * w, 6, 14)), _period(feat[:, 0]))  # 무늬 한 주기보다 넓게
    n = len(ys)

    def wmean(a_, b_):
        idx = [i for i in range(max(0, a_), min(n, b_)) if keep[i]]
        return feat[idx].mean(axis=0) if idx else None

    def dist(p, q):
        if p is None or q is None:
            return 0.0
        c = np.sqrt(((p[1:] - q[1:]) ** 2).sum()) * min(p[0], q[0]) / 255.0
        return float(c + abs(p[0] - q[0]) * 0.9)
    score = np.zeros(n)
    for i in range(1, n):
        score[i] = dist(wmean(i - win, i), wmean(i, i + win))
    cuts = []  # (행 위치(경계, sub 좌표), 종류)
    for s, e in lines:
        c = (s + e) / 2
        if s == 0:
            continue  # 꼭대기 테두리에 붙은 줄
        if dist(wmean(s - win, s), wmean(e + 1, e + 1 + win)) >= 45 or e >= n - 2:
            if e < n - 2:
                cuts.append((t_in + c + 0.5, "line"))
    thr = 55
    i = 1
    while i < n:
        if score[i] >= thr and score[i] == score[max(0, i - win):i + win].max():
            if not any(abs((t_in + i) - c) < win for c, _ in cuts):
                cuts.append((float(t_in + i), "change"))
            i += win
        else:
            i += 1
    cuts.sort()
    bounds = [top_px] + [c for c, _ in cuts] + [float(h)]
    lineset = {c for c, k_ in cuts if k_ == "line"}

    def sig_of(y_a, y_b):
        ra, rb = int(math.ceil(y_a)) + 1, int(math.floor(y_b)) - 1
        if rb < ra:
            ra, rb = int(y_a), max(int(y_a), int(y_b) - 1)
        rows = [y for y in range(max(t_in, ra), min(h, rb + 1)) if not line[y - t_in]] or list(range(max(0, ra), min(h, rb + 1))) or [min(h - 1, int(y_a))]
        px = sub[rows][:, st].reshape(-1, 3)
        im = ink[rows][:, st].reshape(-1)
        wall = np.concatenate([sub[rows][:, l:l + 2].reshape(-1, 3), sub[rows][:, r - 1:r + 1].reshape(-1, 3)])
        return fill_sig(px, im, wall)
    segs = [{"y0": float(y_b), "y1": float(y_a), "sig": sig_of(y_a, y_b)} for y_a, y_b in zip(bounds[:-1], bounds[1:])]
    segs.reverse()  # 아래 → 위 (y0 = 아래 경계, y1 = 위 경계)
    # 막대 안 글자 상자 (같은 채움 사이에 낀 짧은 조각) → 합침
    i = 1
    while i < len(segs) - 1:
        a_, b_, c_ = segs[i - 1], segs[i], segs[i + 1]
        hb = b_["y0"] - b_["y1"]
        if sig_dist(a_["sig"], c_["sig"]) < 30 and hb < 0.6 * ((a_["y0"] - a_["y1"]) + (c_["y0"] - c_["y1"])):
            segs[i - 1:i + 2] = [{"y0": a_["y0"], "y1": c_["y1"], "sig": a_["sig"] if a_["y0"] - a_["y1"] >= c_["y0"] - c_["y1"] else c_["sig"]}]
        else:
            i += 1
    # 같은 채움이 이웃하면 합침 (무늬 잡음으로 생긴 경계)
    j = 1
    while j < len(segs):
        p_, q_ = segs[j - 1]["sig"], segs[j]["sig"]
        same_hatch = (p_["kind"] == q_["kind"] == "hatch" and segs[j - 1]["y1"] not in lineset and
                      np.sqrt(((np.array(D.hex2rgb(p_["color"]), float) - D.hex2rgb(q_["color"])) ** 2).sum()) < 45)  # 굵은 무늬의 위상 차
        if sig_dist(p_, q_) < 55 or same_hatch:
            segs[j - 1:j + 1] = [{"y0": segs[j - 1]["y0"], "y1": segs[j]["y1"], "sig": sig_of(segs[j]["y1"], segs[j - 1]["y0"])}]
        else:
            j += 1
    # 얇은 조각(4px, 무늬 옆이면 6px 미만): 색 방향이 같은 이웃의 가장자리(테두리·무늬 끝)면 붙임, 2px 미만은 무조건
    j = 0
    while j < len(segs) and len(segs) > 1:
        s_ = segs[j]
        hgt = s_["y0"] - s_["y1"]
        nb = [k for k in (j - 1, j + 1) if 0 <= k < len(segs)]
        cd = {k: _absorb_cos(segs[k]["sig"]["color"], s_["sig"]["color"]) for k in nb}
        k = max(nb, key=lambda k_: cd[k_])
        if s_["sig"]["kind"] == "empty" and hgt < 8.0:  # 무늬 선 사이의 빈 줄
            hn = [k_ for k_ in nb if segs[k_]["sig"]["kind"] == "hatch"]
            if hn:
                k = hn[0]
                cd[k] = 1.0
        if hgt < 2.0 or (hgt < 4.0 and cd[k] > 0.97) or (hgt < 8.0 and cd[k] > 0.97 and segs[k]["sig"]["kind"] == "hatch"):
            lo, hi = min(j, k), max(j, k)
            segs[lo:hi + 1] = [{"y0": segs[lo]["y0"], "y1": segs[hi]["y1"], "sig": segs[k]["sig"]}]
            j = max(0, lo - 1)
        else:
            j += 1
    for j in range(1, len(segs)):
        segs[j]["y0"] = segs[j - 1]["y1"]
    return segs


def _absorb_cos(c1, c2):
    """두 색의 '잉크 방향'(흰색에서 뺀 값) 코사인 — 흰색과 섞인 같은 색이면 1 에 가깝다"""
    u = 255.0 - np.array(D.hex2rgb(c1), float)
    v = 255.0 - np.array(D.hex2rgb(c2), float)
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))


def _errbar(sub, ink, l, r, st, top, t_in, h, segs):
    """막대 가운데 위·아래로 뻗은 가는 줄기(+ 끝 가로 막대) → 오차 끝 픽셀. 없으면 {}"""
    w = r - l + 1
    cx = (l + r) / 2
    hw = max(1, int(round(0.06 * w)))
    cc = np.arange(int(cx) - hw, int(math.ceil(cx)) + hw + 1)
    cc = cc[(cc >= l) & (cc <= r)]
    out = {}
    # 위
    y = top - 1
    run = []
    gap = 0
    while y >= 0:
        if ink[y, cc].any() and ink[y, st].mean() < 0.3:
            run.append(y)
            gap = 0
        elif ink[y, cc].any():
            run.append(y)  # 끝 가로 막대(캡)가 표본 띠까지 넓을 때
            gap = 0
        else:
            gap += 1
            if gap > 1:
                break
        y -= 1
        if len(run) > 0.95 * h:
            break
    wide_rows = sum(1 for yy in run if ink[yy, st].mean() >= 0.3)
    if len(run) >= 3 and run[0] >= top - 2 and wide_rows > max(4, 0.25 * len(run)):
        run = []  # 가운데 줄기가 아니라 막대 폭 전체에 잉크(겹친 범례 상자·글자) — 오차 막대 아님
    if len(run) >= 3 and run[0] >= top - 2:
        ytop = run[-1]
        mid = run[len(run) // 2]
        stem = max(1, _contig(ink[mid], max(0, int(cx) - 3 * hw - 2), min(ink.shape[1] - 1, int(cx) + 3 * hw + 2)))
        if stem > max(5, 0.25 * w):
            return out  # 가는 줄기가 아님 (막대 꼭대기에 겹친 범례 견본·글자)
        capw = max(5, 2.5 * stem)
        cap = [yy for yy in range(ytop, min(top, ytop + 4)) if _contig(ink[yy], max(0, l - w // 2), min(ink.shape[1] - 1, r + w // 2)) >= capw]
        out["hi_px"] = float(np.mean(cap)) + 0.5 if cap else float(ytop)
        out["cap"] = bool(cap)
        # 아래: 맨 위 조각 안에서 가운데만 다른 색(줄기)
        y = t_in
        last = None
        while y < h:
            c_px = sub[y, cc].astype(float)
            s_px = sub[y, st].astype(float).mean(axis=0)
            diff = np.sqrt(((c_px - s_px) ** 2).sum(axis=1)).max()
            if ink[y, cc].any() and diff > 60:
                last = y
                y += 1
            else:
                break
        if last is not None and last - t_in >= 2:
            cap = [yy for yy in range(max(t_in, last - 3), last + 1)
                   if capw <= _contig(ink[yy], l, r) < 0.85 * w]
            out["lo_px"] = float(np.mean(cap)) + 0.5 if cap else float(last + 1)
            out["sym"] = False
        else:
            out["sym"] = True
    return out


def _neg_bars(a, ax, orient):
    """기준선(0) 반대쪽으로 뻗은 막대(음수): 그림을 값축 방향으로 뒤집어 같은 코드로 찾은 뒤 좌표를 되돌린다.
    가로 막대는 기준선 왼쪽, 세로 막대는 기준선 아래. ax["neg_to"] = 그쪽 그림 영역 끝(없으면 그림 끝)."""
    H, W = a.shape[:2]
    xa, ya, box = ax["xaxis"], ax["yaxis"], ax["box"]
    if orient == "h":
        f = np.ascontiguousarray(a[:, ::-1])
        far = float(ax.get("neg_to", 2))
        bb = {"x0": box["y0"], "x1": box["y1"] - 1, "y0": far, "y1": float(ya["x"])}
        bs = detect_bars(rot(f), bb, ya["c0"], left_col=box["y0"])
        n = W
    else:
        f = np.ascontiguousarray(a[::-1])
        far = float(ax.get("neg_to", H - 2))
        bb = {"x0": box["x0"], "x1": box["x1"], "y0": H - far, "y1": H - float(xa["y"])}
        bs = detect_bars(f, bb, H - 1 - xa["r1"], left_col=ya["c1"])
        n = H
    out = []
    for b in bs:
        if b.get("zero"):
            continue
        b["top_outer"], b["top_px"] = n - b["top_outer"], n - b["top_px"]
        for sg in b["segs"]:
            sg["y0"], sg["y1"] = n - sg["y0"], n - sg["y1"]
        for k in ("hi_px", "lo_px"):
            if b["err"].get(k) is not None:
                b["err"][k] = n - b["err"][k]
        b["neg"] = True
        out.append(b)
    return out


def find_bars(a, ax, orient=None, two_sided=False):
    """축 검출 결과(digitize.detect_axes)로 세로·가로 막대를 찾는다. orient=None 이면 막대가 더 잘 서는 쪽.
    two_sided=True 면 기준선 반대쪽(음수) 막대도 찾는다.
    → (orient 'v'|'h', 막대 목록 — 원래 그림 좌표로 바꾼 것)"""
    H, W = a.shape[:2]
    xa, ya, box = ax["xaxis"], ax["yaxis"], ax["box"]
    res = {}
    if orient in (None, "v"):
        res["v"] = detect_bars(a, box, xa["r0"], left_col=ya["c1"])
    if orient in (None, "h"):
        b = rot(a)
        bb = {"x0": box["y0"], "x1": box["y1"] - 1, "y0": W - box["x1"], "y1": W - box["x0"]}
        res["h"] = detect_bars(b, bb, W - 1 - ya["c1"], left_col=box["y0"])  # 돌린 좌표: 원래 x = W - y', y = x'

    def score(bs):
        return sum(1 for k in bs if k["r"] - k["l"] >= 5 and k["segs"] and k["segs"][0]["y0"] - k["top_px"] >= 3)
    if orient is None:
        orient = "h" if score(res.get("h", [])) > score(res.get("v", [])) else "v"
    bs = res.get(orient, [])
    if two_sided:
        neg = [b for b in _neg_bars(a, ax, orient) if not any(min(b["r"], o["r"]) - max(b["l"], o["l"]) > 0.5 * (b["r"] - b["l"]) for o in bs)]
        bs = sorted(bs + neg, key=lambda b: b["l"])
    return orient, bs


def value_px(orient, W, p):
    """막대 좌표계(세로 막대 기준 y) → 원래 그림의 값축 픽셀 (세로: y, 가로: x)"""
    return p if orient == "v" else W - p


def _cut_long(m, lh, lv):
    """가로 lh·세로 lv 픽셀 이상 이어진 줄을 지운 마스크"""
    m = m.copy()
    keep = m.copy()
    for y in range(m.shape[0]):
        for s_, e in D.runs(keep[y]):
            if e - s_ + 1 >= lh:
                m[y, s_:e + 1] = False
    for x in range(m.shape[1]):
        for s_, e in D.runs(keep[:, x]):
            if e - s_ + 1 >= lv:
                m[s_:e + 1, x] = False
    return m


def legend_patches(a, avoid=(), region=None):
    """범례 견본: 테두리 있는(또는 꽉 찬) 작은 사각형 + 바로 오른쪽에 글자. → [{box, sig, text}] (text = 이름 글자 범위)
    avoid: 막대 상자들 — 그 안의 사각형(막대 속 글자 상자)은 뺀다. region=[x0,y0,x1,y1] 이면 그 안에서만."""
    H, W = a.shape[:2]
    rx0, ry0, rx1, ry1 = [int(round(v)) for v in (region or (0, 0, W, H))]
    rx0, ry0, rx1, ry1 = max(0, rx0), max(0, ry0), min(W, rx1), min(H, ry1)
    sub = a[ry0:ry1, rx0:rx1]
    if sub.size == 0:
        return []
    ink = ink_mask(sub)
    dark = D.gray(sub) < 170
    lab, n = _label(_cut_long(ink, 80, 70))  # 범례 틀에 붙은 견본: 긴 직선(틀)을 끊고 본다
    try:
        from scipy import ndimage
        objs = ndimage.find_objects(lab)
    except ImportError:
        objs = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            objs.append((slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)))
    pats = []
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        Y0_, Y1_, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if not (7 <= Y1_ - Y0_ <= 130 and 10 <= x1 - x0 <= 400):
            continue
        # 위아래로 맞붙은 견본(빽빽한 범례): 오른쪽 글자 줄 사이에서 나눈다
        tb = dark[Y0_:Y1_, min(dark.shape[1], x1 + 3):min(dark.shape[1], x1 + 3 + 4 * min(40, Y1_ - Y0_))]
        txt = tb[:, tb.mean(axis=0) < 0.7].any(axis=1) if tb.size else np.zeros(Y1_ - Y0_, bool)  # 범례 틀 세로줄은 빼고
        tl = [r_ for r_ in D.runs(txt) if r_[1] - r_[0] >= 4]
        cuts = [Y0_]
        for (s1, e1), (s2, e2) in zip(tl[:-1], tl[1:]):
            c = Y0_ + (e1 + s2 + 1) // 2
            if c - cuts[-1] >= 7 and Y1_ - c >= 7:
                cuts.append(c)
        cuts.append(Y1_)
        for y0, y1 in zip(cuts[:-1], cuts[1:]):
            h, w = y1 - y0, x1 - x0
            if not (7 <= h <= 60 and max(10, 0.9 * h) <= w <= 6 * h):
                continue
            m = (lab[y0:y1, x0:x1] == i + 1) | ink[y0:y1, x0:x1]  # 테두리는 끊기 전 잉크로 (맞붙은 견본은 세로 테두리가 길어 지워짐)
            e0, e1 = max(0, x0 - 3), min(ink.shape[1], x1 + 3)
            mx = ink[y0:y1, e0:e1]  # 지워진 세로 테두리가 바로 바깥 열에 있을 수 있음
            lcol = max(mx[:, k].mean() for k in range(0, min(4, mx.shape[1])))
            rcol = max(mx[:, -k - 1].mean() for k in range(0, min(4, mx.shape[1])))
            border = min(m[0].mean() if len(cuts) == 2 or y0 == Y0_ else 1.0, m[-1].mean() if len(cuts) == 2 or y1 == Y1_ else 1.0,
                         lcol, rcol)
            if lcol >= 0.8 and x0 - e0 and not m[:, 0].mean() >= 0.8:  # 바깥 열의 테두리까지 견본 상자에 넣음
                x0 = e0 + next(k for k in range(0, 4) if mx[:, k].mean() >= 0.8)
            if rcol >= 0.8 and not m[:, -1].mean() >= 0.8:
                x1 = e1 - next(k for k in range(0, 4) if mx[:, -k - 1].mean() >= 0.8)
            w = x1 - x0
            if border < 0.8 and m.mean() < 0.9:
                continue
            X0, Y0 = x0 + rx0, y0 + ry0
            if X0 <= 1 or Y0 <= 1 or X0 + w >= W - 1 or Y0 + h >= H - 1:  # 그림 가장자리에서 잘린 견본(잘린 범례)
                continue
            if any(b[0] - 2 <= X0 and b[1] - 2 <= Y0 and X0 + w <= b[2] + 2 and Y0 + h <= b[3] + 2 for b in avoid):
                continue
            # 오른쪽 글자: 견본 높이 3배 안에 어두운 픽셀
            right = dark[y0:y1, x1 + 2:min(dark.shape[1], x1 + 2 + 3 * h)]
            if right.size == 0 or right.any(axis=0).sum() < 2:
                continue
            tr_ = D.runs(right[:, right.mean(axis=0) < 0.7].any(axis=1)) if right.size else []
            th_ = max((e_ - s_ + 1 for s_, e_ in tr_), default=0)
            if th_ < 4 or h > 2.0 * th_ + 4:  # 견본은 글자 줄 높이 정도 (막대 끝에 붙은 % 주석은 막대가 훨씬 큼)
                continue
            pats.append([X0, Y0, X0 + w, Y0 + h, i + 1])
    out = []
    for X0, Y0, X1, Y1, cid in pats:
        h = Y1 - Y0
        # 글자 범위: 다음 견본이나 넓은 빈칸(견본 높이)까지
        stop = min([p[0] for p in pats if p[0] > X1 and abs(p[1] - Y0) < h / 2] + [W])
        cols = D.gray(a[Y0:Y1, X1 + 2:stop - 1]) < 170
        rs = D.merge(D.runs(cols.any(axis=0)), max(3, int(0.7 * h)))
        if not rs:
            continue
        tx1 = X1 + 2 + rs[0][1] + 1
        # 안쪽 표본 (테두리 빼고)
        t = max(2, int(round(0.18 * min(h, X1 - X0))))
        inner = a[Y0 + t:Y1 - t, X0 + t:X1 - t]
        if inner.size == 0:
            continue
        wall = np.concatenate([a[Y0:Y0 + 2, X0:X1].reshape(-1, 3), a[Y1 - 2:Y1, X0:X1].reshape(-1, 3)])
        sig = fill_sig(inner.reshape(-1, 3), ink_mask(inner).reshape(-1), wall)
        out.append({"box": [X0, Y0, X1, Y1], "sig": sig, "text": [X1 + 2, Y0, tx1, Y1]})
    return out


def analyze(a, ax, calib_val, orient=None, two_sided=False):
    """막대 검출 + 값 계산. calib_val = 값축 보정 {"p1": 픽셀, "p2": 픽셀, "v1", "v2", "log"} (픽셀은 값축 방향 좌표 하나).
    → {orient, bars: [{l, r, cx, top_px, value, segs: [{y0, y1, sig, top, value}], err}]}"""
    H, W = a.shape[:2]
    orient, bs = find_bars(a, ax, orient, two_sided)
    f = (lambda v: math.log10(v)) if calib_val.get("log") else (lambda v: v)
    p1, p2 = float(calib_val["p1"]), float(calib_val["p2"])
    f1, f2 = f(calib_val["v1"]), f(calib_val["v2"])

    def val(p_bar):  # 막대 좌표 → 값
        p = value_px(orient, W, p_bar)
        x = f1 + (p - p1) * (f2 - f1) / (p2 - p1)
        return 10 ** x if calib_val.get("log") else x
    for b in bs:
        b["value"] = val(b["top_px"])
        prev = 0.0
        for sgm in b["segs"]:
            sgm["top"] = val(sgm["y1"])
            sgm["value"] = sgm["top"] - prev
            prev = sgm["top"]
        e = b["err"]
        if e.get("hi_px") is not None:
            e["hi"] = val(e["hi_px"])
            e["lo"] = val(e["lo_px"]) if e.get("lo_px") is not None else 2 * b["value"] - e["hi"]
            e["plus"], e["minus"] = e["hi"] - b["value"], b["value"] - e["lo"]
    return {"orient": orient, "bars": bs}


def assign_series(bars, legend=()):
    """조각 채움 모양 → 계열. 범례 견본이 있으면 가장 닮은 견본, 없으면 비슷한 것끼리 묶는다.
    → 계열 목록 [{name, sig, legend(번호 또는 None)}], 각 조각에 sgm['series'] = 계열 번호"""
    series = []
    segs = sorted((sgm for b in bars for sgm in b["segs"]), key=lambda q: -(q["y0"] - q["y1"]))  # 큰 조각부터
    for sgm in segs:
        best = None
        for i, s in enumerate(series):
            d = sig_dist(s["sig"], sgm["sig"])
            if d < 70 and (best is None or d < best[0]):
                best = (d, i)
        if best is None and sgm["y0"] - sgm["y1"] < 6 and sgm["sig"]["kind"] != "empty":
            # 얇은 조각은 무늬가 안 보여 꽉 찬 색처럼 나온다 → 색 방향이 같은 계열
            cs = [(_absorb_cos(s["sig"]["color"], sgm["sig"]["color"]), i) for i, s in enumerate(series) if s["sig"]["kind"] != "empty"]
            if cs and max(cs)[0] > 0.985:
                best = (0, max(cs)[1])
        if best is None:
            series.append({"sig": sgm["sig"], "n": 1, "legend": None})
            sgm["series"] = len(series) - 1
        else:
            series[best[1]]["n"] += 1
            sgm["series"] = best[1]
    # 계열 순서 = 쌓인 순서(아래 → 위), 같으면 처음 나온 막대 순
    key = {}
    for bi, b in enumerate(bars):
        for k, sgm in enumerate(b["segs"]):
            key[sgm["series"]] = min(key.get(sgm["series"], (99, 99)), (k, bi))
    order = sorted(range(len(series)), key=lambda i: key.get(i, (99, 99)))
    new = {old_: n for n, old_ in enumerate(order)}
    series = [series[i] for i in order]
    for b in bars:
        for sgm in b["segs"]:
            sgm["series"] = new[sgm["series"]]
    pairs = sorted((sig_dist(s["sig"], p["sig"]) + p.get("penalty", 0), i, j) for i, s in enumerate(series) for j, p in enumerate(legend))
    used_s, used_l = set(), set()
    for d, i, j in pairs:
        if d < 120 + legend[j].get("penalty", 0) and i not in used_s and j not in used_l:
            series[i]["legend"] = j
            used_s.add(i)
            used_l.add(j)
    return series


def merge_washed(bars, n_legend):
    """반투명 범례 상자에 가린 쌓인 막대 꼭대기: 맨 위 조각이 바로 아래 조각과 같은 색 방향의 옅은 색이고, 다른 막대에는 없는 채움이며,
    찾은 채움 수가 범례(VLM) 항목 수보다 많으면 → 아래 조각에 붙인다. 붙인 막대 수를 돌려준다."""
    if not n_legend or len(bars) < 3:
        return 0
    import copy
    prov = assign_series(copy.deepcopy(bars))
    if len(prov) <= n_legend:
        return 0
    n = 0
    for b in bars:
        if len(b["segs"]) < 2:
            continue
        t, u = b["segs"][-1], b["segs"][-2]
        gt_, gu = (float(D.gray(np.array([[D.hex2rgb(x["sig"]["color"])]], float))[0, 0]) for x in (t, u))
        if gt_ < gu + 25 or _absorb_cos(t["sig"]["color"], u["sig"]["color"]) < 0.95:
            continue
        others = [sg for o in bars if o is not b for sg in o["segs"]]
        if any(sig_dist(sg["sig"], t["sig"]) < 70 for sg in others) or not any(sig_dist(sg["sig"], u["sig"]) < 70 for sg in others):
            continue
        u["y1"] = t["y1"]
        b["segs"].pop()
        n += 1
    return n


def one_series_by_color(bars, series, vlm_legend=None):
    """범례 없는 색깔 막대: 범주마다 칠한 막대 하나(고른 간격, 쌓인 조각 없음)면 색은 범주 장식(범주별·양수/음수 색칠) → 한 계열로 본다.
    계열로 나누면 표가 범주마다 한 칸만 찬 대각선이 된다. 묶음 막대(간격이 묶음마다 벌어짐)·쌓인 막대·범례가 있는 그림·빈 막대는 그대로."""
    if len(series) < 2 or len(bars) < 3 or vlm_legend or any(s.get("legend") is not None for s in series):
        return False
    if any(len(b["segs"]) != 1 or b.get("zero") for b in bars):
        return False
    if any(b.get("open") or b["segs"][0]["sig"]["kind"] == "empty" for b in bars):
        return False  # 테두리만 있는 빈 막대는 테두리 색이 계열을 뜻하는 경우가 많음 → 그대로
    cs = sorted(b["cx"] for b in bars)
    d = np.diff(cs)
    ws = [b["r"] - b["l"] for b in bars]
    if d.min() <= 0 or d.std() > 0.08 * d.mean() or d.min() - max(ws) < 0.15 * float(np.median(ws)):
        return False  # 고르지 않은 간격(묶음) 또는 맞붙은 막대
    return True


def axis_break(a, ax):
    """값축이 끊긴 그래프(// 표시): 그림 영역 위쪽 바깥에, 틈을 두고 같은 세로축선이 다시 이어지면 끊김.
    → {"y": 끊긴 곳 픽셀, "side": "top"} 또는 None. (위 구간은 눈금이 달라 따로 보정해야 함)"""
    H, W = a.shape[:2]
    ya, box = ax["yaxis"], ax["box"]
    h = box["y1"] - box["y0"]
    y0 = int(ya["r0"])  # 세로축선의 실제 위 끝
    lo = max(0, int(y0 - 0.35 * h))
    if y0 - lo < 8:
        return None
    for cols in ((ya["c0"], ya["c1"]), (int(box["x1"]) - 1, int(box["x1"]) + 1)):
        col = (D.gray(a[lo:y0 + 2, max(0, cols[0]):min(W, cols[1] + 1)]) < 140).any(axis=1)
        rs = [r_ for r_ in D.runs(col) if r_[1] - r_[0] + 1 >= 8]
        # 맨 아래(그림 영역 위 끝) 바로 위에 2px 넘는 빈 틈, 그 위에 다시 축선
        if rs and rs[-1][1] - rs[-1][0] >= 14:  # 위 구간도 가는 세로선이어야 (글자·범례가 아님)
            ra, rb = lo + rs[-1][0], lo + rs[-1][1] + 1
            lft = (D.gray(a[ra:rb, max(0, cols[0] - 7):max(0, cols[0] - 2)]) < 140).any(axis=1).mean()
            if lft > 0.4:
                rs = []
        if rs and rs[-1][1] - rs[-1][0] >= 14 and rs[-1][1] < len(col) - 3 and rs[-1][1] >= len(col) - 3 - 0.15 * h:
            g0, g1 = lo + rs[-1][1] + 1, y0 + 2  # 틈 (위 축선 끝 ~ 그림 영역 위)
            r0_, r1_ = max(0, g0 - 4), g1 + 2
            near = D.gray(a[r0_:r1_, max(0, cols[0] - 12):min(W, cols[1] + 13)]) < 140
            wide = (D.gray(a[r0_:r1_, max(0, cols[0] - 40):min(W, cols[1] + 41)]) < 140).sum(axis=1)
            L, R = near[:, :10].any(axis=1), near[:, -10:].any(axis=1)
            if (L & R & (wide < 25)).sum() >= 2:  # 축을 가로지르는 짧은 // 끊김 표시 (긴 가로선·범례 틀은 아님)
                return {"y": float(y0), "side": "top"}
    return None


def slanted_labels(a, ax, bars):
    """비스듬히(회전해) 쓴 범주 글자: 축 아래 글자 덩어리마다 위 끝(축에 가까운 끝)이 어느 막대 밑인지로 짝짓는다.
    글자가 비스듬하지 않으면 None. → 막대마다 글자 상자 [x0, y0, x1, y1] 또는 None"""
    H, W = a.shape[:2]
    xa = ax["xaxis"]
    top = int(xa["r1"]) + 3
    bot = min(H, top + int(0.3 * H))
    band = D.gray(a[top:bot]) < 150
    if not band.any() or len(bars) < 2:
        return None
    grp = D._shift_and(band, 2, "or")  # 글자끼리 잇기
    lab, n = _label(grp)
    cs = [b["c"] for b in bars]
    sp = float(np.median(np.diff(cs))) if len(cs) > 1 else 50.0
    found, slant = {}, 0
    for i in range(1, n + 1):
        ys, xs = np.nonzero((lab == i) & band)
        if len(ys) < 15 or ys.min() > 0.08 * H:  # 축 바로 밑에서 시작하는 글자만
            continue
        hgt, wid = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        r = abs(np.corrcoef(xs, ys)[0, 1]) if hgt > 3 and wid > 3 else 0
        if r > 0.6 and hgt > 0.35 * wid:
            slant += 1
        up = ys <= ys.min() + 0.25 * hgt
        ax_ = float(xs[up].mean())  # 위 끝
        j = int(np.argmin([abs(c - ax_) for c in cs]))
        if abs(cs[j] - ax_) <= 0.6 * sp and (j not in found or found[j][4] < len(ys)):
            found[j] = [int(xs.min()), top + int(ys.min()), int(xs.max()) + 1, top + int(ys.max()) + 1, len(ys)]
    if slant < max(2, 0.5 * len(found)):
        return None
    return [found[j][:4] if j in found else None for j in range(len(bars))]


# ── 값 라벨만 있는 막대 (값축 눈금 없음) ─────────────────────────────────────
def label_axes(a, ax, orient):
    """값 라벨로 보정할 때 쓸 그림 영역: 범주축선(막대가 서는 선)을 따라 넓힌 상자. 눈금 없는 웹 차트는 아래 가로축을 잘못 잡기 쉬움"""
    H, W = a.shape[:2]
    xa, ya = dict(ax["xaxis"]), dict(ax["yaxis"])
    if orient == "h":
        box = {"x0": float(ya["x"]), "x1": float(W - 2), "y0": float(ya["r0"]), "y1": float(ya["r1"])}
        xa["y"] = box["y1"]
        # 범주축선 아래 끝을 지나는 가는 가로선(바닥 축선)이 있으면 그 양 끝이 그림 영역 좌우 (막대가 0 양쪽으로 뻗는 그림)
        g = D.gray(a)
        y_ = int(round(ya["r1"]))
        best = None
        for yy in range(max(0, y_ - 3), min(H, y_ + 4)):
            for s_, e_ in D.runs(g[yy] < 200):
                if s_ <= ya["x"] <= e_ + 1 and e_ - s_ > 0.3 * W and (best is None or e_ - s_ > best[1] - best[0]):
                    best = (s_, e_)
        if best:
            ax_ = dict(ax)
            ax_["neg_to"] = float(best[0])
            box["x1"] = float(best[1] + 1)
            box["plot_x0"] = float(best[0])
            return dict(ax_, box=box, xaxis=xa, yaxis=ya)
    else:
        box = {"x0": float(xa["c0"]), "x1": float(xa["c1"]), "y0": 2.0, "y1": float(xa["y"])}
        ya["x"] = box["x0"]
    return dict(ax, box=box, xaxis=xa, yaxis=ya)


def value_label_boxes(a, bars, orient):
    """막대 끝 바로 바깥(세로: 위, 가로: 오른쪽)에 찍힌 값 글자 상자 → 막대마다 [x0, y0, x1, y1](원래 그림 좌표) 또는 None"""
    H, W = a.shape[:2]
    s = a if orient == "v" else rot(a)
    Hs, Ws = s.shape[:2]
    dark = D.gray(s) < 170
    out = []
    for b in bars:
        l, r, w = int(b["l"]), int(b["r"]), b["r"] - b["l"]
        g = 3 if orient == "v" else max(7, int(0.25 * w))  # 글자 사이 틈 (가로 막대는 돌린 그림에서 낱말 사이 — '−€ 75.000')
        c0, c1 = (max(0, int(l - 0.5 * w)), min(Ws, int(r + 0.5 * w))) if orient == "v" else (l, r)
        d = 1 if b.get("neg") else -1  # 음수 막대는 끝이 기준선 반대쪽 → 그쪽으로 찾음
        y = int(b["top_outer"]) - 1 if d < 0 else int(math.ceil(b["top_outer"]))
        lim = max(0, int(b["top_outer"] - 0.4 * Hs)) if d < 0 else min(Hs - 1, int(b["top_outer"] + 0.4 * Hs))
        start = None
        gap = 0
        first_gap = max(12, int(0.04 * Hs))
        end = None
        while (y >= lim) if d < 0 else (y <= lim):
            has = dark[y, c0:c1].any()
            if has:
                if start is None:
                    start = y
                end, gap = y, 0
            else:
                gap += 1
                if start is None and gap > first_gap:
                    break
                if start is not None and gap > g:
                    break
            y += d
        if start is None or abs(start - end) + 1 < 3:
            out.append(None)
            continue
        ra, rb = min(start, end), max(start, end)
        cols = np.nonzero(dark[ra:rb + 1, c0:c1].any(axis=0))[0]
        if len(cols) < 3:
            out.append(None)
            continue
        bx = [c0 + int(cols[0]), ra, c0 + int(cols[-1]) + 1, rb + 1]
        out.append([float(v) for v in (bx if orient == "v" else [W - bx[3], bx[0], W - bx[1], bx[2]])])
    return out


def fit_labels(pairs, base_px):
    """(막대 끝 픽셀, 인쇄값) 짝 → 값축 직선. 막대가 0 에서 서는지(바닥선 = 0)도 맞으면 함께 쓴다. → (p1, v1, p2, v2, 잔차 px, 버린 짝) 또는 None"""
    P = np.array([p for p, _ in pairs], float)
    V = np.array([v for _, v in pairs], float)
    dropped = []

    def fit(P_, V_):
        if len(P_) < 2 or np.ptp(V_) == 0:
            return None
        k, c = np.polyfit(V_, P_, 1)  # 픽셀 = k·값 + c
        if k == 0:
            return None
        return k, c, float(np.abs(P_ - (k * V_ + c)).max())
    while True:
        best = None
        P0, V0 = np.append(P, base_px), np.append(V, 0.0)
        f0 = fit(P0, V0)
        f1 = fit(P, V) if np.ptp(P) >= 15 else None
        if f0 and f0[2] <= max(1.5, (f1[2] + 0.5) if f1 else 1.5):
            best = f0
        elif f1:
            best = f1
        if best is None:
            return None
        if best[2] <= 2.0 or len(P) <= 3:
            break
        k, c, _ = best
        i = int(np.argmax(np.abs(P - (k * V + c))))
        dropped.append((float(V[i]), float(P[i])))
        P, V = np.delete(P, i), np.delete(V, i)
    k, c, res = best
    if res > 3.0:
        return None
    v1, v2 = 0.0, float(V[np.argmax(np.abs(V))])
    if v2 == 0:
        return None
    return k * v1 + c, v1, k * v2 + c, v2, res, dropped


def grid_rows(a, box, base):
    """값축 숫자가 없는 세로 막대그래프의 연한 가로 격자선 행 (아래 기준선 제외, 위→아래). 기준선에서 고른 간격일 때만 → [행] 또는 []"""
    x0, x1 = int(box["x0"]) + 1, int(box["x1"])
    if x1 - x0 < 40:
        return []
    g = D.gray(a).astype(float)
    s = a.max(axis=2).astype(int) - a.min(axis=2)
    top = max(0, int(box["y0"] - 0.35 * (base - box["y0"])))
    bg = float(np.median(g[top:int(base), x0:x1]))
    rows = []
    for y in range(top, int(base) - 3):
        ok = s[y, x0:x1] < 30
        if ok.sum() < 0.1 * (x1 - x0):
            continue
        f = ((g[y, x0:x1] < bg - 6) & ok).sum() / ok.sum()
        if f > 0.6:
            rows.append(y)
    lines = [(s_ + e) / 2 for s_, e in D.merge(D.runs(np.isin(np.arange(int(base) + 1), rows)), 1)] if rows else []
    lines = [y for y in lines if base - y > 5]
    if len(lines) < 2:
        return []
    d = np.diff(sorted(lines + [float(base)]))
    if np.std(d) > 0.08 * np.mean(d) + 1:
        return []
    return sorted(lines)
