"""지원하지 않는 그림(3D·극좌표·파이/도넛·히트맵·등고선 등) 알아보기 — VLM 판정 + 픽셀 규칙.

규칙 신호(결정론):
  colorbar  : 그림 밖(또는 옆)에 색이 위아래로 차례로 바뀌는 좁은 띠
  round     : 큰 원(파이·도넛의 테두리, 극좌표의 동심원 격자) — 각도 대부분을 덮는 같은 반지름의 픽셀
  oblique   : 비스듬한 긴 무채색 직선이 두 방향 이상 (3D 축·바닥면 격자)
  filled    : 그림 영역 대부분이 여러 색 칸으로 차 있음 (히트맵·채운 등고선·영상)
  loops     : 같은 색 곡선이 한 열에 두 번 이상 지남 (닫힌 곡선·등고선)
판정(judge): 큰 원(파이·도넛·극좌표)과 컬러바 + 꽉 찬 색 칸(히트맵)은 규칙만으로도 거절.
VLM 이 지원 안 하는 종류라 하면 그 종류에 맞는 규칙 신호로 확인해서 거절하고, 신호가 없으면
축·눈금 글자로 두 축이 보정되는 그림은 거절하지 않고 경고만 한다('기타'는 늘 규칙 신호가 있어야 거절).
"""
import numpy as np

import digitize as D

LABEL = {"pie": "파이·도넛 그래프", "polar": "극좌표 그래프", "3d": "3D 그래프", "heatmap": "히트맵", "contour": "등고선 그래프",
         "other": "2D 직교좌표가 아닌 그림"}
VLM_KINDS = {"pie": "pie", "donut": "pie", "polar": "polar", "radar": "polar", "rose": "polar", "3d": "3d", "surface": "3d",
             "heatmap": "heatmap", "heat": "heatmap", "image": "heatmap", "contour": "contour", "other": "other"}
CODE = "unsupported_chart"


def vlm_kind(t):
    t = str(t or "").strip().lower()
    if not t:
        return None
    if t in VLM_KINDS:
        return VLM_KINDS[t]
    for k, v in VLM_KINDS.items():
        if k in t:
            return v
    return None


def _small(a, maxdim=700):
    H, W = a.shape[:2]
    s = max(H, W) / maxdim
    if s <= 1:
        return a, 1.0
    from PIL import Image
    im = Image.fromarray(a).resize((max(1, int(W / s)), max(1, int(H / s))), Image.BILINEAR)
    return np.asarray(im), s


def _sat(a):
    a = a.astype(np.int16)
    return a.max(axis=2) - a.min(axis=2)


def colorbar(a, box=None):
    """색이 차례로 바뀌는 좁은 세로(또는 가로) 띠 → {"x0","x1","y0","y1"} 또는 None"""
    for t in (False, True):
        m = a.transpose(1, 0, 2) if t else a
        H, W = m.shape[:2]
        q = (m // 24).astype(np.int32)
        key = q[..., 0] * 4096 + q[..., 1] * 64 + q[..., 2]
        nonw = (D.gray(m) < 235) | (_sat(m) > 30)
        cand = []
        for c in range(W):
            col = nonw[:, c]
            rs = D.runs(col)
            if not rs:
                continue
            s, e = max(rs, key=lambda r: r[1] - r[0])
            if e - s + 1 < 0.25 * H:
                continue
            k = key[s:e + 1, c]
            ch = int((k[1:] != k[:-1]).sum())
            nd = len(np.unique(k))
            if nd >= 7 and ch <= 6 * nd and (_sat(m[s:e + 1, c:c + 1]).mean() > 25 or nd >= 10):
                cand.append((c, s, e, nd))
        # 옆으로 이어진 열 6~(8%) 개가 거의 같은 색 줄을 가지면 띠
        for s_, e_ in D.runs(np.isin(np.arange(W), [c for c, *_ in cand])):
            w = e_ - s_ + 1
            if not (5 <= w <= max(8, 0.08 * W)):
                continue
            cols = [x for x in cand if s_ <= x[0] <= e_]
            r0 = int(np.median([x[1] for x in cols]))
            r1 = int(np.median([x[2] for x in cols]))
            mid = (s_ + e_) // 2
            same = np.abs(m[r0:r1 + 1, s_ + 1].astype(int) - m[r0:r1 + 1, mid].astype(int)).max(axis=1)
            if (same < 30).mean() < 0.9:
                continue
            # 띠 양옆은 바탕(띠 밖으로 같은 색이 이어지지 않음)
            if s_ > 2 and nonw[r0:r1 + 1, s_ - 3].mean() > 0.8 and e_ < W - 3 and nonw[r0:r1 + 1, e_ + 3].mean() > 0.8:
                continue
            out = {"x0": s_, "x1": e_ + 1, "y0": r0, "y1": r1 + 1}
            if t:
                out = {"x0": r0, "x1": r1 + 1, "y0": s_, "y1": e_ + 1}
            if box:  # 그림 영역 안쪽 깊숙한 띠(막대 하나)는 아님
                cx, cy = (out["x0"] + out["x1"]) / 2, (out["y0"] + out["y1"]) / 2
                if box["x0"] + 6 < cx < box["x1"] - 6 and box["y0"] + 6 < cy < box["y1"] - 6:
                    continue
            return out
    return None


def round_shape(a):
    """큰 원(테두리)이 있으면 {"r": 반지름 비율, "rings": 원 개수, "fill": 원 안 채움} 또는 None"""
    s, f = _small(a, 500)
    H, W = s.shape[:2]
    g = D.gray(s)
    ink = (g < 235) | (_sat(s) > 40)
    edge = ink & ~(np.roll(ink, 1, 0) & np.roll(ink, -1, 0) & np.roll(ink, 1, 1) & np.roll(ink, -1, 1))
    ys, xs = np.nonzero(edge)
    if len(xs) < 200:
        return None
    best = None
    lab = D._components(ink)
    centers = []
    if lab:
        big = max(lab, key=lambda c: c[3] * c[4])
        centers.append((big[5] + big[3] / 2, big[6] + big[4] / 2))
    centers.append((float(np.median(np.nonzero(ink)[1])), float(np.median(np.nonzero(ink)[0]))))
    for cx0, cy0 in centers:
        for dx in (-3, 0, 3):
            for dy in (-3, 0, 3):
                cx, cy = cx0 + dx, cy0 + dy
                d = np.hypot(xs - cx, ys - cy)
                ang = ((np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi) * 72).astype(int) % 72
                rmin = 0.08 * min(H, W)
                rb = np.round(d).astype(int)
                ok = d >= rmin
                if not ok.any():
                    continue
                cov = np.zeros(int(d.max()) + 3)
                for r in np.unique(rb[ok]):
                    sel = np.abs(d - r) <= 1.2
                    cov[r] = len(np.unique(ang[sel])) / 72
                rings = [r for r in range(len(cov)) if cov[r] >= 0.88 and r >= rmin and cov[r] >= cov[max(0, r - 2):r + 3].max()]
                rings = [r for i, r in enumerate(rings) if i == 0 or r - rings[i - 1] > 3]
                if rings:
                    R = max(rings)
                    key = (len(rings), R)
                    if best is None or key > best[0]:
                        yy, xx = np.mgrid[0:H, 0:W]
                        inside = np.hypot(xx - cx, yy - cy) < R - 2
                        best = (key, {"r": R / min(H, W), "rings": len(rings), "fill": float(ink[inside].mean()) if inside.any() else 0.0,
                                      "cx": cx * f, "cy": cy * f, "R": R * f})
    return best[1] if best else None


def oblique_lines(a):
    """비스듬한(축과 15° 넘게 기운) 긴 무채색 직선 → 서로 다른 방향 수, 선 수"""
    s, f = _small(a, 600)
    H, W = s.shape[:2]
    g = D.gray(s)
    m = (g < 200) & (_sat(s) < 40)
    ys0, xs0 = np.nonzero(m)
    xs, ys = xs0, ys0
    if len(xs) < 100:
        return {"dirs": 0, "lines": 0}
    if len(xs) > 60000:
        idx = np.random.default_rng(0).choice(len(xs), 60000, replace=False)
        xs, ys = xs[idx], ys[idx]
    th = np.deg2rad(np.arange(0, 180, 1.0))
    diag = int(np.hypot(H, W)) + 1
    rho = (xs[:, None] * np.cos(th)[None] + ys[:, None] * np.sin(th)[None]).round().astype(int) + diag
    acc = np.zeros((2 * diag + 1, len(th)), np.int32)
    np.add.at(acc, (rho, np.broadcast_to(np.arange(len(th)), rho.shape)), 1)
    L = 0.25 * min(H, W)
    lines = []
    A = acc.copy()
    for _ in range(40):
        r, t = np.unravel_index(int(np.argmax(A)), A.shape)
        if A[r, t] < L:
            break
        deg = float(np.rad2deg(th[t]))
        A[max(0, r - 4):r + 5, max(0, t - 4):t + 5] = 0
        # θ 는 법선 방향: 0°/180° = 세로선, 90° = 가로선
        tilt = min(abs(deg - 90), abs(deg), abs(deg - 180))
        tilt = min(tilt, 90 - tilt)
        # 실제로 이어진 선분인지 (잡티·글자가 우연히 한 줄에 놓인 것은 아님): 선 위 픽셀을 선 방향으로 늘어놓고 3px 틈까지 이은 가장 긴 구간
        c, sn = np.cos(th[t]), np.sin(th[t])
        on = np.abs(xs0 * c + ys0 * sn - (r - diag)) <= 1.0
        if on.sum() < L:
            continue
        tt = np.sort(-xs0[on] * sn + ys0[on] * c)
        brk = np.flatnonzero(np.diff(tt) > 3)
        st = np.r_[0, brk + 1]
        en = np.r_[brk, len(tt) - 1]
        seg = float((tt[en] - tt[st]).max()) if len(tt) else 0.0
        if seg < L:
            continue
        lines.append((deg, tilt, int(seg)))
    obl = [l for l in lines if l[1] >= 15]
    dirs = []
    for d, _, _ in obl:
        if not any(min(abs(d - e), 180 - abs(d - e)) < 8 for e in dirs):
            dirs.append(d)
    return {"dirs": len(dirs), "lines": len(obl), "straight": len(lines)}


def filled(a, box):
    """그림 영역 안이 여러 색 칸으로 찬 비율·색 수"""
    if not box:
        return {"frac": 0.0, "colors": 0}
    x0, y0, x1, y1 = int(box["x0"]) + 3, int(box["y0"]) + 3, int(box["x1"]) - 2, int(box["y1"]) - 2
    sub = a[max(0, y0):y1, max(0, x0):x1]
    if sub.size < 300:
        return {"frac": 0.0, "colors": 0}
    g = D.gray(sub)
    sat = _sat(sub)
    bg = np.median(sub.reshape(-1, 3), axis=0)
    fill = (np.abs(sub.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 25) & ((sat > 30) | (g < 120))
    q = (sub[fill] // 32).astype(np.int32)
    if not len(q):
        return {"frac": 0.0, "colors": 0}
    _, cnt = np.unique(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], return_counts=True)
    return {"frac": float(fill.mean()), "colors": int((cnt >= 0.01 * fill.size).sum())}


def loops(a, box):
    """같은 색 곡선이 한 열에 두 번 이상 지나는(닫힌 곡선·등고선) 비율의 최댓값 — 함수 그래프(선)는 열마다 한 번"""
    if not box:
        return 0.0
    best = 0.0
    for c in D.suggest_colors(a, box)[:6]:
        m, _ = D.mask_for(a, D.hex2rgb(c["color"]), 50, box)
        if m.sum() < 200:
            continue
        cols = m.any(axis=0)
        if cols.sum() < 20:
            continue
        multi = 0
        for x in np.flatnonzero(cols):
            rs = D.merge(D.runs(m[:, x]), 4)
            multi += len(rs) >= 2
        best = max(best, multi / cols.sum())
    return float(best)


def signals(a, det=None):
    box = (det or {}).get("box") if (det or {}).get("ok") else None
    out = {}
    for name, f in (("colorbar", lambda: colorbar(a, box)), ("round", lambda: round_shape(a)), ("oblique", lambda: oblique_lines(a)),
                    ("filled", lambda: filled(a, box)), ("loops", lambda: loops(a, box))):
        try:
            v = f()
        except Exception:
            v = None
        if name == "round" and v and v["r"] >= 0.18:
            out[name] = v
        elif name == "oblique" and v and v["dirs"] >= 2 and v["lines"] >= 3:
            out[name] = v
        elif name == "filled" and v and v["frac"] >= 0.15 and v["colors"] >= 6:
            out[name] = v
        elif name == "loops" and v and v >= 0.35:
            out[name] = round(v, 2)
        elif name == "colorbar" and v:
            out[name] = v
    return out


WHY = {"colorbar": "색 띠(컬러바)", "round": "원형 축·원판", "oblique": "비스듬한 긴 직선(3D 축)", "filled": "여러 색 칸으로 찬 그림 영역",
       "loops": "닫힌 곡선(등고선)"}


def _heat(sig):
    f = sig.get("filled")
    return bool(f and ((f["frac"] >= 0.5 and f["colors"] >= 8) or ("colorbar" in sig and f["frac"] >= 0.15)))


def judge(a, det, vlm_type=None, strong_2d=False, sig=None):
    """→ None(지원) 또는 {"kind", "label", "why": [...], "vlm", "signals"}.
    strong_2d: 직교 축선·눈금 글자로 두 축 보정이 됨 (그러면 VLM 판정만으로는 거절하지 않음).
    VLM 판정은 종류별 규칙 신호로 확인한다: 3D↔비스듬한 직선, 파이·극좌표↔큰 원, 히트맵↔색 칸·컬러바, 등고선↔닫힌 곡선·컬러바."""
    sig = signals(a, det) if sig is None else sig
    vk = vlm_kind(vlm_type)
    rd = sig.get("round")
    why = [WHY[k] for k in sig]
    out = lambda kind, extra=(): {"kind": kind, "label": LABEL[kind], "why": why + list(extra), "vlm": vlm_type, "signals": list(sig)}  # noqa: E731
    # 규칙만으로도: 큰 원(파이·도넛·극좌표 격자), 컬러바 + 색 칸 (히트맵)
    if rd and rd["r"] >= 0.25 and (rd["rings"] >= 3 or rd["fill"] >= 0.5):
        return out(vk if vk in ("pie", "polar") else ("polar" if rd["rings"] >= 3 else "pie"))
    if "colorbar" in sig and _heat(sig) and (sig["filled"]["frac"] >= 0.5 or vk):
        return out(vk if vk in ("heatmap", "contour") else "heatmap")
    if not vk:
        return None
    ok = {"3d": "oblique" in sig or not (det or {}).get("ok"),
          "pie": bool(rd), "polar": bool(rd),
          "heatmap": _heat(sig),
          "contour": "loops" in sig or "colorbar" in sig or _heat(sig),
          "other": any(k in sig for k in ("colorbar", "round", "oblique")) or _heat(sig)}[vk]
    if ok:
        return out(vk if vk != "other" else ("3d" if "oblique" in sig else "heatmap" if _heat(sig) or "colorbar" in sig else
                                               "contour" if "loops" in sig else "polar" if rd and rd["rings"] >= 3 else "pie" if rd else "other"),
                   [f"AI 판정 {vlm_type}"])
    if not strong_2d and vk != "other":  # 규칙 신호는 없지만 직교 축·눈금 글자도 제대로 안 잡힘 ('기타'는 축 없는 인포그래픽일 수 있어 규칙 신호가 있어야 거절)
        return out(vk, [f"AI 판정 {vlm_type}", "직교 축·눈금 글자로 보정되지 않음"])
    return None


def doubt(vlm_type):
    """VLM 은 지원 안 하는 종류라 했지만 규칙·보정이 2D 그래프로 본 경우의 경고 (거절은 안 함)"""
    vk = vlm_kind(vlm_type)
    return (f"AI 는 이 그림을 {LABEL[vk]}(으)로 봤지만 직교 축·눈금이 잡혀 계속 진행했습니다 — 결과를 겹쳐 보기로 꼭 확인하세요"
            if vk else None)


def message(u):
    return (f"지원하지 않는 형식({u['label']})입니다 — 2D 직교좌표(선·산점·막대)만 지원합니다. "
            "자동 추출은 멈췄습니다. 직교 축이 있는 그림이면 '그래도 자동 보정'을 누르거나 수동 보정(축 위 두 점씩)으로 진행할 수 있습니다.")
