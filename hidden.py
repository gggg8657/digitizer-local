"""다른 선 밑에 완전히 가린 구간 채우기 (보간).

선 계열 S 가 그림 안쪽에서 끊겼는데(가운데 틈, 또는 다른 선들은 더 가는데 S 만 일찍 시작·끝남) 끊긴 자리가 다른 선 O 위이고,
S 가 들어가는 방향과 O 의 방향이 이어지면 S 는 O 밑에 숨어 같이 간다고 보고 O 의 경로를 S 의 점으로 채운다.
가운데 틈인데 양 끝이 같은 선 위가 아니면, 짧은 틈(그림 폭 10% 이하)만 직선으로 잇는다.
채운 구간은 계열의 interp = [[x 시작 px, x 끝 px], ...] 로 표시하고 경고를 남긴다 (화면·표에서 '보간됨').
파선·점선 계열은 끝을 늘리지 않는다 (예측선이 실측선 끝에서 시작하는 그림을 잘못 잇지 않게).
"""
import math

import numpy as np


def _line(s):
    return s.get("px") and s.get("kind") == "line" and len(s["px"]) >= 5 and not s.get("calib")


def _poly(s):
    P = np.asarray(sorted(s["px"]), float)
    return P[:, 0], P[:, 1]


def _yat(poly, x):
    xs, ys = poly
    if x < xs[0] - 1 or x > xs[-1] + 1:
        return None
    return float(np.interp(x, xs, ys))


def _dir(xs, ys, x, toward, L):
    """x 에서 toward(-1 왼쪽, +1 오른쪽) 방향으로 길이 L 만큼의 평균 방향 (각도, 라디안). 점이 모자라면 None"""
    if toward < 0:
        m = (xs >= x - L) & (xs <= x)
    else:
        m = (xs >= x) & (xs <= x + L)
    if m.sum() < 2:
        return None
    k = np.polyfit(xs[m], ys[m], 1)[0]
    return math.atan(k)


def _dashed(s):
    st = str(s.get("style") or "").lower()
    return "dash" in st or "dot" in st or any("파선" in w or "점선·파선" in w for w in s.get("warn") or [])


def fill(det, series):
    box = det["box"]
    W, H = box["x1"] - box["x0"], box["y1"] - box["y0"]
    lines = [s for s in series if _line(s)]
    if len(lines) < 2:
        return series
    tol = max(4.0, 0.012 * H)
    L = max(12.0, 0.04 * W)
    edge = max(4.0, 0.015 * W)
    for s in lines:
        others = [o for o in lines if o is not s]
        xs, ys = _poly(s)
        add, spans = [], []

        def follow(o, xa, xb):
            ox, oy = _poly(o)
            m = (ox > xa) & (ox < xb)
            return [[float(x), float(y)] for x, y in zip(ox[m], oy[m])]
        # 가운데 틈
        dx = np.diff(xs)
        med = float(np.median(dx)) if len(dx) else 0
        for i in np.flatnonzero(dx > max(3 * med, 0.03 * W)):
            xa, ya, xb, yb = xs[i], ys[i], xs[i + 1], ys[i + 1]
            occ = None
            for o in others:
                pa, pb = _yat(_poly(o), xa), _yat(_poly(o), xb)
                if pa is not None and pb is not None and abs(pa - ya) <= tol and abs(pb - yb) <= tol:
                    occ = o
                    break
            if occ is not None:
                pts = follow(occ, xa, xb)
            elif xb - xa <= 0.1 * W and all(any((lambda q: q is not None and abs(q - yy) <= tol)(_yat(_poly(o), xx)) for o in others)
                                             for xx, yy in ((xa, ya), (xb, yb))):  # 양 끝이 다른 선 위 (교차점 근처에서 가림)
                n = max(2, int((xb - xa) / max(med, 1)))
                pts = [[float(xa + (xb - xa) * q / n), float(ya + (yb - ya) * q / n)] for q in range(1, n)]
            else:
                continue
            if pts:
                add += pts
                spans.append([float(xa), float(xb)])
        # 양 끝: 다른 선들은 더 가는데 이 선만 먼저 끝남
        if not _dashed(s):
            lo = min(float(_poly(o)[0][0]) for o in others)
            hi = max(float(_poly(o)[0][-1]) for o in others)
            for side, xe, ye, bound in ((-1, xs[0], ys[0], lo), (1, xs[-1], ys[-1], hi)):
                if (bound - xe) * side < 0.03 * W:
                    continue
                if min(xe - box["x0"], box["x1"] - xe) < edge or min(ye - box["y0"], box["y1"] - ye) < edge:
                    continue  # 그림 테두리에서 끝남 (밖으로 나감)
                d_s = _dir(xs, ys, xe, -side, L)  # S 가 끝으로 들어오는 방향
                if d_s is None:
                    continue
                best = None
                for o in others:
                    po = _poly(o)
                    yo = _yat(po, xe)
                    if yo is None or abs(yo - ye) > tol:
                        continue
                    d_o = _dir(po[0], po[1], xe, side, L)  # O 가 끝 너머로 가는 방향
                    if d_o is None or abs(d_o - d_s) > math.radians(12):
                        continue
                    if best is None or abs(yo - ye) < best[0]:
                        best = (abs(yo - ye), o, d_o)
                if not best:
                    continue
                o, d0 = best[1], best[2]
                ox, oy = _poly(o)
                m = (ox < xe) if side < 0 else (ox > xe)
                seg = sorted(zip(ox[m], oy[m]), key=lambda p: -p[0] if side < 0 else p[0])
                pts = []
                for j, (x, y) in enumerate(seg):  # O 가 방향을 크게 바꾸면(갈라짐) 거기서 멈춤
                    if pts and j >= 2:
                        d = _dir(ox, oy, x, -side, L)
                        if d is not None and abs(d - d0) > math.radians(20):
                            break
                    pts.append([float(x), float(y)])
                if len(pts) >= 2:
                    add += pts
                    spans.append(sorted([float(xe), float(pts[-1][0])]))
        if add:
            s["px"] = sorted([list(p) for p in s["px"]] + add)
            s["interp"] = spans
            s["warn"] = list(s.get("warn") or []) + [f"다른 선에 가려 안 보이는 구간 {len(spans)}곳을 보간했습니다(위에 그려진 선의 경로 또는 직선) — 화면·표의 '보간됨' 점을 확인하세요"]
    return series
