"""다른 선 밑에 완전히 가린 구간 채우기 (보간).

선 계열 S 가 그림 안쪽에서 끊겼는데(가운데 틈, 또는 다른 선들은 더 가는데 S 만 일찍 시작·끝남) 끊긴 자리가 다른 선 O 위이고,
S 가 들어가는 방향과 O 의 방향이 이어지면 S 는 O 밑에 숨어 같이 간다고 보고 O 의 경로를 S 의 점으로 채운다.
가운데 틈인데 양 끝이 같은 선 위가 아니면, 짧은 틈(그림 폭 10% 이하)만 직선으로 잇는다.
채운 구간은 계열의 interp = [[x 시작 px, x 끝 px], ...] 로 표시하고 경고를 남긴다 (화면·표에서 '보간됨').
그 가운데 다른 선의 경로를 따라 채운 구간(겹침 — 두 선이 같은 값)은 shared 에도 넣는다 (화면·표에서 '겹침', 2026-10-20).
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
        add, spans, shared = [], [], []
        bends = []

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
            joins = None
            for o in others:
                po = _poly(o)
                pa, pb = _yat(po, xa), _yat(po, xb)
                if pa is None or pb is None:
                    continue
                # 틈 끝 점은 재표본 간격(med)만큼 갈라지는 자리 앞에 있을 수 있음 — 두 선 기울기 차 × 간격만큼 더 봐 준다 (2026-10-20)
                ta = tb = tol
                ka, ko = _dir(xs, ys, xa, -1, L), _dir(po[0], po[1], xa, -1, L)
                if ka is not None and ko is not None:
                    ta = tol + abs(math.tan(ka) - math.tan(ko)) * max(med, 1.0)
                kb, kob = _dir(xs, ys, xb, 1, L), _dir(po[0], po[1], xb, 1, L)
                if kb is not None and kob is not None:
                    tb = tol + abs(math.tan(kb) - math.tan(kob)) * max(med, 1.0)
                def meets(x0_, y0_, k_, toward):
                    """S 의 끝을 제 기울기로 1.6 간격만큼 늘리면 O 와 만나는 x (재표본 간격 사이의 합류점) — 없으면 None"""
                    if k_ is None:
                        return None
                    for t_ in np.linspace(0, 1.6 * max(med, 1.0), 9):
                        xx = x0_ + toward * t_
                        yo_ = _yat(po, xx)
                        if yo_ is not None and abs(y0_ + math.tan(k_) * toward * t_ - yo_) <= tol:
                            return float(xx)
                    return None
                ja = xa if abs(pa - ya) <= ta else meets(xa, ya, ka, 1)
                jb = xb if abs(pb - yb) <= tb else meets(xb, yb, kb, -1)
                if ja is not None and jb is not None and ja < jb:
                    occ = o
                    joins = (ja, jb, ka, kb)
                    break
            if occ is not None:
                ja, jb, ka_, kb_ = joins
                pts = []
                for xx in np.arange(xa + 2, ja, 2.0):  # 합류점까지는 S 제 기울기로
                    pts.append([float(xx), float(ya + math.tan(ka_) * (xx - xa))])
                pts += follow(occ, ja, jb)
                for xx in np.arange(jb + 2, xb, 2.0):
                    pts.append([float(xx), float(yb + math.tan(kb_) * (xx - xb))])
                if pts:
                    shared.append([float(ja), float(jb)])
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
                    if d_o is None:
                        continue
                    if abs(d_o - d_s) > math.radians(12):
                        # 수직에 가깝게 꺾여 평평한 선(기준선·0 선) 위로 내려앉은 끝 — 그 선 밑에 숨어 같이 감 (wm_03 꼬리, 2026-10-20)
                        yin = _yat((xs, ys), xe - side * L)
                        if not (abs(d_s) >= math.radians(30) and abs(d_o) <= math.radians(10) and yin is not None
                                and abs(yin - (yo if yo is not None else ye)) >= 0.4 * L):
                            continue
                        bends.append(o)
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
                    shared.append(spans[-1])
        if add:
            s["px"] = sorted([list(p) for p in s["px"]] + add)
            s["interp"] = spans
            if shared:  # 위에 그려진 다른 선을 따라 채운 구간 = 겹침(같은 값) — 직선 보간과 구분 (화면 '겹침')
                s["shared"] = sorted(list(s.get("shared") or []) + shared)
            ns = len(shared)
            if bends:
                s["warn"] = list(s.get("warn") or []) + ["선 끝이 가파르게 꺾여 평평한 다른 선(기준선) 위로 내려앉아, 그 선 밑에 숨어 같이 간다고 보고 늘렸습니다 — 실제로 끝난 선이면 '겹침' 점을 지우세요"]
            if ns == len(spans):
                msg = f"다른 선 밑에 같은 값으로 숨은 구간 {ns}곳을 위에 그려진 선의 경로로 채웠습니다 — 화면·표의 '겹침' 점을 확인하세요"
            elif ns:
                msg = (f"다른 선에 가려 안 보이는 구간 {len(spans)}곳을 채웠습니다(겹침 {ns}곳은 위에 그려진 선의 경로, "
                       f"나머지 {len(spans) - ns}곳은 직선 보간) — 화면·표의 '겹침'·'보간됨' 점을 확인하세요")
            else:
                msg = f"다른 선에 가려 안 보이는 구간 {len(spans)}곳을 보간했습니다(위에 그려진 선의 경로 또는 직선) — 화면·표의 '보간됨' 점을 확인하세요"
            s["warn"] = list(s.get("warn") or []) + [msg]
    return series
