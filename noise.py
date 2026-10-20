"""범례가 없을 때 색으로 찾은 계열 가운데 잡음(글자·화살표·안티앨리어싱 번짐·겹친 반투명 표식·컬러바 산점)을 정리한다.

규칙 (범례 이름이 없을 때만 — 범례가 있으면 사용자가 읽은 계열을 그대로 둔다):
 1. 컬러바가 붙은 산점도: 색이 값을 뜻하므로 색마다 계열로 나누지 않고 채도 있는 표식 전부를 한 계열로.
 2. 끊긴 무채색 선(글자·화살표·검정 선의 회색 번짐): 다른 이어진 선이 그림 폭 절반 넘게 지나가면 뺀다.
 3. 그림 폭 12% 안에 몰린 짧은 선 조각(반투명 표식 두 개가 겹친 진한 색 등): 다른 계열이 있으면 뺀다.
 4. 같은 표식의 채움·테두리·번짐 색(점이 대부분 같은 자리): 한 계열로 합친다 (점 합집합).
 5. 무채색 글자 조각 점(글자 경고가 붙고 8개 미만): 넓게 지나는 선 계열이 있으면 뺀다.
빼거나 합친 것은 warn 문장으로 남긴다.
"""
import numpy as np

import digitize as D


def _achromatic(c):
    r = D.hex2rgb(c)
    return max(r) - min(r) < 30


def _span(s):
    xs = [p[0] for p in s.get("px") or []]
    return (max(xs) - min(xs)) if xs else 0.0


def _broken(s):
    return any("끊긴 선" in w for w in s.get("warn") or [])


def _inside(a, box, color, Q, grow=4):
    """점들이 color 표식 마스크(grow px 넓힘) 안에 드는 비율"""
    m, off = D.mask_for(a, D.hex2rgb(color), 60, box, remove_lines=False)
    if not m.any():
        return 0.0
    try:
        from scipy import ndimage
        m = ndimage.binary_dilation(m, iterations=grow)
    except ImportError:
        m = D._shift_and(m, grow, "or")
    k = 0
    for x, y in Q:
        i, j = int(y - off[1]), int(x - off[0])
        if 0 <= i < m.shape[0] and 0 <= j < m.shape[1] and m[i, j]:
            k += 1
    return k / max(1, len(Q))


def colorbar_points(a, calib, box, cbar, legends=None):
    """컬러바 산점도: 채도 있는(또는 아주 어두운 보라·남색) 표식 전부 → 점 계열 하나 또는 None"""
    x0, y0, x1, y1 = D._roi(box, a.shape)
    sub = a[y0:y1, x0:x1].astype(np.int16)
    sat = sub.max(axis=2) - sub.min(axis=2)
    m = sat >= 40
    if cbar:  # 그림 영역 안으로 들어온 컬러바 조각은 뺌
        m[max(0, int(cbar["y0"]) - y0 - 2):max(0, int(cbar["y1"]) - y0 + 3), max(0, int(cbar["x0"]) - x0 - 2):max(0, int(cbar["x1"]) - x0 + 3)] = False
    for lb in legends or []:
        m[max(0, int(lb[1]) - y0):max(0, int(lb[3]) - y0 + 1), max(0, int(lb[0]) - x0):max(0, int(lb[2]) - x0 + 1)] = False
    if not m.any():
        return None
    px, warn = D.extract_points(m, (x0, y0), 4, None, a[y0:y1, x0:x1])
    if len(px) < 3:
        return None
    cols = sub[m]
    col = D.rgb2hex(np.median(cols, axis=0))
    return {"px": px, "warn": warn, "color": col}


def clean(a, det, out, chart_type="", colorbar=None):
    """extract_all 결과(out: 계열 목록)를 정리해 돌려준다. 범례 이름이 있는 계열이 하나라도 있으면 그대로."""
    named = any(s.get("name") and not str(s["name"]).startswith("계열 ") for s in out)
    if named:
        return out
    box = det["box"]
    W = box["x1"] - box["x0"]
    got = [s for s in out if s.get("px")]
    if not got:
        return out
    notes = []
    # 1. 컬러바 산점도
    if colorbar and chart_type == "scatter" and len(got) >= 2 and all(s.get("kind") == "points" for s in got):
        cp = colorbar_points(a, det["calib"], box, colorbar, det.get("legend_boxes"))
        if cp and len(cp["px"]) >= max(len(s["px"]) for s in got):
            s0 = dict(got[0], px=cp["px"], color=cp["color"], mode="points", how="all", kind="points", name="계열 1",
                      warn=[f"색 띠(컬러바)가 있는 산점도 — 색은 값을 뜻하므로 색 {len(got)}개를 한 계열(표식 {len(cp['px'])}개)로 묶었습니다"] + cp["warn"][:3])
            return [s0]
    keep = list(got)
    # 2. 끊긴 무채색 선
    solid = [s for s in keep if s.get("kind") in ("line", "line+markers") and not _broken(s) and _span(s) >= 0.5 * W]
    for s in list(keep):
        if s.get("kind") == "line" and _broken(s) and _achromatic(s.get("color") or "#000000") and any(t is not s for t in solid):
            keep.remove(s)
            notes.append(f"{s['color']} 끊긴 무채색 선(글자·화살표·번짐)")
    # 3. 짧은 선 조각
    for s in list(keep):
        if len(keep) >= 2 and s.get("kind") == "line" and _span(s) < 0.12 * W:
            keep.remove(s)
            notes.append(f"{s['color']} 좁은 곳에 몰린 짧은 선 조각")
    # 4. 같은 표식의 채움·테두리·번짐 색
    pts = [s for s in keep if s.get("kind") == "points" and len(s["px"]) >= 3]
    cnt = {c["color"]: c["count"] for c in det.get("colors") or []}
    pts.sort(key=lambda s: (_achromatic(s.get("color") or "#000000"), -cnt.get(s.get("color"), 0), -len(s["px"])))  # 채움(색 있고 픽셀 많은) 색이 기준
    merged = set()
    for i, s in enumerate(pts):
        if id(s) in merged:
            continue
        P = np.asarray(s["px"], float)
        for t in pts[i + 1:]:
            if id(t) in merged:
                continue
            Q = np.asarray(t["px"], float)
            d = np.sqrt(((Q[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
            if _achromatic(t.get("color") or "#000000") and (d <= 8).mean() < 0.6 and _inside(a, box, s["color"], Q) >= 0.7:
                merged.add(id(t))  # 큰 표식의 테두리·번짐(무채색): 점이 기준 색 표식 위·가장자리에만 있음 → 버림
                keep.remove(t)
                notes.append(f"{t['color']} (표식 테두리·번짐)")
                continue
            if (d <= 8).mean() >= 0.6:
                add = Q[d > 8]
                # 합칠 점: 다른 계열 점과 겹치지 않는 표식 (테두리 색만 보인 작은 표식 등)
                if len(add):
                    nn = np.sqrt(((add[:, None, :] - add[None, :, :]) ** 2).sum(-1)) + np.eye(len(add)) * 1e9
                    add = add[nn.min(1) > 6] if len(add) > 1 else add
                s["px"] = sorted([list(map(float, p)) for p in P] + [list(map(float, p)) for p in add])
                P = np.asarray(s["px"], float)
                merged.add(id(t))
                keep.remove(t)
                notes.append(f"{t['color']} (같은 표식의 테두리·번짐 색 — {s['color']} 계열에 합침)")
    # 5. 무채색 글자 조각 점
    wide = [s for s in keep if s.get("kind") in ("line", "line+markers") and _span(s) >= 0.5 * W]
    for s in list(keep):
        if (s.get("kind") == "points" and len(s["px"]) < 8 and _achromatic(s.get("color") or "#000000") and wide
                and any("글자" in w for w in s.get("warn") or [])):
            keep.remove(s)
            notes.append(f"{s['color']} 글자 조각")
    if not notes:
        return out
    res = [s for s in out if s in keep or not s.get("px")]
    res = [s for s in res if s.get("px")]
    for i, s in enumerate(res):
        s["name"] = f"계열 {i + 1}"
    det.setdefault("warn", []).append("범례가 없어 색으로 찾은 계열 가운데 잡음을 정리했습니다: " + ", ".join(notes))
    return res
