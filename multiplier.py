"""축 배수(×10ⁿ) — 눈금 숫자는 0.5·1·1.5 인데 축 끝이나 축 이름에 '×10⁻⁵'·'x 10^5'·'1e5'·'(×10³)' 가 따로 붙은 그림.

1) 축 이름(VLM 이 이미 읽은 x_label·y_label)에 '(×10³)' 꼴이 있으면 그대로 쓴다 (VLM 추가 호출 없음).
2) 아니면 축 모서리(y축 위 끝 위, x축 오른쪽 끝 아래)에 눈금 글자가 아닌 작은 글자 덩어리가 있을 때만 그 조각을 VLM 에
   '글자를 그대로 옮겨 적기'로 읽히고, 배수 꼴인지는 여기 정규식이 정한다 (VLM 이 배수를 지어내도 꼴이 아니면 버림).
보정(calib)의 v1·v2 에 10ⁿ 을 곱한다 — 선형축만, 눈금 값이 이미 그 자릿수로 읽혔으면(예: 0.00001) 곱하지 않는다.
"""
import re

import numpy as np

import digitize as D

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
PROMPT = """Each image is a small piece cut from next to a chart axis (above the top of the y-axis, or near the right end of the x-axis).
For each image copy the printed text exactly, writing superscripts with ^ (for example "x 10^-5", "×10^3", "1e5", "Time (s)").
Return ONLY JSON: {"texts": {"1": "text", "2": "text"}} with one entry per image; "" if there is no text."""


def _norm(t):
    t = str(t or "")
    t = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^" + m.group().translate(_SUP), t)  # 10⁻⁵ → 10^-5 (10-5 가 아니게)
    return t.replace("−", "-").replace("–", "-").replace("{", "").replace("}", "").replace("$", "").replace("\\times", "×")


def parse(text, whole=False):
    """'x 10^-5' '×10^3' '(×10³)' '1e5' '1e-3' → 지수 n (정수) 또는 None.
    whole=True(모서리 조각): 조각 글자 전체가 배수일 때만 ('1e5'·'10^5' 단독 포함). 축 이름 안에서는 ×/x 가 붙은 꼴만."""
    t = _norm(text).strip()
    if not t:
        return None
    m = re.search(r"(?:^|[\s(\[,;])(?:[x×X*·]|\\cdot)\s*10\s*(?:\^|\*\*)\s*\(?\s*([-+]?\d{1,3})\s*\)?", t)
    if m is None and whole:
        m = re.search(r"(?:^|[\s(\[,;])(?:[x×X*·])\s*10\s*([-+]\d{1,3})(?![\d.])", t)  # 위첨자를 ^ 없이 'x 10-5' 로 옮긴 것
    if m is None and whole:
        m = re.fullmatch(r"\(?\s*(?:1\s*)?[eE]\s*([-+]?\d{1,3})\s*\)?", t) or re.fullmatch(r"\(?\s*10\s*\^\s*\(?([-+]?\d{1,3})\)?\s*\)?", t)
    if not m:
        return None
    n = int(m.group(1).replace("+", ""))
    return n if n != 0 and abs(n) <= 30 else None


def corner_boxes(a, det):
    """배수 글자가 있을 만한 두 모서리 → {"y": [x0,y0,x1,y1] 또는 없음, "x": ...}: 눈금 글자 덩어리를 뺀 남은 잉크의 범위"""
    H, W = a.shape[:2]
    box, lb = det["box"], det.get("labels") or {}
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    w, h = x1 - x0, y1 - y0
    ink = D.gray(a) < 160
    excl = np.zeros_like(ink)
    for b in (lb.get("x_boxes") or []) + (lb.get("y_boxes") or []):
        excl[max(0, int(b[1]) - 3):int(b[3]) + 4, max(0, int(b[0]) - 3):int(b[2]) + 4] = True
    out = {}
    regions = {"y": (max(0, int(y0 - max(0.16 * h, 50))), max(0, int(y0) - 2), max(0, int(x0 - 0.12 * w)), min(W, int(x0 + 0.3 * w))),
               "x": (min(H, int(y1) + 3), min(H, int(y1 + max(0.2 * h, 70))), max(0, int(x1 - 0.22 * w)), min(W, int(x1 + 0.12 * w)))}
    for k, (r0, r1, c0, c1) in regions.items():
        if r1 - r0 < 6 or c1 - c0 < 6:
            continue
        m = ink[r0:r1, c0:c1] & ~excl[r0:r1, c0:c1]
        m[m.mean(axis=1) > 0.6, :] = False  # 축선·테두리
        m[:, m.mean(axis=0) > 0.6] = False
        if m.sum() < 12:
            continue
        rows, cols = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
        # 축에 가장 가까운 글줄 하나 (y: 맨 아래 글줄, x: 맨 위 글줄 — 그 아래 축 이름은 뺌)
        rr = D.merge(D.runs(m.any(axis=1)), 3)
        s, e = rr[-1] if k == "y" else rr[0]
        if k == "x" and len(rr) >= 2 and e - s < 6:
            s, e = rr[1]
        sub = m[s:e + 1]
        cols = np.where(sub.any(axis=0))[0]
        cr = D.merge(D.runs(sub.any(axis=0)), max(6, int(1.2 * (e - s + 1))))
        # 축 끝에 가까운 글자 묶음 하나 (y: 왼쪽, x: 오른쪽)
        cs, ce = cr[0] if k == "y" else cr[-1]
        bw, bh = ce - cs + 1, e - s + 1
        if bh < 5 or bw > 0.45 * w or bh > 0.12 * H:
            continue
        out[k] = [c0 + cs, r0 + s, c0 + ce + 1, r0 + e + 1]
    return out


def find(a, det, v, read_texts):
    """→ {"x": n, "y": n, "src": {...}, "text": {...}} (찾은 축만). read_texts(a, boxes) → 글자 목록 (VLM)"""
    res = {"src": {}, "text": {}}
    for k in ("x", "y"):
        n = parse((v or {}).get(k + "_label"))
        if n is not None:
            res[k], res["src"][k], res["text"][k] = n, "label", (v or {}).get(k + "_label")
    want = [k for k in ("y", "x") if k not in res]
    if not want:
        return res
    cb = corner_boxes(a, det)
    ks = [k for k in want if k in cb]
    if not ks:
        return res
    try:
        texts = read_texts(a, [cb[k] for k in ks]) or []
    except Exception:
        texts = []
    for k, t in zip(ks, texts):
        n = parse(t, whole=True)
        if n is not None:
            res[k], res["src"][k], res["text"][k] = n, "corner", t
            res.setdefault("box", {})[k] = cb[k]
    return res


def apply(calib, mult):
    """보정에 배수 곱하기 (선형·숫자축만, 이미 그 자릿수면 건너뜀) → 실제로 곱한 축 {k: n}"""
    done = {}
    for k in ("x", "y"):
        n = mult.get(k)
        c = (calib or {}).get(k)
        if n is None or not c or c.get("log") or c.get("cat"):
            continue
        mag = max(abs(float(c["v1"])), abs(float(c["v2"])))
        if mag == 0:
            continue
        lm = np.log10(mag)
        if (n < 0 and lm <= n + 2) or (n > 0 and lm >= n - 1):  # 눈금 값을 이미 배수까지 넣어 읽음 (0.00001·200000)
            continue
        c["v1"], c["v2"] = float(c["v1"]) * 10.0 ** n, float(c["v2"]) * 10.0 ** n
        c["mult"] = n
        done[k] = n
    return done
