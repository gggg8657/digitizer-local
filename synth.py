"""정확도 검증용 합성 그래프 — 참값(함수·점)과 정확한 픽셀 변환을 알고 그린다.
  pillow_plot(...)      Pillow 만으로 (어디서나)
  mpl_plot(...)         matplotlib 이 있으면 (안티앨리어싱·격자·범례가 있는 '논문 같은' 그림)
둘 다 (PNG bytes, truth) 를 돌려준다. truth = {"calib": 참 보정, "box": 그림 영역, "series": [{name,color,kind,x,y}]}
"""
import io
import math

from PIL import Image, ImageDraw, ImageFont


def _font(size):
    for p in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _fmt(v):
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 0.01:
        return f"{v:.0e}".replace("e+0", "e").replace("e-0", "e-")
    return f"{v:g}"


def pillow_plot(series, xr, yr, xticks, yticks, xlog=False, ylog=False, size=(800, 560), box=(90, 40, 760, 480), grid=True):
    W, H = size
    bx0, by0, bx1, by1 = box
    f = (lambda v, lg: math.log10(v) if lg else v)

    def px(x):
        return bx0 + (f(x, xlog) - f(xr[0], xlog)) / (f(xr[1], xlog) - f(xr[0], xlog)) * (bx1 - bx0)

    def py(y):
        return by1 - (f(y, ylog) - f(yr[0], ylog)) / (f(yr[1], ylog) - f(yr[0], ylog)) * (by1 - by0)
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    fn = _font(15)
    if grid:
        for t in xticks:
            d.line([(px(t), by0), (px(t), by1)], fill=(225, 225, 225))
        for t in yticks:
            d.line([(bx0, py(t)), (bx1, py(t))], fill=(225, 225, 225))
    for s in series:
        pts = [(round(px(x)), round(py(y))) for x, y in zip(s["x"], s["y"])]
        if s["kind"] == "line":
            d.line(pts, fill=s["color"], width=2, joint="curve")
        else:
            for x, y in pts:
                d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=s["color"])
    d.rectangle([bx0, by0, bx1, by1], outline="black", width=1)
    for t in xticks:
        x = round(px(t))
        d.line([(x, by1), (x, by1 + 6)], fill="black")
        d.text((x, by1 + 10), _fmt(t), fill="black", font=fn, anchor="mt")
    for t in yticks:
        y = round(py(t))
        d.line([(bx0 - 6, y), (bx0, y)], fill="black")
        d.text((bx0 - 10, y), _fmt(t), fill="black", font=fn, anchor="rm")
    b = io.BytesIO()
    im.save(b, "PNG")
    # Pillow 선은 정수 픽셀 중심에 그려진다: 픽셀 (i) 의 중심은 i+0.5 → 참 보정도 같은 규칙
    calib = {"x": {"p1": [px(xr[0]) + 0.5, by1], "p2": [px(xr[1]) + 0.5, by1], "v1": xr[0], "v2": xr[1], "log": xlog},
             "y": {"p1": [bx0, py(yr[0]) + 0.5], "p2": [bx0, py(yr[1]) + 0.5], "v1": yr[0], "v2": yr[1], "log": ylog}}
    return b.getvalue(), {"calib": calib, "box": {"x0": bx0, "y0": by0, "x1": bx1, "y1": by1}, "series": series,
                          "xticks": [t for t in xticks], "yticks": [t for t in yticks]}


def mpl_plot(series, xlog=False, ylog=False, dpi=100, figsize=(7, 4.8), grid=True, legend=True, xlabel="x", ylabel="y",
             xlim=None, ylim=None):
    """matplotlib 으로 그리고, 그림 변환(transData)에서 참 보정을 얻는다. matplotlib 없으면 ImportError."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for s in series:
        if s["kind"] == "line":
            ax.plot(s["x"], s["y"], color=s["color"], lw=1.6, label=s["name"], ls=s.get("ls", "-"))
        else:
            ax.plot(s["x"], s["y"], ls="none", marker=s.get("marker", "o"), ms=6, color=s["color"], label=s["name"])
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if grid:
        ax.grid(True, color="#dddddd", lw=0.8)
    if legend:
        ax.legend(loc="best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.canvas.draw()
    H = fig.canvas.get_width_height()[1]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    T = ax.transData

    def pix(x, y):  # matplotlib 은 아래가 0, 픽셀 모서리 좌표 → 위가 0 인 이미지 좌표
        a, b = T.transform((x, y))
        return [float(a), float(H - b)]
    calib = {"x": {"p1": pix(x0, y0), "p2": pix(x1, y0), "v1": x0, "v2": x1, "log": xlog},
             "y": {"p1": pix(x0, y0), "p2": pix(x0, y1), "v1": y0, "v2": y1, "log": ylog}}
    bb = ax.get_window_extent()
    xt = [t for t in ax.get_xticks() if min(x0, x1) <= t <= max(x0, x1)]
    yt = [t for t in ax.get_yticks() if min(y0, y1) <= t <= max(y0, y1)]
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"calib": calib, "box": {"x0": bb.x0, "y0": H - bb.y1, "x1": bb.x1, "y1": H - bb.y0},
                          "series": series, "xticks": xt, "yticks": yt}


def paper_plot(kind="A", dpi=150):
    """'논문 같은' 그림: 세리프 글꼴·안쪽 눈금·작은 눈금·네 변 눈금·틀 없는 범례.
    A: 로그 y, 선+표식 3계열(파랑·빨강·검정)   B: 로그-로그, 빈 원 표식 + 회색 점선 맞춤선, 아래 오른쪽 범례
    E: 엑셀형 — 세로축선·눈금 없음, 가로 격자만, '1,600' 꼴 눈금 글자, 굵은 선+표식 2계열, 틀 없는 범례"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rc = {"font.family": "serif", "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
          "xtick.minor.visible": True, "ytick.minor.visible": True}
    with plt.rc_context(rc):
        if kind == "A":
            T = np.linspace(300, 1500, 13)
            fs = [("UO2", lambda t: 4040 / (464 + t) + 1e-10 * t ** 3, "#1f77b4", "o"),
                  ("UO2-BeO", lambda t: 1.4 * 4040 / (464 + t), "#d62728", "s"),
                  ("U3Si2", lambda t: 0.6 + 0.012 * t, "#000000", "^")]
            fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=dpi)
            series = []
            for name, f, c, m in fs:
                ax.plot(T, f(T), color=c, marker=m, ms=5, lw=1.2, label=name)
                series.append({"name": name, "color": c, "kind": "line+points", "x": T, "y": f(T), "f": f})
            ax.set_yscale("log"); ax.set_ylim(1, 30); ax.set_xlim(200, 1600)
            ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Thermal conductivity (W/m·K)")
            ax.legend(frameon=False, loc="upper left", fontsize=8)
        elif kind == "E":
            from matplotlib.ticker import FuncFormatter
            t = np.arange(0, 49, 3.0)
            fs = [("core", lambda v: 1200 * (1 - np.exp(-v / 12)) + 150, "#4472c4", "o"), ("flow", lambda v: 900 * np.exp(-v / 30) + 200, "#ed7d31", "s")]
            fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=dpi)
            series = []
            for name, f, c, m in fs:
                ax.plot(t, f(t), color=c, lw=2.2, marker=m, ms=5, label=name)
                series.append({"name": name, "color": c, "kind": "line+points", "x": t, "y": f(t), "f": f})
            ax.set_xlim(0, 48); ax.set_ylim(0, 1600); ax.set_xticks(range(0, 49, 6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
            ax.grid(axis="y", color="#d9d9d9"); ax.tick_params(which="both", length=0)
            ax.minorticks_off()
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.set_xlabel("time (h)"); ax.legend(loc="center right", frameon=False)
        else:
            x = np.array([0.02, 0.05, 0.1, 0.3, 0.7, 1.5, 4, 9, 20, 50]); y = 3.2 * x ** 0.62
            fig, ax = plt.subplots(figsize=(4.5, 3.6), dpi=dpi)
            ax.loglog(x, y, "o", mfc="none", mec="k", ms=6, label="Measured")
            xx = np.logspace(-2, 2, 50)
            ax.loglog(xx, 3.2 * xx ** 0.62, "--", color="gray", label="Fit")
            ax.set_xlim(1e-2, 1e2); ax.set_ylim(0.1, 100)
            ax.set_xlabel("Dose rate (Gy/s)"); ax.set_ylabel("Yield (a.u.)"); ax.legend(loc="lower right", fontsize=8)
            series = [{"name": "Measured", "color": "#000000", "kind": "points", "x": x, "y": y}]
        fig.tight_layout()
        fig.canvas.draw()
        H = fig.canvas.get_width_height()[1]
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        T_ = ax.transData

        def pix(a, b):
            u, v = T_.transform((a, b))
            return [float(u), float(H - v)]
        xl, yl = ax.get_xscale() == "log", ax.get_yscale() == "log"
        calib = {"x": {"p1": pix(x0, y0), "p2": pix(x1, y0), "v1": x0, "v2": x1, "log": xl},
                 "y": {"p1": pix(x0, y0), "p2": pix(x0, y1), "v1": y0, "v2": y1, "log": yl}}
        bb = ax.get_window_extent()
        xt = [t for t in ax.get_xticks() if min(x0, x1) <= t <= max(x0, x1)]
        yt = [t for t in ax.get_yticks() if min(y0, y1) <= t <= max(y0, y1)]
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=dpi)
        plt.close(fig)
    return b.getvalue(), {"calib": calib, "box": {"x0": bb.x0, "y0": H - bb.y1, "x1": bb.x1, "y1": H - bb.y0},
                          "series": series, "xticks": xt, "yticks": yt}


def bar_plot(kind="simple", dpi=110):
    """막대그래프 (참값을 아는). kind:
    simple  — 세로 막대 5개, 채운 색 + 검은 테두리, 막대 위 숫자 주석
    stacked — 쌓인 막대 4개 × 3계열(꽉 찬 색) + 범례, 점선 구분선
    hatched — 쌓인 막대: 흰 바탕 무늬(/// · xxx) + 색 무늬, 범례(무늬 견본)
    error   — 빈 막대(색 테두리) + 오차 막대(캡), 같은 이름이 여러 막대에 걸친 범주
    hbar    — 가로 막대 (쌓임 2계열)
    log     — 로그 값축 세로 막대
    → (PNG, truth) truth = {orient, log, ticks, lim, bars: [{tops: [쌓인 위 끝], series: [계열 번호], err: (아래, 위) | None}], n_series}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    cats = ["A", "B", "C", "D", "E"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=dpi)
    bars, orient, log = [], "v", False
    if kind == "simple":
        v = [3.2, 7.5, 5.1, 9.3, 1.4]
        ax.bar(cats, v, width=0.6, color="#4c72b0", edgecolor="black", lw=1)
        for i, x in enumerate(v):
            ax.text(i, x + 0.25, f"{x:.1f}", ha="center", fontsize=8)
        ax.set_ylim(0, 11)
        bars = [{"tops": [x], "series": [0], "err": None} for x in v]
        ns = 1
    elif kind in ("stacked", "hatched"):
        data = np.array([[40, 25, 12], [18, 30, 9], [55, 8, 20], [22, 14, 31]], float)
        cols = ["#2ca02c", "#ff7f0e", "#1f77b4"] if kind == "stacked" else ["white", "#d62728", "white"]
        hat = [None, None, None] if kind == "stacked" else ["///", "xxx", "xx"]
        ecs = ["black"] * 3 if kind == "stacked" else ["black", "#d62728", "#1f3fd0"]
        bottom = np.zeros(4)
        for j in range(3):
            if kind == "hatched":
                ax.bar(range(4), data[:, j], 0.55, bottom=bottom, color=cols[j], hatch=hat[j], edgecolor=ecs[j], lw=0.8, label=f"S{j + 1}")
                ax.bar(range(4), data[:, j], 0.55, bottom=bottom, fill=False, edgecolor="black", lw=0.8)
            else:
                ax.bar(range(4), data[:, j], 0.55, bottom=bottom, color=cols[j], edgecolor="black", lw=0.8, label=f"S{j + 1}")
            bottom += data[:, j]
        ax.axvline(1.5, ls="--", color="black", lw=0.8)
        ax.set_xticks(range(4), ["Conv.", "Cir.", "Conv.", "Cir."])
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=8)
        bars = [{"tops": list(np.cumsum(r)), "series": [0, 1, 2], "err": None} for r in data]
        ns = 3
    elif kind == "error":
        v, e = [74.0, 52.0, 41.0, 63.0], [13.0, 0.0, 23.5, 6.0]
        ecol = ["#e31a1c", "#1f3fd0", "#444444", "#e31a1c"]
        for i in range(4):
            ax.bar(i, v[i], 0.55, fill=False, edgecolor=ecol[i], lw=1.4)
            if e[i]:
                ax.errorbar(i, v[i], yerr=e[i], color=ecol[i], capsize=6, lw=1.2)
            ax.text(i, v[i] + e[i] + 3, f"−{20 + i * 7:.1f}%", ha="center", fontsize=8)
        ax.set_xticks(range(4), ["Py*", "Hy*", "Direct*", "Py2"])
        ax.set_ylim(0, 110)
        bars = [{"tops": [v[i]], "series": [0 if ecol[i] == "#e31a1c" else 1 if i == 1 else 2], "err": (v[i] - e[i], v[i] + e[i]) if e[i] else None}
                for i in range(4)]
        ns = 3
    elif kind == "hbar":
        orient = "h"
        a1, a2 = [12.0, 30.5, 22.0, 8.0], [10.0, 6.0, 15.5, 20.0]
        y = np.arange(4)
        ax.barh(y, a1, 0.55, color="#8c564b", edgecolor="black", lw=0.8, label="P")
        ax.barh(y, a2, 0.55, left=a1, color="#17becf", edgecolor="black", lw=0.8, label="Q")
        ax.set_yticks(y, ["north", "south", "east", "west"])
        ax.set_xlim(0, 50)
        ax.legend(loc="lower right", fontsize=8)
        bars = [{"tops": [a1[i], a1[i] + a2[i]], "series": [0, 1], "err": None} for i in range(4)]
        ns = 2
    elif kind == "log":
        log = True
        v = [3.0, 45.0, 800.0, 12.0, 2500.0]
        ax.bar(cats, v, 0.6, color="#9467bd", edgecolor="black", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1, 1e4)
        bars = [{"tops": [x], "series": [0], "err": None} for x in v]
        ns = 1
    ax.set_xlabel("category" if orient == "v" else "value")
    fig.tight_layout()
    lim = ax.get_ylim() if orient == "v" else ax.get_xlim()
    tk = ax.get_yticks() if orient == "v" else ax.get_xticks()
    tk = [float(t) for t in tk if min(lim) - 1e-9 <= t <= max(lim) + 1e-9]
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"orient": orient, "log": log, "ticks": tk, "lim": [float(x) for x in lim], "bars": bars, "n_series": ns}


def annotated_scatter(dpi=120):
    """점마다 바로 오른쪽에 이름표가 붙은 산점(범례는 그래프 밖) — 이름표 붙은 점을 범례 견본으로 오인하지 않는지"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [20, 35, 60, 300, 480, 700, 850, 980, 40, 520]
    y = [11.2, 9.5, 7.4, 1.7, 0.3, 2.7, 0.4, 1.6, 5.5, 4.0]
    names = ["Geothermal", "Hydraulic", "Others", "CISO", "NG", "WACM", "Oil", "Coal", "Biological", "Mid"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=dpi)
    ax.plot(x, y, "o", ms=8, mfc="none", mec="#e01010", mew=1.6, label="Electricity Sources")
    for xi, yi, n in zip(x, y, names):
        ax.text(xi + 30, yi, n, va="center", fontsize=9)
    ax.set_xlim(-100, 1100)
    ax.set_ylim(-1, 13)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=True)
    fig.tight_layout()
    fig.canvas.draw()
    H = fig.canvas.get_width_height()[1]
    T = ax.transData

    def pix(a, b):
        u, v = T.transform((a, b))
        return [float(u), float(H - v)]
    calib = {"x": {"p1": pix(-100, -1), "p2": pix(1100, -1), "v1": -100, "v2": 1100, "log": False},
             "y": {"p1": pix(-100, -1), "p2": pix(-100, 13), "v1": -1, "v2": 13, "log": False}}
    bb = ax.get_window_extent()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"calib": calib, "box": {"x0": bb.x0, "y0": H - bb.y1, "x1": bb.x1, "y1": H - bb.y0}, "x": x, "y": y}


def boxed_bars(dpi=110):
    """값축을 같이 쓰는 칸막이 막대그래프 3칸 (왼쪽 칸에만 눈금 글자) + 위에 범례 틀"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(7.5, 3.8), dpi=dpi, sharey=True, gridspec_kw={"wspace": 0.03})
    data = [[(60, 30), (80, 20)], [(70, 30), (50, 50)], [(40, 60), (90, 10)]]
    bars = []
    for ax, d, t in zip(axs, data, ["Energy", "CO2", "Water"]):
        for i, (a1, a2) in enumerate(d):
            ax.bar(i, a1, 0.5, color="#d62728", edgecolor="black", lw=0.8)
            ax.bar(i, a2, 0.5, bottom=a1, color="#404040", edgecolor="black", lw=0.8)
            bars.append({"tops": [a1, a1 + a2], "series": [0, 1], "err": None})
        ax.set_title(t, y=0.88)
        ax.set_xticks([0, 1], ["Scrap", "Battery"])
        ax.set_ylim(0, 120)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#d62728", ec="k"), plt.Rectangle((0, 0), 1, 1, fc="#404040", ec="k")],
               labels=["Electricity", "Alkali"], loc="upper center", ncol=2)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.1, right=0.98)
    tk = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"orient": "v", "log": False, "ticks": tk, "lim": [0.0, 120.0], "bars": bars, "n_series": 2}


def grouped_bars(dpi=110):
    """묶음 막대: 해마다 Plan·Actual 두 막대가 나란히 (범주 글자는 묶음 가운데 하나)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plan, act = [12, 15, 9, 20], [8, 11, 14, 6]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=dpi)
    ax.bar(x - 0.175, plan, 0.35, color="#1f77b4", label="Plan")
    ax.bar(x + 0.175, act, 0.35, color="#ff7f0e", label="Actual")
    ax.set_xticks(x, ["2021", "2022", "2023", "2024"])
    ax.set_ylim(0, 21)
    ax.legend(loc="upper left")
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"rows": {"2021": (12, 8), "2022": (15, 11), "2023": (9, 14), "2024": (20, 6)}, "ticks": [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]}


# ── 2026-10-19 개선 시험용 (거절·축선 없는 차트·점선·같은 색 선/표식·오차막대·기울기) ─────────────────
def _mpl_truth(fig, ax, series, xlog=False, ylog=False, dpi=100):
    """matplotlib 그림 → (PNG, 참값) — mpl_plot 과 같은 규칙"""
    import matplotlib.pyplot as plt
    fig.canvas.draw()
    H = fig.canvas.get_width_height()[1]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    T = ax.transData

    def pix(x, y):
        a, b = T.transform((x, y))
        return [float(a), float(H - b)]
    calib = {"x": {"p1": pix(x0, y0), "p2": pix(x1, y0), "v1": x0, "v2": x1, "log": xlog},
             "y": {"p1": pix(x0, y0), "p2": pix(x0, y1), "v1": y0, "v2": y1, "log": ylog}}
    bb = ax.get_window_extent()
    xt = [float(t) for t in ax.get_xticks() if min(x0, x1) <= t <= max(x0, x1)]
    yt = [float(t) for t in ax.get_yticks() if min(y0, y1) <= t <= max(y0, y1)]
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"calib": calib, "box": {"x0": bb.x0, "y0": H - bb.y1, "x1": bb.x1, "y1": H - bb.y0},
                          "series": series, "xticks": xt, "yticks": yt}


def styled_lines(style="ggplot", dpi=100):
    """축선 없는 회색 바탕(ggplot·seaborn darkgrid·fivethirtyeight) 2계열 — 축선·눈금선이 없거나 흰 격자만"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(0, 20, 200)
    series = [dict(name="decay", color="#e24a33", kind="line", x=xs, y=3 * np.exp(-xs / 7)),
              dict(name="growth", color="#348abd", kind="line", x=xs, y=1 + 0.1 * xs)]
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=(7.68, 5.76), dpi=dpi)
        for s in series:
            ax.plot(s["x"], s["y"], color=s["color"], label=s["name"])
        ax.legend(loc="lower left")
        return _mpl_truth(fig, ax, series, dpi=dpi)


def web_chart(size=(850, 600)):
    """OWID 같은 웹 차트: 축선 없음, 연한 파선 가로 격자, 왼쪽 '90 years' 꼴 단위 붙은 눈금 글자(선 왼쪽 끝에 붙음),
    아래 x 글자의 첫·끝 글자는 그림 안쪽으로 당겨 찍음, 제목·부제·출처 글자, 선 + 작은 점 표식 2계열 (Pillow)"""
    import numpy as np
    W, H = size
    bx0, by0, bx1, by1 = 72, 110, 724, 532
    xr, yr = (1950, 2020), (0, 90)
    px = lambda x: bx0 + (x - xr[0]) / (xr[1] - xr[0]) * (bx1 - bx0)  # noqa: E731
    py = lambda y: by1 - (y - yr[0]) / (yr[1] - yr[0]) * (by1 - by0)  # noqa: E731
    im = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(im)
    fn, ft = _font(14), _font(22)
    d.text((16, 18), "Life expectancy", fill=(30, 30, 30), font=ft)
    d.text((16, 56), "Period life expectancy at birth, years", fill=(90, 90, 90), font=fn)
    for v in range(0, 91, 10):
        y = round(py(v))
        if v == 0:
            d.line([(bx0, y), (bx1, y)], fill=(200, 200, 200), width=1)
        else:
            for x in range(bx0, bx1, 8):
                d.line([(x, y), (min(bx1, x + 3), y)], fill=(225, 225, 225), width=1)
        d.text((bx0 - 6, y), f"{v} years", fill=(90, 90, 90), font=fn, anchor="rm")
    for v in range(1950, 2021, 10):
        x = round(px(v))
        anchor = "lt" if v == 1950 else "rt" if v == 2020 else "mt"
        d.text((x, by1 + 9), str(v), fill=(90, 90, 90), font=fn, anchor=anchor)
    xs = np.arange(1950, 2021)
    series = [dict(name="Japan", color=(196, 140, 76), kind="line", x=xs, y=60 + 25 * (1 - np.exp(-(xs - 1950) / 25))),
              dict(name="India", color=(45, 120, 160), kind="line", x=xs, y=41 + 0.42 * (xs - 1950))]
    for s in series:
        pts = [(px(x), py(y)) for x, y in zip(s["x"], s["y"])]
        d.line(pts, fill=s["color"], width=2, joint="curve")
        for x, y in pts[::2]:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=s["color"])
        d.text((bx1 + 6, pts[-1][1]), s["name"], fill=s["color"], font=fn, anchor="lm")
    d.text((16, H - 30), "Data source: synthetic (selftest)", fill=(120, 120, 120), font=fn)
    b = io.BytesIO()
    im.save(b, "PNG")
    calib = {"x": {"p1": [px(xr[0]) + 0.5, by1], "p2": [px(xr[1]) + 0.5, by1], "v1": xr[0], "v2": xr[1], "log": False},
             "y": {"p1": [bx0, py(yr[0]) + 0.5], "p2": [bx0, py(yr[1]) + 0.5], "v1": yr[0], "v2": yr[1], "log": False}}
    return b.getvalue(), {"calib": calib, "box": {"x0": bx0, "y0": by0, "x1": bx1, "y1": by1}, "series": series,
                          "xticks": list(range(1950, 2021, 10)), "yticks": list(range(0, 91, 10))}


def pie_chart(size=(640, 480), donut=False):
    """파이·도넛 그래프 (거절 시험)"""
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    cx, cy, r = 320, 240, 170
    a0 = 0
    for frac, col in ((0.15, (31, 119, 180)), (0.30, (255, 127, 14)), (0.45, (44, 160, 44)), (0.10, (214, 39, 40))):
        d.pieslice([cx - r, cy - r, cx + r, cy + r], a0, a0 + 360 * frac, fill=col)
        a0 += 360 * frac
    if donut:
        d.ellipse([cx - 90, cy - 90, cx + 90, cy + 90], fill="white")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def colorbar_scatter(dpi=100):
    """색 띠(컬러바)가 붙은 2D 산점도 — 지원하는 그림 (AI 가 히트맵이라 해도 거절하지 않아야 함)"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    x, y = rng.random(30), rng.random(30)
    fig, ax = plt.subplots(figsize=(7.68, 5.76), dpi=dpi)
    sc = ax.scatter(x, y, c=x + y, cmap="viridis")
    fig.colorbar(sc)
    series = [dict(name="pts", color="#000000", kind="points", x=x, y=y)]
    return _mpl_truth(fig, ax, series, dpi=dpi)


def dashed_lines(dpi=100):
    """실선·파선·점선 3계열이 서로 가로지름 (점선 틈 잇기·교차점 계열 유지 시험)"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(0, 10, 300)
    series = [dict(name="k=1", color="#1f77b4", kind="line", ls="-", x=xs, y=np.sqrt(xs)),
              dict(name="k=1.5", color="#ff7f0e", kind="line", ls="--", x=xs, y=4.2 - 0.35 * xs),
              dict(name="k=2", color="#2ca02c", kind="line", ls=":", x=xs, y=1 + 0.25 * xs)]
    fig, ax = plt.subplots(figsize=(7.68, 5.76), dpi=dpi)
    for s in series:
        ax.plot(s["x"], s["y"], color=s["color"], ls=s["ls"], label=s["name"])
    ax.legend(loc="lower center")
    return _mpl_truth(fig, ax, series, dpi=dpi)


def errorbar_hollow(dpi=100):
    """같은 색 오차막대(줄기·캡)가 달린 빈 원 표식 — 캡·줄기를 점으로 잡지 않아야 함"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(1, 9)
    y = 6.7 - 0.3 * x
    fig, ax = plt.subplots(figsize=(7.68, 5.76), dpi=dpi)
    ax.errorbar(x, y, yerr=0.6, fmt="o", mfc="white", color="#ff7f0e", capsize=3, label="run")
    ax.legend(loc="lower left")
    series = [dict(name="run", color="#ff7f0e", kind="points", x=x, y=y)]
    return _mpl_truth(fig, ax, series, dpi=dpi)


def bw_model_exp(dpi=150):
    """흑백 논문 그림: 검은 모형 곡선(표식 없음) + 같은 색 실험 점(빈 원), 테두리 없는 범례"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xm = np.linspace(0, 1, 101)
    xe = np.linspace(0.1, 0.9, 9)
    series = [dict(name="model", color="#000000", kind="line", x=xm, y=100 * xm ** 2),
              dict(name="exp.", color="#000000", kind="points", x=xe, y=100 * xe)]
    fig, ax = plt.subplots(figsize=(7, 5.4), dpi=dpi)
    ax.plot(xm, 100 * xm ** 2, color="black", lw=1.5, label="model")
    ax.plot(xe, 100 * xe, ls="none", marker="o", mfc="white", mec="black", mew=1.5, ms=7, label="exp.")
    ax.legend(loc="upper left", frameon=False)
    return _mpl_truth(fig, ax, series, dpi=dpi)


# ── 2026-10-20 막대 개선 시험용 (가로 막대·히스토그램·범례에 가린 누적·값 라벨만·음수·색깔 막대) ─────────────
def bar_case(kind, dpi=100):
    """참값을 아는 막대그래프. kind:
    hbar_err   — 가로 막대 + 막대를 지나는 가로 오차 막대 (오차 줄이 축 상자로 잘못 잡히던 경우)
    hbar_group — 가로 묶음 막대(두 색이 맞닿음) + 범례
    hist       — 검은 테두리로 맞붙은 히스토그램 + 높이 0 인 칸, 숫자 x 눈금
    hist_flat  — 테두리 없이 맞붙은 히스토그램(높이 계단으로만 나뉨)
    stack_leg  — 누적 막대 꼭대기를 반투명 범례가 가림
    labels_h   — 값축 눈금 없이 막대 끝에 값만 찍힌 가로 막대 (웹 차트 꼴)
    neg_h      — 0 양쪽으로 뻗는 가로 막대 + 0 선 + 값 라벨 + 바닥 회색 선 (축선 없음)
    neg_v      — 음수 세로 막대 (축선은 아래·왼쪽, 0 선 없음)
    colors     — 범례 없이 막대마다 색이 다른 단순 막대 (+ 막대 꼭대기에 맞닿은 값 글자)
    → (PNG, truth) truth = {orient, ticks, xticks, lim, xlim, bars: [{tops, series, err}], n_series, labels}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=dpi)
    orient, ns, labels, xt = "v", 1, None, None
    if kind == "hbar_err":
        orient = "h"
        v, e = [5.6, 5.5, 11.7, 10.6, 3.2], [0.9, 0.4, 0.9, 0.7, 0.3]
        y = np.arange(5)
        ax.barh(y, v, xerr=e, align="center")
        ax.set_yticks(y, ["Tom", "Dick", "Harry", "Slim", "Jim"])
        ax.invert_yaxis()
        bars = [{"tops": [v[i]], "series": [0], "err": (v[i] - e[i], v[i] + e[i])} for i in range(5)]
        bars = bars[::-1]  # 아래 → 위 순서로 (가로 막대 truth 는 아래부터, 검출은 위부터)
    elif kind == "hbar_group":
        orient = "h"
        a1, a2 = [12, 15, 9, 20], [8, 11, 14, 10]
        y = np.arange(4)
        ax.barh(y - 0.2, a1, 0.4, label="2023")
        ax.barh(y + 0.2, a2, 0.4, label="2024")
        ax.set_yticks(y, ["Q1", "Q2", "Q3", "Q4"])
        ax.legend(loc="lower right")
        bars = []
        for i in range(4):
            bars += [{"tops": [a1[i]], "series": [0], "err": None}, {"tops": [a2[i]], "series": [1], "err": None}]
        ns = 2
    elif kind in ("hist", "hist_flat"):
        cnt = [3, 0, 8, 21, 47, 80, 66, 30, 12, 5] if kind == "hist" else [4, 9, 22, 40, 33, 18, 7]
        edges = np.linspace(-2.5, 2.5, len(cnt) + 1)
        w = edges[1] - edges[0]
        ax.bar(edges[:-1] + w / 2, cnt, w, edgecolor="black" if kind == "hist" else None, color="#1f77b4", lw=1)
        bars = [{"tops": [float(c)], "series": [0], "err": None, "c": float(edges[i] + w / 2)} for i, c in enumerate(cnt)]
        xt = None  # 그린 뒤 x 범위 안 눈금만 (아래)
    elif kind == "stack_leg":
        data = np.array([[30, 20, 5], [28, 22, 8], [25, 23, 12], [22, 24, 17]], float)
        bottom = np.zeros(4)
        cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for j in range(3):
            ax.bar(range(4), data[:, j], 0.6, bottom=bottom, color=cols[j], edgecolor="black", label=["coal", "gas", "solar"][j])
            bottom += data[:, j]
        ax.set_xticks(range(4), ["2019", "2020", "2021", "2022"])
        ax.legend(loc="upper right", framealpha=0.8)
        bars = [{"tops": list(np.cumsum(r)), "series": [0, 1, 2], "err": None} for r in data]
        ns = 3
    elif kind == "labels_h":
        orient = "h"
        v = [84.7, 83.7, 82.2, 81.0, 77.0]
        y = np.arange(5)[::-1]
        ax.barh(y, v, 0.65, color="#6d87b0")
        for yy, x in zip(y, v):
            ax.text(x + 0.8, yy, f"{x:g} years", va="center", fontsize=9, color="#555555")
        ax.set_yticks(y, ["Japan", "South Korea", "France", "Germany", "United States"])
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.set_xlim(0, 100)
        bars = [{"tops": [x], "series": [0], "err": None} for x in v][::-1]
        labels = v
    elif kind == "neg_h":
        orient = "h"
        v = [125000, -75000, -275000, -475000]
        y = np.arange(4)[::-1]
        ax.barh(y, v, 0.6, color=["#2a78d6" if x > 0 else "#e34948" for x in v])
        ax.axvline(0, color="black", lw=1)
        for yy, x in zip(y, v):
            ax.text(x + (8000 if x > 0 else -8000), yy, f"{'+' if x > 0 else '−'}€ {abs(x) // 1000}.000", va="center",
                    ha="left" if x > 0 else "right", fontsize=9)
        ax.set_yticks(y, ["Arbeid", "Studie", "Gezin", "Asiel"])
        ax.set_xlim(-650000, 300000)
        ax.set_xticks([-400000, -200000, 0, 200000], ["−400.000", "−200.000", "0", "+200.000"])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.tick_params(length=0)
        bars = [{"tops": [float(x)], "series": [0], "err": None} for x in v][::-1]
        labels = v
    elif kind == "neg_v":
        v = [4.0, -2.5, 6.5, -5.0, 1.5]
        ax.bar(range(5), v, 0.6, color="#4c72b0")
        ax.set_xticks(range(5), ["A", "B", "C", "D", "E"])
        ax.set_ylim(-6, 8)
        bars = [{"tops": [x], "series": [0], "err": None} for x in v]
    elif kind == "colors":
        v = [100, 88, 97, 85, 90]
        cs = ["#4f81bc", "#c0504f", "#9bbb59", "#8165a2", "#4aacc5"]
        ax.bar(range(5), v, 0.7, color=cs)
        for i, x in enumerate(v):
            ax.text(i, x + 0.3, f"{x} ± 2", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(5), ["TCP", "A", "B", "C", "316L"])
        ax.set_ylim(0, 105)
        bars = [{"tops": [float(x)], "series": [0], "err": None} for x in v]
    else:
        raise ValueError(kind)
    fig.tight_layout()
    lim = ax.get_ylim() if orient == "v" else ax.get_xlim()
    tk = ax.get_yticks() if orient == "v" else ax.get_xticks()
    tk = [float(t) for t in tk if min(lim) - 1e-9 <= t <= max(lim) + 1e-9]
    xlim = [float(x) for x in ax.get_xlim()]
    if kind in ("hist", "hist_flat"):
        xt = [float(t) for t in ax.get_xticks() if min(xlim) <= t <= max(xlim)]
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    plt.close(fig)
    return b.getvalue(), {"orient": orient, "ticks": tk, "xticks": xt, "lim": [float(x) for x in lim], "xlim": xlim, "bars": bars,
                          "n_series": ns, "labels": labels}


# ── 2026-10-20 잡음·논문 그림 개선 시험용 ─────────────────────────────────────
def noise_case(kind, dpi=100):
    """참값을 아는 그림. kind:
    mult_offset — y 값이 1e-5 자릿수: matplotlib 이 눈금은 0…2.5, 축 위 끝에 '1e−5' 배수 글자를 따로 찍음
    mult_label  — 눈금은 0…8, 축 이름에 '(×10³)' 배수
    log_one     — 로그 y 축에 숫자 글자가 10¹ 하나뿐 (나머지는 작은 눈금)
    occluded    — 선 A 의 왼쪽 절반이 같은 경로의 선 B 밑에 완전히 가림 (A 는 범례에 있음)
    halo        — 범례 없는 검은 곡선 하나 + 회색 격자 (번짐·격자를 계열로 잡지 않아야 함)
    bubbles     — 범례 없는 반투명 거품(크기 여러 개) 산점 한 계열 (채움·테두리·겹친 색이 계열로 갈라지지 않아야 함)
    sharey      — y축을 같이 쓰는 1×3 칸 (안쪽 칸엔 y 글자 없음, x 눈금은 칸마다 0…8)
    → (PNG, 참값 dict: calib, box, series, xticks, yticks, ...)"""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = np.linspace(0, 10, 200)
    if kind == "mult_offset":
        s = [dict(name="decay", color="#1f77b4", kind="line", x=xs, y=2.4e-5 * np.exp(-xs / 4))]
        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=dpi)
        ax.plot(xs, s[0]["y"], color="#1f77b4")
        ax.set_ylim(0, 2.6e-5)
        p, t = _mpl_truth(fig, ax, s, dpi=dpi)
        t["yticks_printed"] = [round(v * 1e5, 6) for v in t["yticks"]]
        t["mult_text"] = "1e-5"
        return p, t
    if kind == "mult_label":
        s = [dict(name="load", color="#d62728", kind="line", x=xs, y=4000 + 3000 * np.sin(xs / 2))]
        fig, ax = plt.subplots(figsize=(7, 4.8), dpi=dpi)
        ax.plot(xs, s[0]["y"] / 1000, color="#d62728")
        ax.set_ylim(0, 8)
        ax.set_ylabel("Load (×10³ N)")
        p, t = _mpl_truth(fig, ax, s, dpi=dpi)
        t["calib"]["y"].update(v1=t["calib"]["y"]["v1"] * 1000, v2=t["calib"]["y"]["v2"] * 1000)
        t["yticks_printed"] = [float(v) for v in t["yticks"]]
        t["yticks"] = [v * 1000 for v in t["yticks"]]
        t["y_label"] = "Load (×10³ N)"
        return p, t
    if kind == "log_one":
        x = np.arange(300, 1501, 100)
        s = [dict(name="k", color="#1f77b4", kind="line", x=x, y=30 * np.exp(-(x - 300) / 600))]
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=dpi)
        ax.plot(x, s[0]["y"], "o-", color="#1f77b4")
        ax.set_yscale("log")
        ax.set_ylim(2.6, 34)
        ax.tick_params(which="both", direction="in", right=True, top=True)
        p, t = _mpl_truth(fig, ax, s, ylog=True, dpi=dpi)
        t["yticks"] = [10.0]
        return p, t
    if kind == "occluded":
        x = np.logspace(0, 3, 300)
        ya = 100 * x ** -0.32
        yb = np.where(x < 10, ya, 100 * 10 ** -0.32)
        s = [dict(name="Wright", color="#1f77b4", kind="line", x=x[x <= 60], y=ya[x <= 60]),
             dict(name="Plateau", color="#ff7f0e", kind="line", x=x, y=yb)]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax.plot(x, ya, color="#1f77b4", lw=1.6, label="Wright")
        ax.plot(x, yb, color="#ff7f0e", lw=1.6, label="Plateau")  # 나중에 그려 위에 덮음
        ax.set_xscale("log")
        ax.set_ylim(25, 110)
        ax.legend(loc="upper right")
        return _mpl_truth(fig, ax, s, xlog=True, dpi=dpi)
    if kind == "halo":
        rng = np.random.default_rng(5)
        y = np.cumsum(rng.normal(0, 1, 200))
        s = [dict(name="walk", color="#000000", kind="line", x=xs, y=y)]
        fig, ax = plt.subplots(figsize=(9.6, 4), dpi=dpi)
        ax.plot(xs, y, color="black", lw=1)
        ax.grid(True, alpha=0.4)
        ax.set_title("random walk")
        return _mpl_truth(fig, ax, s, dpi=dpi)
    if kind == "bubbles":
        rng = np.random.default_rng(11)
        x, y = rng.uniform(15, 100, 20), rng.uniform(0, 48, 20)
        s = [dict(name="pts", color="#e67d7e", kind="points", x=x, y=y)]
        fig, ax = plt.subplots(figsize=(7.68, 5.76), dpi=dpi)
        ax.scatter(x, y, s=rng.uniform(40, 400, 20), color="tab:red", alpha=0.6, edgecolors="black")
        return _mpl_truth(fig, ax, s, dpi=dpi)
    if kind == "sharey":
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.7), dpi=dpi, sharey=True)
        xi = np.arange(10)
        truth = []
        for i, (ax, c) in enumerate(zip(axs, ("#1f77b4", "#ff7f0e", "#2ca02c"))):
            yi = (xi * (0.4 + 0.3 * i)) % 9
            ax.plot(xi, yi, "o-", color=c)
            truth.append(dict(name=f"p{i}", color=c, kind="line", x=xi, y=yi))
        fig.tight_layout()
        fig.canvas.draw()
        H = fig.canvas.get_width_height()[1]
        boxes = []
        for ax in axs:
            bb = ax.get_window_extent()
            boxes.append([bb.x0, H - bb.y1, bb.x1, H - bb.y0])
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=dpi)
        y0, y1 = axs[0].get_ylim()
        x0, x1 = axs[0].get_xlim()
        xt = [float(t) for t in axs[0].get_xticks() if x0 <= t <= x1]
        yt = [float(t) for t in axs[0].get_yticks() if y0 <= t <= y1]
        plt.close(fig)
        return b.getvalue(), {"series": truth, "boxes": boxes, "xlim": (x0, x1), "ylim": (y0, y1),
                              "xticks": xt * 3, "xticks1": xt, "yticks": yt}
    raise ValueError(kind)


def relative_bars(size=(680, 450)):
    """값축 숫자·값 라벨 없는 묶음 세로 막대(월 × 3계열, 맞붙은 세 색), 연한 가로 격자, 아래 색 점 범례 (Pillow).
    참값: 격자 간격 = 20, 가장 높은 격자선 = 100 → 막대 높이(상대값)"""
    W, H = size
    im = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(im)
    fn = _font(12)
    x0, x1, base, top = 44, 636, 326, 158
    for k in range(1, 6):
        y = base - k * (base - top) / 5
        d.line([(x0, y), (x1, y)], fill=(225, 225, 225), width=1)
    d.line([(x0, base), (x1, base)], fill=(150, 150, 150), width=1)
    cols = [(251, 29, 167), (252, 211, 24), (107, 29, 172)]
    vals = [[90, 46, 24], [42, 91, 51], [42, 91, 69], [42, 110, 39], [24, 46, 55], [68, 50, 39],
            [69, 80, 15], [118, 86, 64], [24, 69, 89], [68, 106, 46], [34, 65, 79], [95, 56, 27]]
    gw = (x1 - x0) / 12
    bars = []
    for m, vs in enumerate(vals):
        for j, v in enumerate(vs):
            l = x0 + 6 + m * gw + j * 14
            h = v / 100 * (base - top)
            d.rectangle([l, base - h, l + 13, base - 1], fill=cols[j])
            bars.append({"m": m, "s": j, "v": v, "c": l + 7})
        d.text((x0 + 6 + m * gw + 21, base + 8), f"{m + 1}월", fill=(40, 40, 40), font=fn, anchor="mt")
    for j, nm in enumerate(("팝", "재즈", "록")):
        cx = 190 + j * 110
        d.ellipse([cx - 8, 362, cx + 8, 378], fill=cols[j])
        d.text((cx + 14, 370), nm, fill=(40, 40, 40), font=fn, anchor="lm")
    d.text((W // 2, 92), "인기 음악", fill=(0, 0, 0), font=_font(26), anchor="mm")
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue(), {"vals": vals, "bars": bars, "colors": ["#fb1da7", "#fcd318", "#6b1dac"], "names": ["팝", "재즈", "록"]}


# ── 2026-10-20 이중축 막대 + 선 시험용 ───────────────────────────────────────
def combo_case(kind, dpi=100):
    """참값을 아는 막대 콤보 그림 (matplotlib). kind:
    bar_line_twin — 막대(왼쪽 축) + 선 계열(오른쪽 축, 막대 위를 지남), 오른쪽 눈금 글자는 선 색, 범례 하나
    bars_both     — 맞붙은 좁은 두 색 막대: 한 계열은 왼쪽 선형축, 다른 계열은 오른쪽 로그축(막대가 축 아래 끝에서 섬), 축 글자 색 = 계열 색
    dashed_across — 한 축 묶음 막대 + 막대를 가로지르는 검은 추세 파선(범례 항목)
    → (PNG, truth) truth = {ticks, ticks2, lim, lim2, log2, bars: [{c_px, top, series, axis}], names, colors, axes, lines: [{name, axis, px, val}]}"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=dpi)
    ax2, lines, handles = None, [], []
    if kind == "bar_line_twin":
        x = np.arange(8)
        v = [12, 18, 25, 22, 35, 41, 38, 52]
        w = [220, 340, 310, 560, 610, 790, 880, 1050]
        handles.append(ax.bar(x, v, 0.6, color="#4c72b0", label="Sales", zorder=2))
        ax.set_ylim(0, 60)
        ax.set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"])
        ax2 = ax.twinx()
        handles += ax2.plot(x, w, color="#dd5511", lw=2.2, label="Visitors", zorder=5)
        ax2.set_ylim(0, 1200)
        ax2.tick_params(axis="y", colors="#dd5511")
        ax2.set_ylabel("Visitors", color="#dd5511")
        ax.set_ylabel("Sales")
        series = [("Sales", "#4c72b0", "left", v, x, 0.0)]
        lines = [("Visitors", "right", x, w)]
        names, colors, axes = ["Sales", "Visitors"], ["#4c72b0", "#dd5511"], ["left", "right"]
    elif kind == "bars_both":
        x = np.arange(10)
        a = [2, 3, 1, 4, 6, 5, 9, 12, 15, 21]
        b = [3, 40, 12, 300, 900, 150, 2500, 6000, 4000, 30000]
        bw = 0.3
        handles.append(ax.bar(x - bw / 2, a, bw, color="#8b008b", label="Blocks"))
        ax.set_ylim(0, 25)
        ax.tick_params(axis="y", colors="#8b008b")
        ax.set_ylabel("Blocks", color="#8b008b")
        ax2 = ax.twinx()
        ax2.set_yscale("log")
        handles.append(ax2.bar(x + bw / 2, b, bw, color="#008080", label="Edits"))
        ax2.set_ylim(0.6, 50000)
        ax2.tick_params(axis="y", colors="#008080")
        ax2.set_ylabel("Edits (log)", color="#008080")
        ax.set_xticks(x, [f"M{i + 1}" for i in x])
        series = [("Blocks", "#8b008b", "left", a, x - bw / 2, 0.0), ("Edits", "#008080", "right", b, x + bw / 2, 0.0)]
        names, colors, axes = ["Blocks", "Edits"], ["#8b008b", "#008080"], ["left", "right"]
    elif kind == "dashed_across":
        x = np.arange(6)
        a = [30, 45, 52, 61, 70, 84]
        b = [20, 38, 41, 55, 66, 73]
        bw = 0.38
        handles.append(ax.bar(x - bw / 2, a, bw, color="#2a9d8f", label="North"))
        handles.append(ax.bar(x + bw / 2, b, bw, color="#e9c46a", label="South"))
        tr_ = 28 + 10.5 * x
        handles += ax.plot(x, tr_, "k--", lw=2, label="Trend", zorder=5)
        ax.set_ylim(0, 100)
        ax.set_xticks(x, ["2019", "2020", "2021", "2022", "2023", "2024"])
        series = [("North", "#2a9d8f", "left", a, x - bw / 2, 0.0), ("South", "#e9c46a", "left", b, x + bw / 2, 0.0)]
        lines = [("Trend", "left", x, tr_)]
        names, colors, axes = ["North", "South", "Trend"], ["#2a9d8f", "#e9c46a", "#000000"], ["left", "left", "left"]
    else:
        raise ValueError(kind)
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    H = fig.canvas.get_width_height()[1]

    def px(axx, xv, yv):
        p = axx.transData.transform(np.column_stack([xv, yv]))
        return p[:, 0], H - p[:, 1]
    bars = []
    for nm, col, side, vals, xs, _ in series:
        cx, _ = px(ax, xs, np.zeros(len(xs)))
        for c_, t_ in zip(cx, vals):
            bars.append({"c_px": float(c_), "top": float(t_), "series": nm, "axis": side})
    bars.sort(key=lambda b_: b_["c_px"])
    lt = []
    for nm, side, xs, ys in lines:
        axx = ax2 if side == "right" else ax
        xx = np.linspace(xs[0], xs[-1], 200)
        yy = np.interp(xx, xs, ys)
        cx, _ = px(axx, xx, yy)
        lt.append({"name": nm, "axis": side, "px": [float(q) for q in cx], "val": [float(q) for q in yy]})
    lim = [float(q) for q in ax.get_ylim()]
    tk = [float(t) for t in ax.get_yticks() if lim[0] - 1e-9 <= t <= lim[1] + 1e-9]
    lim2 = [float(q) for q in ax2.get_ylim()] if ax2 else None
    tk2 = [float(t) for t in ax2.get_yticks() if lim2[0] - 1e-9 <= t <= lim2[1] * (1 + 1e-9)] if ax2 else []
    b_ = io.BytesIO()
    fig.savefig(b_, format="png", dpi=dpi)
    plt.close(fig)
    return b_.getvalue(), {"ticks": tk, "ticks2": tk2, "lim": lim, "lim2": lim2, "log2": bool(ax2 and ax2.get_yscale() == "log"),
                           "bars": bars, "names": names, "colors": colors, "axes": axes, "lines": lt}
