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
