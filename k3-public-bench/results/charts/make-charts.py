#!/usr/bin/env python3
"""Static charts for the benchmark package (GitHub renders SVG in markdown; a
<picture> element in the READMEs swaps in the -dark variant for dark themes).

Design: a calm steel-blue ordinal ramp for 1..4 drives, one terracotta accent used
only for "drafter on", warm neutral for the plain-decode baseline; thin bars with
rounded data ends on a square baseline; recessive grid; text in text tokens, never
the series colour; title / subtitle / footnote hierarchy; system font stack.
Colour separation checked (OKLab dE >= 15 adjacent, CVD-simulated >= 8) in both modes.
Values from results/SCALING.md, RESULTS-2026-09-08-standard-lengths.md, PREFILL.md."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
import numpy as np

OUT = "k3-public-bench/results/charts"
FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(surface="#fcfcfb", t1="#0b0b0b", t2="#52514e", muted="#8a8880", grid="#eceae4", base="#cfcdc5",
                  ramp=["#bccbdd", "#7c9cbf", "#3f6a98", "#152f4d"], primary="#2f5d8a", accent="#c2683b",
                  neutral="#b3b1a8", peak="#a9bfd6", ref="#8a8880", tile="#ffffff", tileline="#e6e5df",
                  drives=["#2f5d8a", "#c2683b", "#1f8a7a", "#b8860b"], drives_tint=["#a6b8ca", "#e0b39e", "#9bccc5", "#dacb95"]),
    "dark":  dict(surface="#0d1117", t1="#e6edf3", t2="#b1bac4", muted="#7d8590", grid="#21262d", base="#30363d",
                  ramp=["#35527a", "#6a8fb5", "#a3c0dc", "#e0ecf7"], primary="#7fa6cf", accent="#e08a5e",
                  neutral="#6e6e68", peak="#3f5f83", ref="#9da7b3", tile="#161b22", tileline="#30363d",
                  drives=["#7fa6cf", "#e08a5e", "#4fbfa5", "#e0b84a"], drives_tint=["#40546a", "#6c4737", "#2b5f57", "#6c5c2e"]),
}

def rc(T):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": T["base"],
                         "axes.labelcolor": T["t2"], "xtick.color": T["t2"], "ytick.color": T["t2"],
                         "text.color": T["t1"], "svg.fonttype": "none", "figure.dpi": 100})

def style(ax, T, ylabel=""):
    ax.set_facecolor(T["surface"]); ax.figure.set_facecolor(T["surface"])
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["base"]); ax.spines["bottom"].set_linewidth(0.8)
    ax.yaxis.grid(True, color=T["grid"], linewidth=0.7); ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=8.5, pad=6)
    if ylabel: ax.set_ylabel(ylabel, color=T["t2"], fontsize=8.5, labelpad=8)

def bar(ax, x, h, w, color):
    """Thin bar, rounded data end (~4 px), square at the baseline."""
    if h is None or h <= 0: return
    fig = ax.figure; bb = ax.get_position()
    ax_w_px = bb.width * fig.get_figwidth() * fig.dpi; ax_h_px = bb.height * fig.get_figheight() * fig.dpi
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    px_x = (x1 - x0) / ax_w_px; px_y = (y1 - y0) / ax_h_px          # data units per pixel
    r = 4 * px_x                                                     # 4 px radius in x units
    aspect = px_y / px_x
    p = FancyBboxPatch((x - w/2, -h), w, 2*h, boxstyle=f"round,pad=0,rounding_size={r}",
                       mutation_aspect=aspect, linewidth=0, facecolor=color)
    ax.add_patch(p)
    p.set_clip_path(Rectangle((x - w, 0), 2*w, h, transform=ax.transData))

def titles(ax, T, title, subtitle=None, footnote=None, y=1.0):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=T["t1"], pad=26 if subtitle else 14)
    if subtitle:
        ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=9, color=T["t2"], va="bottom")
    if footnote:
        ax.text(0, -0.30, footnote, transform=ax.transAxes, fontsize=7.8, color=T["muted"], va="top")

def legend(ax, T, handles, labels, **kw):
    lg = ax.legend(handles, labels, frameon=False, fontsize=8.5, handlelength=1.1, handleheight=1.1,
                   labelcolor=T["t2"], **kw)
    return lg

def save(fig, name, T, mode):
    suffix = "" if mode == "light" else "-dark"
    path = f"{OUT}/{name}{suffix}.svg"
    fig.savefig(path, bbox_inches="tight", facecolor=T["surface"], pad_inches=0.18)
    if mode == "light": fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor=T["surface"], dpi=160, pad_inches=0.18)
    plt.close(fig)
    s = open(path).read()
    s = re.sub(r"font-family:\s*'?DejaVu Sans'?", f"font-family: {FONT_STACK}", s)
    open(path, "w").write(s); print("wrote", path)

def render(mode):
    T = THEMES[mode]; rc(T)

    # ---- 1. drive-count ladder ----
    tests = ["17-token prompt\n(median of 3)", "128 tokens\ndrafter on", "128 tokens\ndrafter off",
             "512 tokens, drafter on\n(shared 300-token prefix)", "512 tokens\ndrafter off"]
    vals = {0: [0.5483, 0.7485, 0.8869, 0.9631], 1: [0.5350, 0.7596, 0.9375, 1.0377],
            2: [0.4269, 0.6245, 0.7855, 0.8630], 3: [0.5160, 0.7382, 0.9201, 1.0271],
            4: [0.4127, 0.6326, None, 0.9063]}
    fig, ax = plt.subplots(figsize=(10, 4.8)); fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.2)
    style(ax, T, "tokens per second (inclusive)"); ax.set_xlim(-0.55, 4.55); ax.set_ylim(0, 1.25)
    w, gap = 0.15, 0.035
    for ti in range(5):
        for di in range(4):
            v = vals[ti][di]; x = ti + (di - 1.5) * (w + gap)
            if v is None:
                ax.text(x, 0.03, "not\nrun", ha="center", va="bottom", fontsize=7, color=T["muted"]); continue
            bar(ax, x, v, w, T["ramp"][di])
            if di == 3: ax.text(x, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=T["t2"])
    ax.set_xticks(range(5)); ax.set_xticklabels(tests); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    legend(ax, T, [Patch(color=c) for c in T["ramp"]], ["1 drive", "2 drives", "3 drives", "4 drives"],
           ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))
    titles(ax, T, "Decode speed by number of drives serving the experts",
           "Same engine, prompts and settings on every rung; the four-drive bar is labelled",
           "1 drive ≈ 52%, 2 full mirrors ≈ 73%, 3 drives ≈ 90% of four-drive speed. Four-drive rows are medians of 2 (17-token: 3); 1–3-drive generation rows are single runs.")
    save(fig, "drive-ladder", T, mode)

    # ---- 2. steady decode by length and drafter ----
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); fig.subplots_adjust(left=0.11, right=0.98, top=0.8, bottom=0.22)
    style(ax, T, "tokens per second (steady)"); ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, 1.35)
    off = [0.9261, 0.9232]; on = [1.1252, 1.0015]; w = 0.26
    for i in range(2):
        bar(ax, i - w/2 - 0.02, off[i], w, T["neutral"]); bar(ax, i + w/2 + 0.02, on[i], w, T["accent"])
        ax.text(i - w/2 - 0.02, off[i] + 0.025, f"{off[i]:.2f}", ha="center", fontsize=8.5, color=T["t2"])
        ax.text(i + w/2 + 0.02, on[i] + 0.025, f"{on[i]:.2f}", ha="center", fontsize=8.5, color=T["t2"])
        ax.text(i, 1.24, f"+{100*(on[i]/off[i]-1):.0f}% with the drafter", ha="center", fontsize=8.5, color=T["t2"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["128 generated tokens", "512 generated tokens"]); ax.set_yticks([0, 0.5, 1.0])
    legend(ax, T, [Patch(color=T["neutral"]), Patch(color=T["accent"])], ["drafter off (plain decode)", "drafter on"],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))
    titles(ax, T, "Plain decode is flat with answer length", "The speculative advantage decays as the answer grows")
    save(fig, "steady-by-length", T, mode)

    # ---- 3. prefill ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.05, 1.15], "wspace": 0.28})
    fig.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.2)
    style(a1, T, "seconds to first token, 512-token prompt"); a1.set_xlim(-0.6, 2.6); a1.set_ylim(0, 950)
    drives = ["1 drive\n13.6 GB/s", "2 drives\n20.7 GB/s", "4 drives\n~24 GB/s"]; tt = [799, 545, 375]
    for i, v in enumerate(tt):
        bar(a1, i, v, 0.46, T["ramp"][[0, 1, 3][i]])
        a1.text(i, v + 18, f"{v} s  ({v/60:.1f} min)", ha="center", fontsize=8.5, color=T["t2"])
    a1.set_xticks(range(3)); a1.set_xticklabels(drives); a1.set_yticks([0, 300, 600, 900])
    titles(a1, T, "First token scales with combined read bandwidth", "One cold run per rung")
    style(a2, T); a2.set_xlim(0, 100); a2.set_ylim(-0.7, 0.7); a2.yaxis.grid(False); a2.spines["bottom"].set_visible(False)
    parts = [("waiting for expert reads", 76, T["ramp"][3]), ("kernels", 16, T["ramp"][2]), ("attention", 5, T["ramp"][1]), ("other", 3, T["ramp"][0])]
    left = 0
    for name, pct, col in parts:
        a2.barh(0, pct - 0.6, left=left, height=0.42, color=col, linewidth=0)
        if pct >= 10:
            a2.text(left + pct/2, 0, f"{name}\n{pct}%", ha="center", va="center", fontsize=8.5,
                    color=(T["surface"] if (mode == "light" and pct > 40) else (T["t1"] if mode == "light" else "#0d1117")))
        left += pct
    a2.text(100, 0.36, "expert kernels 16% · attention 5% · other 3%", fontsize=8, color=T["muted"], ha="right")
    a2.set_yticks([]); a2.set_xticks([0, 25, 50, 75, 100]); a2.set_xticklabels(["0", "25", "50", "75", "100%"])
    a2.set_xlabel("share of the 373 s prefill on four drives (engine timers)", color=T["t2"], fontsize=8.5)
    titles(a2, T, "Where the four-drive prefill goes", "9.0 TB read for a 1.4 TB expert set — 6.2× re-read amplification")
    save(fig, "prefill", T, mode)

    # ---- 4. per-drive draw under the engine vs standalone ceiling (one hue per drive, as on the dashboard) ----
    drives = ["internal SSD\n(2 TB, primary)", "WD SN8100 1 TB\n(direct, hot replica)", "WD SN8100 2 TB\n(direct, full second base)", "WD SN7100 1 TB\n(behind TB5 hub, cold replica)"]
    median = [11.54, 5.87, 5.70, 4.31]; peak = [13.73, 7.02, 6.81, 5.02]; ceil = [13.55, 7.09, 7.04, 5.73]
    fig, ax = plt.subplots(figsize=(10, 4.8)); fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.24)
    style(ax, T, "GB/s read"); ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, 16.5)
    w, g = 0.24, 0.04
    for i in range(4):
        hue, tint = T["drives"][i], T["drives_tint"][i]
        for j, (series, col) in enumerate(((median, hue), (peak, tint))):
            x = i + (j - 0.5) * (w + g)
            bar(ax, x, series[i], w, col)
            ylab = series[i] + 0.3
            if series[i] + 0.15 < ceil[i] < series[i] + 1.0: ylab = ceil[i] + 0.3   # keep the number clear of the ceiling dash
            ax.text(x, ylab, f"{series[i]:.1f}", ha="center", fontsize=8.5, color=T["t2"])
        ax.plot([i - 0.38, i + 0.38], [ceil[i], ceil[i]], color=hue, linewidth=1.3, linestyle=(0, (3, 2)), solid_capstyle="butt")
        ax.text(i, 15.6, f"peak = {100*peak[i]/ceil[i]:.0f}% of ceiling ({ceil[i]:.1f})", ha="center", fontsize=8.3, color=T["t2"])
    ax.set_xticks(range(4)); ax.set_xticklabels(drives); ax.set_yticks([0, 4, 8, 12, 16])
    from matplotlib.lines import Line2D
    legend(ax, T, [Patch(color=T["t2"]), Patch(color=T["base"]), Line2D([0], [0], color=T["t2"], linewidth=1.3, linestyle=(0, (3, 2)))],
           ["median draw under the engine (1-s windows, 200-token decode)", "peak draw under the engine", "standalone ceiling (engine idle)"],
           loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    titles(ax, T, "Every drive runs at 88–100% of its own ceiling under the engine",
           "So the read barrier, not total bandwidth, sets the decode speed — one colour per drive, as on the dashboard",
           "Draw from the per-device sampler on the 200-token record arm; ceilings from whole-file reads with no page cache, one drive at a time (results/SCALING.md).")
    save(fig, "drive-draw", T, mode)

    # ---- 5. hero numbers strip ----
    tiles = [("1.00 tok/s", "steady decode, 512-token answer"), ("1.13 tok/s", "steady decode, 128 tokens"),
             ("0.96 tok/s", "17-token public prompt (upstream 0.68)"), ("6.3 min", "to first token, 512-token prompt")]
    fig = plt.figure(figsize=(10, 1.55)); fig.set_facecolor(T["surface"]); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.55)
    for i, (v, l) in enumerate(tiles):
        x0 = 0.08 + i * 2.48
        ax.add_patch(FancyBboxPatch((x0, 0.12), 2.34, 1.3, boxstyle="round,pad=0,rounding_size=0.1", mutation_aspect=0.155,
                                    facecolor=T["tile"], edgecolor=T["tileline"], linewidth=0.9))
        ax.text(x0 + 0.2, 0.92, v, fontsize=17, fontweight="bold", color=(T["t1"] if i < 3 else T["t2"]), va="center")
        ax.text(x0 + 0.2, 0.45, l, fontsize=7.9, color=T["t2"], va="center")
    save(fig, "hero", T, mode)

if __name__ == "__main__":
    for m in ("light", "dark"): render(m)
