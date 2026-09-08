#!/usr/bin/env python3
"""Static charts for the benchmark package (GitHub renders SVG in markdown).
Palette: the dataviz reference instance — sequential blue steps 200/350/450/600
for 1..4 drives, categorical slot 1 (blue) / slot 2 (orange) for drafter off/on;
text and axis in text tokens, never the series colour. Values from the package
result files (results/SCALING.md, RESULTS-2026-09-08-standard-lengths.md, PREFILL.md)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

SURFACE, T1, T2, MUTED, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SEQ = ["#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]      # 1,2,3,4 drives (light->dark)
C_OFF, C_ON = "#2a78d6", "#eb6834"                       # categorical slots 1, 2
OUT = "k3-public-bench/results/charts"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": BASE,
                     "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2,
                     "text.color": T1, "svg.fonttype": "none"})

def style(ax, ylabel):
    ax.set_facecolor(SURFACE); ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.yaxis.grid(True, color="#ecebe6", linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0); ax.set_ylabel(ylabel, color=T2)

def rounded_bar(ax, x, h, w, color, gap=0.0):
    # thin bar, 4px-ish rounded data end, square at the baseline, no border
    if h <= 0: return
    r = min(0.02 * ax.get_ylim()[1] if ax.get_ylim()[1] else 0.02, h / 2)
    p = FancyBboxPatch((x - w/2, 0), w, h, boxstyle=f"round,pad=0,rounding_size={w*0.18}",
                       linewidth=0, facecolor=color, mutation_aspect=1)
    ax.add_patch(p)

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight", facecolor=SURFACE, dpi=160)
    plt.close(fig); print("wrote", name)

# ---- 1. drive-count ladder: grouped bars, x = test, one bar per drive count ----
tests = ["17-token prompt\n(median of 3)", "128 tokens\ndrafter on", "128 tokens\ndrafter off",
         "512 tokens, drafter on\n(shared 300-token prefix)", "512 tokens\ndrafter off"]
vals = {  # 1,2,3,4 drives
    0: [0.5483, 0.7485, 0.8869, 0.9631], 1: [0.5350, 0.7596, 0.9375, 1.0377],
    2: [0.4269, 0.6245, 0.7855, 0.8630], 3: [0.5160, 0.7382, 0.9201, 1.0271],
    4: [0.4127, 0.6326, None, 0.9063]}
fig, ax = plt.subplots(figsize=(10, 4.6)); style(ax, "tokens per second (inclusive)")
ax.set_ylim(0, 1.2)
w, gap = 0.17, 0.03
for ti in range(5):
    for di in range(4):
        v = vals[ti][di]
        x = ti + (di - 1.5) * (w + gap)
        if v is None:
            ax.text(x, 0.03, "not\nrun", ha="center", va="bottom", fontsize=7, color=MUTED); continue
        ax.bar(x, v, width=w, color=SEQ[di], linewidth=0)
        if di == 3:  # direct label only on the four-drive reference bar
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=T2)
ax.set_xticks(range(5)); ax.set_xticklabels(tests, fontsize=8.5)
from matplotlib.patches import Patch
ax.legend([Patch(color=c) for c in SEQ], ["1 drive", "2 drives", "3 drives", "4 drives"],
          frameon=False, ncol=4, loc="upper left", fontsize=9, handlelength=1.2, handleheight=1.2)
ax.set_title("Decode speed by number of drives serving the experts (same engine, prompts and settings)",
             loc="left", fontsize=11, color=T1, pad=12)
ax.text(0, -0.36, "1 drive ≈ 52%, 2 full mirrors ≈ 73%, 3 drives ≈ 90% of four-drive speed. Four-drive rows are medians of 2 (17-token: 3); 1–3-drive generation rows are single runs.",
        transform=ax.transAxes, fontsize=8, color=MUTED)
save(fig, "drive-ladder")

# ---- 2. steady decode by length and drafter (four drives) ----
fig, ax = plt.subplots(figsize=(6.4, 4.2)); style(ax, "tokens per second (steady)")
ax.set_ylim(0, 1.3)
lengths = ["128 generated tokens", "512 generated tokens"]
off = [0.9261, 0.9232]; on = [1.1252, 1.0015]
w = 0.28
for i in range(2):
    ax.bar(i - w/2 - 0.01, off[i], width=w, color=C_OFF, linewidth=0)
    ax.bar(i + w/2 + 0.01, on[i], width=w, color=C_ON, linewidth=0)
    ax.text(i - w/2 - 0.01, off[i] + 0.02, f"{off[i]:.2f}", ha="center", fontsize=9, color=T2)
    ax.text(i + w/2 + 0.01, on[i] + 0.02, f"{on[i]:.2f}", ha="center", fontsize=9, color=T2)
    ax.text(i, 1.22, f"drafter gain +{100*(on[i]/off[i]-1):.1f}%", ha="center", fontsize=9, color=T2)
ax.set_xticks([0, 1]); ax.set_xticklabels(lengths)
ax.legend([Patch(color=C_OFF), Patch(color=C_ON)], ["drafter off (plain decode)", "drafter on"],
          frameon=False, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.32))
ax.set_title("Plain decode is flat with answer length; the speculative advantage decays", loc="left", fontsize=11, pad=12)
save(fig, "steady-by-length")

# ---- 3. prefill: first token vs drives, and where the four-drive 373 s go ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
style(a1, "seconds to first token, 512-token prompt"); a1.set_ylim(0, 900)
drives = ["1 drive\n13.6 GB/s", "2 drives\n20.7 GB/s", "4 drives\n~33 GB/s"]; tt = [799, 545, 375]
for i, v in enumerate(tt):
    a1.bar(i, v, width=0.5, color=SEQ[[0, 1, 3][i]], linewidth=0)
    a1.text(i, v + 15, f"{v} s  ({v/60:.1f} min)", ha="center", fontsize=9, color=T2)
a1.set_xticks(range(3)); a1.set_xticklabels(drives, fontsize=9)
a1.set_title("First token scales with combined read bandwidth", loc="left", fontsize=11, pad=12)
# stacked single horizontal bar: where the 373 s go
style(a2, ""); a2.set_xlim(0, 100); a2.set_ylim(-0.6, 0.6); a2.yaxis.grid(False); a2.xaxis.grid(True, color="#ecebe6")
parts = [("waiting for expert reads", 76, SEQ[3]), ("expert kernels", 16, SEQ[2]), ("attention", 5, SEQ[1]), ("other", 3, SEQ[0])]
left = 0
for name, pct, col in parts:
    a2.barh(0, pct - 0.5, left=left, height=0.5, color=col, linewidth=0)
    if pct >= 10: a2.text(left + pct/2, 0, f"{name}\n{pct}%", ha="center", va="center", fontsize=8.5, color=SURFACE if pct > 40 else T1)
    left += pct
a2.text(87 + 3, 0.42, "attention 5% · other 3%", fontsize=8, color=MUTED, ha="right")
a2.set_yticks([]); a2.set_xlabel("share of the 373 s prefill on four drives (engine timers)", color=T2, fontsize=9)
a2.set_title("Where the four-drive prefill goes: 9.0 TB read for a 1.4 TB expert set (6.2×)", loc="left", fontsize=10.5, pad=12)
save(fig, "prefill")

# ---- 4. per-drive draw under the engine vs standalone ceiling (like the dashboard's peak/average bars) ----
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs)
drives = ["internal SSD\n(2 TB, primary)", "WD SN8100 1 TB\n(direct, hot replica)", "WD SN8100 2 TB\n(direct, full second base)", "WD SN7100 1 TB\n(behind TB5 hub, cold replica)"]
median = [11.54, 5.87, 5.70, 4.31]   # 1-s windows, 200-token decode, p50
peak   = [13.73, 7.02, 6.81, 5.02]   # 1-s windows, max
ceil   = [13.55, 7.09, 7.04, 5.73]   # standalone, whole-file reads, no page cache
fig, ax = plt.subplots(figsize=(10, 4.6)); style(ax, "GB/s read"); ax.set_ylim(0, 16)
w, g = 0.22, 0.03
for i in range(4):
    for j, (series, col) in enumerate(((median, C1), (peak, C2), (ceil, C3))):
        x = i + (j - 1) * (w + g)
        ax.bar(x, series[i], width=w, color=col, linewidth=0)
        ax.text(x, series[i] + 0.25, f"{series[i]:.1f}", ha="center", fontsize=8, color=T2)
    ax.text(i, 15.2, f"peak = {100*peak[i]/ceil[i]:.0f}% of ceiling", ha="center", fontsize=8.5, color=T2)
ax.set_xticks(range(4)); ax.set_xticklabels(drives, fontsize=8.5)
ax.legend([Patch(color=C1), Patch(color=C2), Patch(color=C3)],
          ["median draw under the engine (1-s windows, 200-token decode)", "peak draw under the engine", "standalone ceiling (engine idle)"],
          frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8.5, handlelength=1.2, handleheight=1.2)
ax.set_title("Every drive runs at 90–100% of its own ceiling under the engine — the barrier, not bandwidth, sets the speed",
             loc="left", fontsize=11, pad=12)
ax.text(0, -0.36, "Draw from the per-device sampler on the 200-token record arm; ceilings from whole-file reads with no page cache, one drive at a time (results/SCALING.md).",
        transform=ax.transAxes, fontsize=8, color=MUTED)
save(fig, "drive-draw")
