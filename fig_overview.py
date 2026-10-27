"""
Figure 1: Framework overview schematic.
Shows the full architecture: substation SFR model <-> N rack-level agents,
with one agent's pipeline expanded (local sensing -> comm channel subject to
latency/jitter/packet loss -> SoC-aware availability gating -> event-trigger
decision with fixed/adaptive threshold -> zero-order-hold power command),
aggregated back into the substation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "serif"})

fig, ax = plt.subplots(figsize=(7.2, 3.55))
ax.set_xlim(0, 14.4)
ax.set_ylim(0, 7.1)
ax.axis("off")


def box(x, y, w, h, text, fc="#EFEFEF", ec="0.3", fs=7.6, lw=0.9, ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, family="serif")
    return (x, y, w, h)


def arrow(p0, p1, color="0.25", lw=1.1, style="-|>", connectionstyle=None, mutation_scale=10):
    kw = dict(arrowstyle=style, mutation_scale=mutation_scale, lw=lw, color=color)
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(p0, p1, **kw))


# ---- Substation (left) ----
box(0.25, 3.55, 2.55, 1.7,
    "Substation SFR model\n$2H\\dot{\\Delta f}=\\Delta P_m-\\Delta P_L$\n$-D\\Delta f+P_{\\mathrm{bess}}$",
    fc="#E4F2E1", fs=7.3)

# ---- Rack-agent fleet dashed box (right side), representative agent i expanded ----
fleet_x, fleet_y, fleet_w, fleet_h = 3.35, 0.25, 10.8, 6.6
ax.add_patch(Rectangle((fleet_x, fleet_y), fleet_w, fleet_h, fill=False,
                        ec="0.55", lw=1.0, linestyle=(0, (4, 2))))
ax.text(fleet_x + fleet_w / 2 + 0.6, fleet_y + fleet_h - 0.28,
        "PULSE rack-level UPS/BESS agents ($N=200$) \u2014 agent $i$ pipeline shown",
        ha="center", va="center", fontsize=7.6, family="serif")

# pipeline blocks (single agent i)
py = 3.55
box(3.60, py, 2.00, 1.55, "Local PLL/PMU\nfilter ($\\tau_f,\\tau_r$)\n+ noise $w_i(t)$\n$\\widehat{\\Delta f}_i,\\widehat{r}_i$", fc="#DCEAF7", fs=6.7)
box(5.90, py, 2.25, 1.55,
    "Comm. channel\nlatency $\\tau_i{\\sim}\\mathcal{U}(1,8)$ms\n+ packet loss $p_{\\mathrm{drop}}$\n+ jitter (log-normal)",
    fc="#FDEFD8", fs=6.7)
box(8.45, py, 2.35, 1.55,
    "PULSE event trigger\n$|p_i^\\ast-p_i|>\\eta_i(t)P_{\\max}$\nFETC ($\\eta_0$) / PULSE ($\\eta_i(t)$)",
    fc="#FBE8E1", fs=6.5)
box(11.10, py, 2.65, 1.55,
    "SoC-aware gate\n$a_i=\\mathrm{clip}\\!\\left(\\frac{\\mathrm{SoC}_i-\\mathrm{SoC}_{\\min}}{\\mathrm{margin}},0,1\\right)$",
    fc="#F2EAF8", fs=6.4)

# ZOH command / output row
box(8.45, 1.35, 5.30, 1.35,
    "Zero-order-hold power command\n$p_i(t)=a_i\\,\\mathrm{sat}(-K_d\\widehat{r}_i-K_p\\widehat{\\Delta f}_i,\\pm P_{\\max})$",
    fc="#DCEAF7", fs=7.0)

# aggregation block
box(3.60, 1.35, 4.00, 1.35, "Aggregate injection\n$P_{\\mathrm{bess}}(t)=\\sum_i p_i(t)$", fc="#E4F2E1")

# ---- arrows within pipeline ----
arrow((5.60, py + 0.78), (5.90, py + 0.78))
arrow((8.15, py + 0.78), (8.45, py + 0.78))
arrow((10.80, py + 0.78), (11.10, py + 0.78))
# down from gate to ZOH command
arrow((12.42, py), (12.42, 2.70), connectionstyle="arc3,rad=0.15")
arrow((9.62, py), (10.15, 2.70), connectionstyle="arc3,rad=-0.15")
# ZOH -> aggregation
arrow((8.45, 2.02), (7.60, 2.02))
# aggregation -> substation (feedback)
arrow((3.60, 2.02), (2.8, 3.55), connectionstyle="arc3,rad=-0.25")
# substation frequency -> local sensing (feed-forward, top loop)
arrow((1.5, 5.25), (1.5, 6.45))
ax.add_patch(FancyArrowPatch((1.5, 6.45), (4.6, 6.45), arrowstyle="-", lw=1.1, color="0.25"))
arrow((4.6, 6.45), (4.6, py + 1.55))
ax.text(2.45, 6.65, "local frequency $\\Delta f(t)$", ha="center", fontsize=6.8, family="serif")

# ---- N agents cue: small stacked squares behind the pipeline to suggest replication ----
for k, dy in enumerate([0.12, 0.06, 0.0]):
    alpha = 0.35 + 0.2 * k
    ax.add_patch(Rectangle((3.55 - dy, py - dy), 10.20, 1.55, fill=False,
                            ec="0.75", lw=0.6, alpha=alpha, zorder=0))

fig.tight_layout(pad=0.2)
fig.savefig("figs/fig0_overview.pdf", bbox_inches="tight")
fig.savefig("figs/fig0_overview.png", dpi=400, bbox_inches="tight")
print("Overview figure saved.")
