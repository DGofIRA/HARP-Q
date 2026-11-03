import numpy as np
import matplotlib.pyplot as plt
from .scoring import normalize


STYLE = {"font.family": "serif", "font.size": 9, "axes.spines.top": False,
         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}


def qualification_heatmaps(scored):
    scored = list(scored)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [5, 3]}, constrained_layout=True)
        for ax, keys, title in zip(axes, [("G", "U", "P", "S", "Psi"), ("Q1", "Q2", "Q3")], ["Capabilities and coupled penalty", "Role qualification"]):
            values = np.array([[row[k] for k in keys] for row in scored])
            im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(keys)), keys)
            ax.set_yticks(range(len(scored)), [row["candidate"] for row in scored])
            ax.set_title(title)
            for i, row in enumerate(values):
                for j, value in enumerate(row):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=8,
                            bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": .3})
        fig.colorbar(im, ax=list(axes), label="Relative score", shrink=.85)
    return fig


def stability_panels(inputs, scored):
    inputs, scored = list(inputs), list(scored)
    score_by_id = {row["candidate"]: row for row in scored}
    x = np.array([abs(np.log(row["bg"]["beta"])) for row in inputs])
    y = np.array([row["bg"]["rho"] for row in inputs])
    heat = np.column_stack([normalize(-np.maximum(y, 0)), normalize(-x), [score_by_id[row["candidate"]]["S"] for row in inputs]])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
        axes[0].scatter(x, y, c=range(len(inputs)), cmap="tab10", s=40)
        for row, a, b in zip(inputs, x, y):
            axes[0].annotate(row["candidate"], (a, b), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axes[0].set(xlabel="Bullwhip deviation |ln beta|", ylabel="Lag-one demand chasing", title="BG input summaries")
        axes[0].margins(x=.18, y=.15)
        im = axes[1].imshow(heat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        axes[1].set_xticks([0, 1, 2], ["Low chasing", "Low deviation", "S"])
        axes[1].set_yticks(range(len(inputs)), [row["candidate"] for row in inputs])
        axes[1].set_title("Midpoint scores and stability")
        for i, row in enumerate(heat):
            for j, value in enumerate(row):
                axes[1].text(j, i, f"{value:.3f}", ha="center", va="center", color="black", fontsize=8,
                             bbox={"facecolor": "white", "alpha": .82, "edgecolor": "none", "pad": .3})
        fig.colorbar(im, ax=axes[1], label="Relative score", shrink=.85)
    return fig


def nvp_panels(rows):
    rows = list(rows)
    x = np.arange(len(rows))
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), constrained_layout=True)
        for ax, metric in zip(axes, ["anchoring", "chasing"]):
            for frame, color in [("pf", "#0072B2"), ("nf", "#D55E00")]:
                for state, line, fill in [("before", "-", color), ("after", ":", "white")]:
                    ax.plot(x, [r[frame + "_" + state][metric] for r in rows], line, color=color,
                            marker="o", markersize=4, markerfacecolor=fill, label=frame.upper()+" "+state)
            ax.set_xticks(x, [r["candidate"] for r in rows])
            ax.set_ylabel(metric.capitalize()+" coefficient")
            ax.axhline(0, color="0.6", linewidth=.6)
        axes[0].legend(fontsize=7, frameon=False)
    return fig


def outcome_summary(rows, metrics=("cost", "bullwhip", "service"), label="condition"):
    rows = list(rows)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(3 * len(metrics), 3.4), squeeze=False, constrained_layout=True)
        for ax, metric in zip(axes[0], metrics):
            ax.errorbar(range(len(rows)), [r[metric]["mean"] for r in rows],
                        yerr=[r[metric]["sd"] for r in rows], fmt="o", capsize=3, color="#0072B2")
            ax.set_xticks(range(len(rows)), [r[label] for r in rows], rotation=25, ha="right")
            ax.set_ylabel(metric.replace("_", " ").capitalize()+" (mean ± SD)")
    return fig
