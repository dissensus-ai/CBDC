"""Generate publication PDF figures from adjudicated-results.json only."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
a = json.loads(
    (ROOT / "revision-2026-09-12/evidence/adjudicated-results.json").read_text()
)
out = ROOT / "revision-2026-09-12/manuscript/figures"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "axes.labelcolor": "#382D30",
    }
)
colors = ["#741E35", "#38747B"]
fig, ax = plt.subplots(figsize=(7.6, 3.1), layout="constrained")
for j, k in enumerate(("dI", "dW")):
    v = [a["confirmatory"][m][k] for m in ("logit", "gboost")]
    x = np.arange(2) + (j - 0.5) * 0.23
    ax.barh(
        x,
        [q["mean"] for q in v],
        height=0.21,
        color=colors[j],
        alpha=0.75,
        label=["Customer attributes", "Watchlist"][j],
    )
    ax.errorbar(
        [q["mean"] for q in v],
        x,
        xerr=[[q["mean"] - q["ci_lo"] for q in v], [q["ci_hi"] - q["mean"] for q in v]],
        fmt="o",
        color=colors[j],
        capsize=4,
    )
ax.set_yticks([0, 1], ["Logistic regression", "Gradient boosting"])
ax.invert_yaxis()
ax.set_xlabel("Fewer missed illicit entities per 10,000")
ax.legend(frameon=False)
ax.grid(axis="x", alpha=0.2)
fig.savefig(out / "independent.pdf")
plt.close(fig)
fig, ax = plt.subplots(figsize=(7.6, 3.5), layout="constrained")
d = a["ladder"]["b=mid|s=mid|p=0.05"]
models = list(d)
for j, k in enumerate(("dI_ap", "dW_ap")):
    v = [d[m][k] for m in models]
    x = np.arange(4) + (j - 0.5) * 0.23
    ax.errorbar(
        x,
        [q["mean"] for q in v],
        yerr=[[q["mean"] - q["ci_lo"] for q in v], [q["ci_hi"] - q["mean"] for q in v]],
        fmt="o",
        color=colors[j],
        capsize=4,
        label=["Customer attributes", "Watchlist"][j],
    )
ax.set_xticks(range(4), ["Linear", "Additive", "Forest", "Boosted"])
ax.set_ylabel("Average-precision increment")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
fig.savefig(out / "ladder.pdf")
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2), layout="constrained")
levels = ("low", "mid", "high")
for ax, m, title in zip(axes.flat, models, ["Linear", "Additive", "Forest", "Boosted"]):
    grid = np.array(
        [
            [a["ladder"][f"b={b}|s={s}|p=0.05"][m]["dF_ap"]["mean"] for s in levels]
            for b in levels
        ]
    )
    im = ax.imshow(grid, vmin=-0.01, vmax=0.14, cmap="YlOrRd", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if grid[i, j] > 0.09 else "#34252A",
            )
    ax.set_title(title)
    ax.set_xticks(range(3), ["Low", "Mid", "High"])
    ax.set_yticks(range(3), ["Low", "Mid", "High"])
    ax.set_xlabel("Auxiliary signal")
    ax.set_ylabel("Behavior recoverability")
fig.colorbar(im, ax=axes, label="Full-access AP increment", shrink=0.85)
fig.savefig(out / "surface.pdf")
plt.close(fig)
print("3 PDF figures generated")
