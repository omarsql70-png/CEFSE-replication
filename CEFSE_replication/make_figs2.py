import os, numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__); A = os.path.join(HERE, "analysis"); F = os.path.join(HERE, "figures"); os.makedirs(F, exist_ok=True)
INK, MUTED, GRID = "#222222", "#666666", "#e3e3e0"
BLUE, GRAY, ORANGE = "#2a78d6", "#9a9a95", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
L = pd.read_csv(os.path.join(A, "line_level_rows.csv"))
ORDER = ["Random", "RecentChange", "Attention", "SHAP", "LIME", "AttnContrib", "SAE-noTemporal", "SAE-noGraph", "SAE"]
LABEL = {"Random": "Random", "RecentChange": "Recent change", "Attention": "Raw attention", "SHAP": "SHAP", "LIME": "LIME",
         "AttnContrib": "Contribution attention", "SAE-noTemporal": "SAE w/o temporal", "SAE-noGraph": "SAE w/o graph", "SAE": "SAE (full)"}
rng = np.random.default_rng(0)

def boot(x, B=2000):
    x = np.asarray(x); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def hbar(metric, xlabel, fname, fmt="{:.3f}"):
    stats = [boot(L[f"{m}_{metric}"].dropna()) for m in ORDER]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=300)
    y = np.arange(len(ORDER))
    cols = [BLUE if m.startswith("SAE") or m == "AttnContrib" else GRAY for m in ORDER]
    ax.barh(y, [s[0] for s in stats], color=cols, height=0.62)
    ax.errorbar([s[0] for s in stats], y, xerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
                fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
    for i, s in enumerate(stats):
        if s[0] >= 0: ax.text(s[2] + 0.004, i, fmt.format(s[0]), va="center", ha="left", fontsize=8.5)
        else: ax.text(s[1] - 0.004, i, fmt.format(s[0]), va="center", ha="right", fontsize=8.5)
    from matplotlib.ticker import MaxNLocator; ax.xaxis.set_major_locator(MaxNLocator(5))
    lo = min(s[1] for s in stats)
    if lo < 0: ax.set_xlim(left=lo - 0.035)
    ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in ORDER]); ax.set_xlabel(xlabel)
    ax.axvline(0, color=MUTED, lw=0.8); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(F, fname)); plt.close(fig)

hbar("recall20", "Recall@20%LOC (mean, 95% bootstrap CI)", "fig2_recall20.png")
hbar("comp10", "Comprehensiveness@10 (drop in predicted probability)", "fig4_comp10.png")
plt.close("all")

# per-project heatmap
P = pd.read_csv(os.path.join(A, "recall20_by_project.csv"), index_col=0)[ORDER]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)
im = ax.imshow(P.values, cmap="Blues", vmin=0, vmax=max(0.5, P.values.max()), aspect="auto")
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LABEL[m] for m in ORDER], rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(len(P))); ax.set_yticklabels(P.index, fontsize=8.5)
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        v = P.values[i, j]; ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.2, color="white" if v > 0.33 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Recall@20%LOC", fontsize=8)
ax.spines[:].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(F, "fig3_heatmap.png")); plt.close(fig)

# selective prediction
S = pd.read_csv(os.path.join(A, "selective.csv"))
fig, ax = plt.subplots(figsize=(5.0, 3.3), dpi=300)
ax.plot(S.coverage * 100, S.errrate_det * 100, color=GRAY, lw=2, marker="o", ms=5, label="Max-probability confidence (deterministic)")
ax.plot(S.coverage * 100, S.errrate_MC * 100, color=BLUE, lw=2, marker="s", ms=5, label="Predictive entropy (MC dropout)")
ax.invert_xaxis(); ax.set_xlabel("Coverage: share of files decided automatically (%)"); ax.set_ylabel("Error rate on retained files (%)")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(F, "fig5_selective.png")); plt.close(fig)
print("figs ok")
