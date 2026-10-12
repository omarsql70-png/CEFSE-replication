import json, glob, os, math
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata

HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "results"); OUT = os.path.join(HERE, "analysis")
os.makedirs(OUT, exist_ok=True)
METHODS = ["Random", "RecentChange", "Attention", "AttnContrib", "LIME", "SHAP", "SAE-noGraph", "SAE-noTemporal", "SAE"]
BASELINES = ["Random", "RecentChange", "Attention", "AttnContrib", "LIME", "SHAP", "SAE-noGraph", "SAE-noTemporal"]

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all(): return float("nan")
    r = rankdata(s); return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

def mcc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    tp = (y & p).sum(); tn = (~y & ~p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum()
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0

def bacc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, bool)
    return 0.5 * ((y & p).sum() / max(y.sum(), 1) + (~y & ~p).sum() / max((~y).sum(), 1))

def ece(y, q, bins=10):
    y = np.asarray(y, float); q = np.asarray(q, float); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (q >= edges[i]) & (q < edges[i + 1] if i < bins - 1 else q <= 1)
        if m.any(): e += m.mean() * abs(y[m].mean() - q[m].mean())
    return e

def holm(pvals):
    p = np.asarray(pvals); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj

def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = rankdata(np.abs(d)); return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())

res = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, "*.json")))]
# ---------------- file-level prediction + uncertainty
rows, allu = [], []
for r in res:
    for rel, R in r["releases"].items():
        y = np.array(R["y"], bool); pd_ = np.array(R["p_det"]); pm = np.array(R["p_mc"])
        rows.append(dict(project=r["project"], release=rel, files=R["n_files"], defective=R["n_bug"],
                         AUC=auc(y, pd_), AUC_MC=auc(y, pm), MCC=mcc(y, pd_ >= .5), BalAcc=bacc(y, pd_ >= .5),
                         TP_files_explained=len(R["lines"])))
        for i in range(len(y)):
            allu.append(dict(project=r["project"], release=rel, y=y[i], p_det=pd_[i], p_mc=pm[i], u_tot=R["u_tot"][i], u_epi=R["u_epi"][i],
                             tau1=r["tau"][0], tau2=r["tau"][1]))
filelvl = pd.DataFrame(rows); filelvl.to_csv(os.path.join(OUT, "file_level.csv"), index=False)
U = pd.DataFrame(allu)
err_det = (U.p_det >= .5) != U.y; err_mc = (U.p_mc >= .5) != U.y
conf_det = np.maximum(U.p_det, 1 - U.p_det)
unc = dict(
    ECE_det=ece(U.y, U.p_det), ECE_MC=ece(U.y, U.p_mc),
    AUROC_err_maxprob=auc(err_det, -conf_det), AUROC_err_entropy=auc(err_mc, U.u_tot), AUROC_err_epistemic=auc(err_mc, U.u_epi),
    MCC_det=mcc(U.y, U.p_det >= .5), MCC_MC=mcc(U.y, U.p_mc >= .5))
sel = []
for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    k = int(round(cov * len(U)))
    keep_mc = np.argsort(U.u_tot.values)[:k]; keep_det = np.argsort(-conf_det.values)[:k]
    sel.append(dict(coverage=cov, MCC_MC_entropy=mcc(U.y.values[keep_mc], U.p_mc.values[keep_mc] >= .5),
                    MCC_det_maxprob=mcc(U.y.values[keep_det], U.p_det.values[keep_det] >= .5),
                    errrate_MC=float(err_mc.values[keep_mc].mean()), errrate_det=float(err_det.values[keep_det].mean())))
sel = pd.DataFrame(sel); sel.to_csv(os.path.join(OUT, "selective.csv"), index=False)
lvl = np.where(U.u_tot < U.tau1, 1, np.where(U.u_tot < U.tau2, 2, 3))
lv = pd.DataFrame(dict(level=lvl, err=err_mc.values)).groupby("level").agg(share=("err", "size"), error_rate=("err", "mean"))
lv["share"] = lv["share"] / lv["share"].sum(); lv.to_csv(os.path.join(OUT, "levels.csv"))
json.dump(unc, open(os.path.join(OUT, "uncertainty.json"), "w"), indent=1)

# ---------------- line level
L = []
for r in res:
    for rel, R in r["releases"].items():
        for row in R["lines"]:
            row = dict(row); row["project"] = r["project"]; row["release"] = rel; L.append(row)
L = pd.DataFrame(L); L.to_csv(os.path.join(OUT, "line_level_rows.csv"), index=False)
summ = []
for m in METHODS:
    summ.append(dict(method=m, n=len(L), recall20=L[f"{m}_recall20"].mean(), recall20_sd=L[f"{m}_recall20"].std(),
                     effort20=L[f"{m}_effort20"].mean(), effort20_sd=L[f"{m}_effort20"].std(),
                     IFA_median=L[f"{m}_ifa"].median(), IFA_mean=L[f"{m}_ifa"].mean(), prec10=L[f"{m}_prec10"].mean(),
                     comp10=L[f"{m}_comp10"].mean(), comp10_sd=L[f"{m}_comp10"].std(),
                     stab10=L[f"{m}_stab10"].mean() if f"{m}_stab10" in L else (1.0 if m not in ("Random",) else np.nan),
                     ms_per_file=1000 * L[f"t_{m}"].mean() if f"t_{m}" in L else np.nan))
summ = pd.DataFrame(summ); summ.to_csv(os.path.join(OUT, "line_summary.csv"), index=False)
perproj = L.groupby("project")[[f"{m}_recall20" for m in METHODS]].mean(); perproj.columns = METHODS
perproj.to_csv(os.path.join(OUT, "recall20_by_project.csv"))
tests = []
for metric, better in [("recall20", 1), ("effort20", -1), ("ifa", -1), ("prec10", 1), ("comp10", 1)]:
    for b in BASELINES:
        x, y = L[f"SAE_{metric}"].values, L[f"{b}_{metric}"].values
        ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
        try: p = wilcoxon(x, y, zero_method="wilcox").pvalue
        except ValueError: p = 1.0
        tests.append(dict(metric=metric, baseline=b, SAE_mean=x.mean(), base_mean=y.mean(), p=p, r_rb=better * rank_biserial(x, y)))
T = pd.DataFrame(tests); T["p_holm"] = holm(T.p.values); T.to_csv(os.path.join(OUT, "wilcoxon.csv"), index=False)
params = pd.DataFrame([dict(project=r["project"], **r["sae_params"], tau1=r["tau"][0], tau2=r["tau"][1], train_min=r["train_sec"] / 60) for r in res])
params.to_csv(os.path.join(OUT, "sae_params.csv"), index=False)
pd.set_option("display.width", 200)
print(filelvl.round(3).to_string()); print(json.dumps(unc, indent=1)); print(sel.round(3).to_string()); print(lv.round(3))
print(summ.round(3).to_string()); print(perproj.round(3).to_string()); print(T.round(4).to_string()); print(params.to_string())
