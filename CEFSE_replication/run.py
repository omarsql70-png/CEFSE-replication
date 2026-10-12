"""CEFSE computational evaluation on line-level defect prediction.

For each project: train a hierarchical attention-pooling defect model on release 1, select the epoch and tune
SAE hyper-parameters on release 2, evaluate file-level prediction, uncertainty (MC dropout) and line-level
explanations (Random, Recent-change, Attention, LIME, SHAP, SAE and its ablations) on releases 3+.
"""
import os, sys, time, json, pickle, collections, math
import numpy as np
import jax, jax.numpy as jnp, optax

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
D, T_MAX, BATCH, EPOCHS, LR, P_DROP, MC_T = 64, 50, 16, 15, 5e-4, 0.3, 20
JAVA_KW = __import__("prep").JAVA_KW

# ---------------------------------------------------------------- model
def init_params(key, V):
    k = jax.random.split(key, 5)
    g = lambda kk, s: jax.random.normal(kk, s) * (1.0 / math.sqrt(s[0]))
    return dict(E=jax.random.normal(k[0], (V, D)) * 0.1, Wh=g(k[1], (D, D)), bh=jnp.zeros(D),
                Wa=g(k[2], (D, D)), ba=jnp.zeros(D), va=g(k[3], (D,)), wo=g(k[4], (D,)), bo=jnp.zeros(()))

def line_vectors(p, tok, msk):
    emb = p["E"][tok] * msk[..., None]
    mean = emb.sum(1) / jnp.maximum(msk.sum(1, keepdims=True), 1.0)
    return jnp.tanh(mean @ p["Wh"] + p["bh"])

def attn_logits(p, h):
    return jnp.tanh(h @ p["Wa"] + p["ba"]) @ p["va"]

def forward(p, tok, msk, seg, nseg, key=None, drop=0.0):
    h = line_vectors(p, tok, msk)
    if key is not None and drop > 0:
        k1, k2 = jax.random.split(key)
        h = h * jax.random.bernoulli(k1, 1 - drop, h.shape) / (1 - drop)
    e = attn_logits(p, h)
    valid = msk.sum(1) > 0
    e = jnp.where(valid, e, -1e9)
    m = jax.ops.segment_max(e, seg, nseg)
    ex = jnp.exp(e - m[seg]) * valid
    den = jax.ops.segment_sum(ex, seg, nseg)
    a = ex / jnp.maximum(den[seg], 1e-12)
    z = jax.ops.segment_sum(a[:, None] * h, seg, nseg)
    if key is not None and drop > 0:
        z = z * jax.random.bernoulli(k2, 1 - drop, z.shape) / (1 - drop)
    return z @ p["wo"] + p["bo"]

def loss_fn(p, tok, msk, seg, y, w, key):
    logit = forward(p, tok, msk, seg, BATCH + 1, key, P_DROP)[:BATCH]
    l = optax.sigmoid_binary_cross_entropy(logit, y) * w
    return l.sum() / jnp.maximum(w.sum(), 1.0)

grad_fn = jax.jit(jax.value_and_grad(loss_fn))
fwd_eval = jax.jit(forward, static_argnums=(4,))
fwd_mc = jax.jit(lambda p, tok, msk, seg, key: forward(p, tok, msk, seg, BATCH + 1, key, P_DROP))
lv_jit = jax.jit(line_vectors)
al_jit = jax.jit(attn_logits)

def bucket(n):
    b = 256
    while b < n: b *= 2
    return b

def make_batch(files, idx, vocab):
    toks, seg = [], []
    for j, i in enumerate(idx):
        for t in files[i]["toks"]:
            toks.append([vocab.get(x, 1) for x in t]); seg.append(j)
    L = bucket(max(len(toks), 1))
    tok = np.zeros((L, T_MAX), np.int32); msk = np.zeros((L, T_MAX), np.float32)
    sg = np.full(L, BATCH, np.int32)
    for r, (t, s) in enumerate(zip(toks, seg)):
        tok[r, :len(t)] = t; msk[r, :len(t)] = 1; sg[r] = s
    y = np.zeros(BATCH, np.float32); w = np.zeros(BATCH, np.float32)
    for j, i in enumerate(idx):
        y[j] = files[i]["bug"]; w[j] = 1.0
    return tok, msk, sg, y, w

def predict(p, files, vocab, key=None):
    """Deterministic probs (key=None) or MC-dropout samples [MC_T, n]."""
    n = len(files); idxs = [list(range(i, min(i + BATCH, n))) for i in range(0, n, BATCH)]
    if key is None:
        out = []
        for idx in idxs:
            tok, msk, sg, _, _ = make_batch(files, idx, vocab)
            out.append(np.asarray(jax.nn.sigmoid(fwd_eval(p, tok, msk, sg, BATCH + 1)))[:len(idx)])
        return np.concatenate(out)
    samples = np.zeros((MC_T, n))
    for idx in idxs:
        tok, msk, sg, _, _ = make_batch(files, idx, vocab)
        for t in range(MC_T):
            key, k = jax.random.split(key)
            samples[t, idx] = np.asarray(jax.nn.sigmoid(fwd_mc(p, tok, msk, sg, k)))[:len(idx)]
    return samples

def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all(): return float("nan")
    r = np.argsort(np.argsort(s)) + 1
    # average ranks for ties
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

def train(files_tr, files_va, vocab, seed):
    key = jax.random.PRNGKey(seed); key, k0 = jax.random.split(key)
    p = init_params(k0, len(vocab) + 2)
    opt = optax.adamw(LR, weight_decay=1e-3); st = opt.init(p)
    ytr = np.array([f["bug"] for f in files_tr]); pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    rng = np.random.default_rng(seed)
    best, best_p, hist = -1, p, []
    for ep in range(EPOCHS):
        # class-balanced sampling: oversample defective files to ~1:1 per epoch
        pos = np.where(ytr)[0]; neg = np.where(~ytr)[0]
        order = np.concatenate([neg, rng.choice(pos, len(neg), replace=True)]); rng.shuffle(order)
        order = order[: min(len(order), 800)]
        tl = []
        for b in range(0, len(order) - BATCH + 1, BATCH):
            tok, msk, sg, y, w = make_batch(files_tr, order[b:b + BATCH], vocab)
            key, k = jax.random.split(key)
            l, g = grad_fn(p, tok, msk, sg, y, w, k)
            upd, st = opt.update(g, st, p); p = optax.apply_updates(p, upd); tl.append(float(l))
        va = auc([f["bug"] for f in files_va], predict(p, files_va, vocab))
        hist.append((ep, float(np.mean(tl)), va)); print(f"   ep{ep} loss={np.mean(tl):.4f} valAUC={va:.4f}", flush=True)
        if va > best: best, best_p = va, jax.tree_util.tree_map(lambda x: x, p)
    return best_p, hist

# ---------------------------------------------------------------- explanations (operate on cached line vectors)
def sig(x): return 1 / (1 + np.exp(-x))

class FileModel:
    def __init__(self, p, f, vocab):
        n = len(f["toks"]); L = bucket(n)
        tok = np.zeros((L, T_MAX), np.int32); msk = np.zeros((L, T_MAX), np.float32)
        for r, t in enumerate(f["toks"]):
            ids = [vocab.get(x, 1) for x in t]; tok[r, :len(ids)] = ids; msk[r, :len(ids)] = 1
        Hf = lv_jit(p, tok, msk)
        self.H = np.asarray(Hf, np.float64)[:n]
        e = np.asarray(al_jit(p, Hf), np.float64)[:n]
        self.w = np.exp(e - e.max()); self.wo = np.asarray(p["wo"], np.float64); self.bo = float(p["bo"])
        self.Hw = self.H @ self.wo  # per-line contribution to the logit numerator
    def prob_masks(self, M):  # M: [k, n] 0/1
        num = M @ (self.w * self.Hw); den = M @ self.w
        z = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
        return sig(z + self.bo)
    def prob_full(self): return float(self.prob_masks(np.ones((1, len(self.w))))[0])
    def attention(self): return self.w / self.w.sum()
    def contribution(self):
        """Exact per-line share of the logit: a_l * (h_l . w_o) (attention weighted by the value vector)."""
        return self.attention() * self.Hw

def lime_scores(fm, rng, n_samples=2000, width=0.25):
    n = len(fm.w)
    frac = rng.uniform(0, 1, (n_samples, 1))
    M = (rng.uniform(0, 1, (n_samples, n)) < frac).astype(float); M[0] = 1
    y = fm.prob_masks(M)
    d = 1 - np.sqrt(M.sum(1) / n)
    sw = np.exp(-d ** 2 / width ** 2)
    X = M - (sw @ M) / sw.sum(); yc = y - (sw @ y) / sw.sum()
    r = np.sqrt(sw); Xs = X * r[:, None]; ys = yc * r
    if n <= n_samples:
        return np.linalg.solve(Xs.T @ Xs + np.eye(n), Xs.T @ ys)
    return Xs.T @ np.linalg.solve(Xs @ Xs.T + np.eye(n_samples), ys)  # dual form of weighted ridge

def shap_scores(fm, rng, n_perm=200):
    """Shapley values by antithetic permutation sampling with O(n) prefix evaluation per permutation."""
    n = len(fm.w); phi = np.zeros(n); p0 = sig(fm.bo)
    a, b = fm.w * fm.Hw, fm.w
    for k in range(n_perm // 2):
        perm = rng.permutation(n)
        for pi in (perm, perm[::-1]):
            num = np.cumsum(a[pi]); den = np.cumsum(b[pi])
            pr = sig(num / den + fm.bo)
            prev = np.concatenate([[p0], pr[:-1]])
            phi[pi] += pr - prev
    return phi / n_perm

def coref_graph(toks):
    """Lines are linked when they share an identifier that occurs in at most 15 lines of the file."""
    occ = collections.defaultdict(list)
    for i, t in enumerate(toks):
        for x in set(t):
            if x[0].isalpha() or x[0] in "_$":
                if x not in JAVA_KW: occ[x].append(i)
    rows, cols = [], []
    for x, ls in occ.items():
        if 2 <= len(ls) <= 15:
            for i in ls:
                for j in ls:
                    if i != j: rows.append(i); cols.append(j)
    return rows, cols

def ppr(n, rows, cols, seed_vec, alpha=0.85, iters=30):
    if not rows: return seed_vec.copy()
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel(); Dinv = np.where(deg > 0, 1 / np.maximum(deg, 1), 0)
    P = sp.diags(Dinv) @ A
    r = seed_vec.copy()
    for _ in range(iters):
        r = alpha * (P.T @ r) + (1 - alpha) * seed_vec + alpha * seed_vec * (deg == 0)
    return r

def sae_components(fm, f, att_type="raw"):
    if att_type == "raw":
        a = fm.attention()
    else:
        c = fm.contribution(); a = np.maximum(c, 0)
        if a.sum() <= 0: a = fm.attention()
        a = a / a.sum()
    n = len(a)
    rows, cols = coref_graph(f["toks"])
    g = ppr(n, rows, cols, a)
    delta = np.array([1.0 if ln in f["changed"] else 0.0 for ln in f["line_no"]])
    if f.get("is_new"): delta[:] = 0.0  # a new file has no change history to exploit
    return a / a.max(), g / max(g.max(), 1e-12), delta

def sae_score(comp, lam, beta):
    a, g, delta = comp
    return (lam * a + (1 - lam) * g) * (1 + beta * delta)

# ---------------------------------------------------------------- line-level metrics
def line_metrics(score, f, rng, fm=None):
    n = len(score); ln = np.array(f["line_no"]); y = np.array([x in f["defective"] for x in ln])
    tie = rng.uniform(0, 1e-9, n)
    order = np.argsort(-(score + tie)); ys = y[order]
    k20 = max(1, int(math.ceil(0.2 * n)))
    rec20 = ys[:k20].sum() / max(y.sum(), 1)
    cum = np.cumsum(ys); need = math.ceil(0.2 * y.sum())
    eff20 = (np.argmax(cum >= need) + 1) / n if y.sum() > 0 else np.nan
    ifa = int(np.argmax(ys)) if y.any() else np.nan
    p10 = ys[:10].mean()
    out = dict(recall20=rec20, effort20=eff20, ifa=ifa, prec10=p10, n_lines=n, n_def=int(y.sum()))
    if fm is not None:
        full = fm.prob_full(); M = np.ones((1, n)); M[0, order[:10]] = 0
        out["comp10"] = full - float(fm.prob_masks(M)[0])
    return out, order[:10]

# ---------------------------------------------------------------- main
def run_project(proj, data, seed=0):
    rels = list(data.keys()); tr, va, tests = rels[0], rels[1], rels[2:]
    cnt = collections.Counter(x for f in data[tr] for t in f["toks"] for x in t)
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(20000)) if c >= 2}
    print(f"[{proj}] train={tr} val={va} test={tests} vocab={len(vocab)}", flush=True)
    t0 = time.time(); p, hist = train(data[tr], data[va], vocab, seed); t_train = time.time() - t0
    res = dict(project=proj, train=tr, val=va, tests=tests, hist=hist, train_sec=t_train, releases={})
    rng = np.random.default_rng(seed)

    # --- tune SAE (lam, beta) and uncertainty thresholds on the validation release
    pv = predict(p, data[va], vocab)
    grid_l, grid_b = [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.5, 1.0, 2.0, 4.0]
    tp_va = [f for f, q in zip(data[va], pv) if f["bug"] and q >= 0.5 and f["defective"]]
    fms_va = [(f, FileModel(p, f, vocab)) for f in tp_va]
    comps_by = {at: [(f, sae_components(fm, f, at)) for f, fm in fms_va] for at in ("raw", "contrib")}
    vr = lambda comps, l, b: np.mean([line_metrics(sae_score(c, l, b), f, np.random.default_rng(5))[0]["recall20"] for f, c in comps]) if comps else 0
    best = (-1, "raw", 1.0, 0.0)
    for at, comps in comps_by.items():
        for lam in grid_l:
            for beta in grid_b:
                r = vr(comps, lam, beta)
                if r > best[0]: best = (r, at, lam, beta)
    AT = best[1]; comps = comps_by[AT]
    lam_b = max(grid_b, key=lambda b: vr(comps, 1.0, b))
    lam_l = max(grid_l, key=lambda l: vr(comps, l, 0.0))
    res["sae_params"] = dict(att_type=AT, lam=best[2], beta=best[3], val_recall20=float(best[0]), noGraph_beta=lam_b, noTemporal_lam=lam_l)
    best = (best[0], best[2], best[3])
    mc_va = predict(p, data[va], vocab, jax.random.PRNGKey(seed + 100))
    ent = lambda q: -(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12))
    u_va = ent(mc_va.mean(0))
    res["tau"] = [float(np.quantile(u_va, 0.5)), float(np.quantile(u_va, 0.8))]
    print(f"[{proj}] SAE params {res['sae_params']}", flush=True)

    for rel in tests:
        files = data[rel]; y = np.array([f["bug"] for f in files])
        pdet = predict(p, files, vocab)
        mc = predict(p, files, vocab, jax.random.PRNGKey(seed + 7))
        pm = mc.mean(0); u_tot = ent(pm); u_ale = ent(mc).mean(0); u_epi = u_tot - u_ale
        R = dict(n_files=len(files), n_bug=int(y.sum()), y=y.tolist(), p_det=pdet.tolist(), p_mc=pm.tolist(),
                 u_tot=u_tot.tolist(), u_epi=u_epi.tolist(), u_ale=u_ale.tolist(), lines=[])
        tp = [f for f, q in zip(files, pdet) if f["bug"] and q >= 0.5 and f["defective"]]
        print(f"[{proj}] {rel}: files={len(files)} bug={y.sum()} AUC={auc(y, pdet):.3f} TPfiles={len(tp)}", flush=True)
        for f in tp:
            fm = FileModel(p, f, vocab); comp = sae_components(fm, f, AT); row = dict(file=f["file"], n_changed=int(comp[2].sum()))
            methods = {}
            t = time.perf_counter(); methods["Random"] = rng.uniform(size=len(fm.w)); row["t_Random"] = time.perf_counter() - t
            methods["RecentChange"] = comp[2] + rng.uniform(0, 1e-6, len(fm.w))
            t = time.perf_counter(); methods["Attention"] = fm.attention(); row["t_Attention"] = time.perf_counter() - t
            methods["AttnContrib"] = fm.contribution()
            t = time.perf_counter(); methods["LIME"] = lime_scores(fm, np.random.default_rng(1)); row["t_LIME"] = time.perf_counter() - t
            t = time.perf_counter(); methods["SHAP"] = shap_scores(fm, np.random.default_rng(1)); row["t_SHAP"] = time.perf_counter() - t
            t = time.perf_counter(); c2 = sae_components(fm, f, AT)
            methods["SAE"] = sae_score(c2, best[1], best[2]); row["t_SAE"] = time.perf_counter() - t
            methods["SAE-noGraph"] = sae_score(comp, 1.0, lam_b)
            methods["SAE-noTemporal"] = sae_score(comp, lam_l, 0.0)
            tops = {}
            for m, s in methods.items():
                met, top = line_metrics(s, f, np.random.default_rng(3), fm)
                for k, v in met.items(): row[f"{m}_{k}"] = v
                tops[m] = set(top.tolist())
            # stability of stochastic explainers across seeds (Jaccard of top-10 lines)
            for m, fn in (("LIME", lime_scores), ("SHAP", shap_scores)):
                s2 = fn(fm, np.random.default_rng(2)); _, top2 = line_metrics(s2, f, np.random.default_rng(3))
                t2 = set(top2.tolist()); row[f"{m}_stab10"] = len(tops[m] & t2) / max(len(tops[m] | t2), 1)
            R["lines"].append(row)
        res["releases"][rel] = R
    with open(os.path.join(OUT, f"{proj}.json"), "w") as fh:
        json.dump(res, fh, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[{proj}] done in {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    for pj in sys.argv[1:]:
        with open(os.path.join(HERE, f"data_{pj}.pkl"), "rb") as fh:
            run_project(pj, pickle.load(fh))
