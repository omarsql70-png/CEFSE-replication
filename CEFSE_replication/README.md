# CEFSE — replication package (line-level defect-prediction evaluation)

This package reproduces the computational evaluation reported in the revised manuscript
"A Computational Explainable AI Framework for Trustworthy Decision-Making in Software Engineering".

## Data
Public benchmark of Wattanakriengkrai et al. (IEEE TSE 2022), CC0 licence:
https://github.com/awsm-research/line-level-defect-prediction

```
git clone --depth 1 https://github.com/awsm-research/line-level-defect-prediction.git ext/line-level-defect-prediction
```

9 Apache projects, 32 releases. For every project the first release is used for training, the second for model
selection and SAE hyper-parameter tuning, and all remaining releases (14 in total) for testing.

## Environment
Python 3.11+, `jax` (CPU), `optax`, `numpy`, `scipy`, `pandas`, `matplotlib`; GNU `diff` must be on the PATH.

```
pip install jax optax numpy scipy pandas matplotlib
```

## Steps
```
python prep.py                     # parse releases, strip comments, tokenize, compute changed lines (GNU diff)
python run.py groovy camel derby hive wicket lucene jruby activemq hbase   # train, predict, explain
python analyze.py                  # tables, Wilcoxon signed-rank tests with Holm correction, uncertainty metrics
python make_figs2.py               # figures
```

`run.py` writes one JSON per project to `results/`; `analyze.py` writes CSV summaries to `analysis/`.

## Main settings (run.py)
| Setting | Value |
|---|---|
| Model | token embedding (64) → mean per line → tanh projection (64) → additive line attention → logistic output |
| Training | AdamW, lr 5e-4, weight decay 1e-3, batch 16 files, 15 epochs × 800 class-balanced files, dropout 0.3, best epoch by validation AUC |
| Vocabulary | training-release tokens with frequency ≥ 2, at most 20,000 |
| MC dropout | T = 20 stochastic passes |
| LIME | 2,000 line-mask samples, cosine-distance kernel width 0.25, weighted ridge (α = 1) |
| SHAP | Shapley values by 200 antithetic permutation samples |
| SAE grid | attention type ∈ {raw, contribution}, λ ∈ {0, .25, .5, .75, 1}, β ∈ {0, .5, 1, 2, 4}; selected on the validation release by mean Recall@20%LOC |
| Seed | 0 (explainer seeds 1 and 2 for the stability analysis) |

Hardware used: 2 vCPU, 8 GB RAM, no GPU.

Note: the Groovy project was used as a pilot run to debug the pipeline before the full run.
