# Reproducing the paper

This branch adds the analysis code needed to regenerate every figure and table in the manuscript.
Given the annotation data, the repository reproduces the paper end to end.

The annotation records themselves are **not** in this repository and cannot be released — they were
collected at an industrial energetic-materials facility. Everything else is here or is fetched
automatically. You need two files from the corresponding author before the real analysis will run.

---

## 1. Environments

```bash
python env_installer.py          # creates the six conda envs
```

Then add the packages the analysis scripts need, into `fsscore-env`:

```bash
conda run -n fsscore-env pip install --no-deps seaborn shap slicer cloudpickle
conda run -n fsscore-env pip install "numba==0.59.1"
```

`--no-deps` is deliberate — it stops pip upgrading `numpy`/`pandas`/`xgboost` underneath the
pipeline. **The `numba==0.59.1` pin matters**: it is the newest release compatible with the env's
`numpy 1.24.4`, and `shap` will not import without it.

> `rascore-env` is the fragile one (Python 3.7, `tensorflow-gpu==2.5.0`, `numpy 1.19.5`) and is the
> most likely thing to fail on a recent machine. Only `run_rascore_xgb.py` and
> `run_rascore_tensor.py` need it. If you cannot build it, everything except the RAscore columns
> still runs.

Verify:

```bash
conda run -n fsscore-env python -c "import shap, seaborn, xgboost, torch; print('ok')"
```

## 2. Data

| File | Where it goes | How to get it |
|---|---|---|
| `final_alpha_for_launch_pair_analysis_2026-03-22.csv` | `data/input/` | **From the corresponding author.** Restricted; never commit it. |
| Reference artifacts (26 files) | `data/reference/` | **From the corresponding author.** Used to verify your run; they contain expert consensus scores, so they are restricted too. |
| `EM_list.csv` | `data/input/` | The 416-compound Elton et al. reference set (public). |
| `data_moses_train.csv` | `data/input/` | Build it — see below (public). |

```python
# data_moses_train.csv, from the public MOSES release
import pandas as pd
df = pd.read_csv("https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv")
tr = df[df.SPLIT == "train"].reset_index(drop=True)   # 1,584,663 rows; keep this order
tr[["SMILES"]].to_csv("data/input/data_moses_train.csv", index=False)
```

Row order matters: the 10k sample is drawn with `sample(n=10000, random_state=42)`, so a reordered
source file gives a different sample.

`.gitignore` blocks `data/input/*.csv` and `data/output/*` apart from the dummy fixtures. Leave
those rules alone — Stage 1 writes real consensus scores into `data/output/`.

## 3. Run order

All commands from the repository root.

| Step | Command | Time |
|---|---|---|
| Stage 1 — annotation analyses | `conda run -n fsscore-env python scripts/analysis/run_full_annotator_analysis.py` | ~2 min |
| MOSES / EM comparison | `conda run -n fsscore-env python scripts/analysis/compare_moses_em_distributions.py` | ~30 min |
| Misclassification rates | `conda run -n fsscore-env python scripts/analysis/calculate_misclassification.py` | ~2 min |
| Structural failure analysis | `conda run -n fsscore-env python scripts/analysis/structural_failure_analysis.py` | ~3 min |
| Stage 2 — sweep + stacking | `./run_training.sh` | **~7 h CPU** |
| Fig 7 interpretability | `conda run -n fsscore-env python scripts/analysis/interpretability_analysis.py` | ~2 min |
| SI Figs S3a–d (SHAP) | `conda run -n fsscore-env python scripts/analysis/shap_analysis.py` | ~2 min |

Stage 2 must run with `REG_CV_MODE=nl`, which `run_training.sh` sets and which is now also the
module default. `rowwise` is the legacy leaky path, kept only for comparison.

Stage 2 needs `data/input/regression_500.csv` and `data/input/consensus_pairs_208.csv`. **These are
outputs of Stage 1**, not inputs — run Stage 1 first, then copy them across from `data/output/`
(`unique_molecules_regression_full.csv` → smiles + consensus_score;
`senior_consensus_high_quality.csv` → smiles_a, smiles_b, consensus_choice) and point `config.json`
at them.

## 4. Verifying your run

Diff `data/output/` against the reference artifacts. Expect **23 of 26 byte-identical**. Three
differ for known reasons:

| File | Difference | Cause |
|---|---|---|
| `context_effect_analysis.csv` | 2.2e-16 | row ordering / float repr |
| `moses_em_comparison.txt` | 1.6e-5, FSscore row only | GNN nondeterminism |
| `em_functional_group_annotations.csv` | column order + 0.0234 on one molecule | tool run order; same GNN jitter |

The FSscore GNN is nondeterministic on exactly one molecule in 416 and is the only numeric source
of divergence in Stage 1.

**Stage 2 reproduces statistically, not exactly.** Expect stacking R² ≈ 0.686 ± 0.024 against the
published 0.665 ± 0.022, and per-config differences averaging about +0.015. RNG is fully seeded
(`TPESampler(seed=SEED)`, plus `random`/`numpy`/`torch`), so the residual is library-version drift.
Do not add tolerances to force an exact match — a large gap means the pipeline is being invoked
differently, not that the numbers are wrong.

## What changed on this branch

- Added `scripts/analysis/` — eleven scripts. Nine existed only in the private repo; two
  (`interpretability_analysis.py`, `shap_analysis.py`) existed nowhere and were written from the
  Methods description.
- `run_cv_pipeline.py` — `REG_CV_MODE` now defaults to `nl`. It defaulted to the leaky `rowwise`,
  so direct invocation silently produced leaked numbers.
- `cv_finetuning.py` — `run_cv()` hardcoded `build_folds_rowwise`, which allows molecule overlap by
  design and could not be overridden. It now defaults to `build_folds_molecule_heldout`, the
  protocol behind the published Stage-1 results, with a `--fold_mode` flag.
- `compare_moses_em_distributions.py` — `CONDA_PATH` was hardcoded to another machine; now reads
  `CONDA_EXE`.
- `.gitignore` — `data/output/` was unprotected while Stage 1 writes real consensus scores there.
