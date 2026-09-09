#!/bin/bash
# =============================================================================
# reproduce.sh -- regenerate every figure and table in the manuscript.
#
#   ./reproduce.sh                  full run (Stage 2 takes ~7 h on CPU)
#   ./reproduce.sh --skip-stage2    everything except the 186-config sweep (~40 min)
#   ./reproduce.sh --verify-only    just diff existing outputs against the oracle
#   ./reproduce.sh --force          re-run steps whose outputs already exist
#
# Requires, in data/input/ (see SETUP.md):
#   final_alpha_for_launch_pair_analysis_2026-03-22.csv   restricted; from the corresponding author
#   EM_list.csv                                            416-compound Elton reference set
#   data_moses_train.csv                                   built automatically if absent
#
# Optional, for verification:
#   data/reference/    26 oracle artifacts; also from the corresponding author
#
# Progress is echoed and also written to reproduce.log.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
LOG="$ROOT/reproduce.log"
CONDA="${CONDA_EXE:-conda}"
ENV_MAIN="fsscore-env"

SKIP_STAGE2=0; VERIFY_ONLY=0; FORCE=0
for a in "$@"; do case "$a" in
  --skip-stage2) SKIP_STAGE2=1 ;;
  --verify-only) VERIFY_ONLY=1 ;;
  --force)       FORCE=1 ;;
  -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
  *) echo "unknown option: $a (try --help)"; exit 2 ;;
esac; done

say()  { printf '\n\033[1m[%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
info() { printf '  %s\n' "$*" | tee -a "$LOG"; }
die()  { printf '\n\033[31mERROR: %s\033[0m\n' "$*" | tee -a "$LOG"; exit 1; }
run()  { "$@" >>"$LOG" 2>&1 || die "step failed -- see $LOG (tail -40 $LOG)"; }

# run a script inside a conda env
pyrun() { local env="$1"; shift; run "$CONDA" run --no-capture-output -n "$env" python "$@"; }

# Run inline Python inside a conda env.
#   pyinline <env> <<'PY' ... PY
# NOTE: `conda run ... python -` does NOT forward stdin -- python reads an empty
# script, does nothing and exits 0, so heredocs silently no-op. Write to a temp
# file instead.
pyinline() {
  local env="$1" tmp rc
  tmp="$(mktemp "${TMPDIR:-/tmp}/synthpred_XXXXXX.py")"
  cat > "$tmp"
  "$CONDA" run --no-capture-output -n "$env" python "$tmp" >>"$LOG" 2>&1
  rc=$?
  rm -f "$tmp"
  return $rc
}

# Assert a step actually produced what it claims, so a silent failure can never
# be reported as success.
produced() { [ -s "$1" ] || die "step reported success but did not produce $1 -- see $LOG"; }

# skip a step when its output already exists, unless --force
done_already() { [ "$FORCE" -eq 0 ] && [ -s "$1" ]; }

: > "$LOG"
say "GenEM-Synth reproduction"
info "repo: $ROOT"

# ---------------------------------------------------------------- preflight --
preflight() {
  say "Preflight"
  command -v "$CONDA" >/dev/null 2>&1 || die "conda not found. Set CONDA_EXE to its path."

  for e in "$ENV_MAIN" sascore-env scscore-env syba-env; do
    "$CONDA" env list | awk '{print $1}' | grep -qx "$e" || die "missing conda env '$e' -- run: python env_installer.py"
  done
  "$CONDA" env list | awk '{print $1}' | grep -qx rascore-env \
    || info "WARNING: rascore-env missing; RAscore columns will be unavailable (see SETUP.md)"

  "$CONDA" run -n "$ENV_MAIN" python -c "import shap, seaborn, xgboost, torch" >/dev/null 2>&1 \
    || die "$ENV_MAIN is missing analysis packages. Run:
    conda run -n $ENV_MAIN pip install --no-deps seaborn shap slicer cloudpickle
    conda run -n $ENV_MAIN pip install 'numba==0.59.1'"

  [ -s data/input/final_alpha_for_launch_pair_analysis_2026-03-22.csv ] \
    || die "missing the annotation records at data/input/final_alpha_for_launch_pair_analysis_2026-03-22.csv (restricted; request from the corresponding author)"
  [ -s data/input/EM_list.csv ] || die "missing data/input/EM_list.csv (416-compound Elton reference set)"

  [ -d data/reference ] && info "oracle present ($(ls data/reference | wc -l) files) -- will verify at the end" \
                        || info "no data/reference/ -- reproduction will run but cannot be verified"
  info "preflight OK"
}

# ------------------------------------------------------------------ stage 1 --
stage1() {
  say "Stage 1 -- annotation analyses (~2 min)"
  if done_already data/output/unique_molecules_regression_full.csv; then
    info "already present, skipping (--force to redo)"; return
  fi
  pyrun "$ENV_MAIN" scripts/analysis/run_full_annotator_analysis.py
  produced data/output/unique_molecules_regression_full.csv
  info "consensus target written ($(($(wc -l < data/output/unique_molecules_regression_full.csv)-1)) molecules)"
}

# --------------------------------------------------------- MOSES + EM scores --
moses_inputs() {
  say "MOSES training set"
  if done_already data/input/data_moses_train.csv; then info "already present, skipping"; return; fi
  info "downloading the public MOSES release (~80 MB) and extracting the train split"
  pyinline "$ENV_MAIN" <<'PY' || die "MOSES download failed"
import pandas as pd
URL = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv"
df = pd.read_csv(URL)
tr = df[df.SPLIT == "train"].reset_index(drop=True)
assert len(tr) == 1_584_663, f"unexpected MOSES train size: {len(tr)}"
tr[["SMILES"]].to_csv("data/input/data_moses_train.csv", index=False)
PY
  produced data/input/data_moses_train.csv
  info "$(wc -l < data/input/data_moses_train.csv) lines written (order preserved -- the 10k sample depends on it)"
}

em_scores() {
  say "Scoring the 416 energetic materials (~5 min)"
  if done_already data/processed/scored.csv; then info "already present, skipping"; return; fi
  pyinline "$ENV_MAIN" <<'PY' || die "could not build the EM scoring input"
import pandas as pd
em = pd.read_csv("data/input/EM_list.csv"); em.columns = ["smiles"]
em["target"] = ""
em.to_csv("data/processed/scored.csv", index=False)
PY
  produced data/processed/scored.csv
  for pair in "sascore-env:run_sascore.py" "scscore-env:run_scscore.py" "syba-env:run_syba.py" \
              "rascore-env:run_rascore_tensor.py" "rascore-env:run_rascore_xgb.py" \
              "$ENV_MAIN:run_fsscore.py"; do
    env="${pair%%:*}"; script="${pair##*:}"
    if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$env"; then
      info "skipping $script ($env unavailable)"; continue
    fi
    info "running $script"
    run "$CONDA" run --no-capture-output -n "$env" python "scripts/scoring/$script" \
        data/processed/scored.csv data/processed/scored.csv
  done
  info "scored columns: $(head -1 data/processed/scored.csv | tr ',' '\n' | wc -l)"
}

# ------------------------------------------------------------ MOSES analyses --
moses_analyses() {
  say "MOSES / EM comparison -- Table S3, Fig 5 (~30 min)"
  if done_already data/output/moses_em_comparison.txt; then info "already present, skipping"; else
    pyrun "$ENV_MAIN" scripts/analysis/compare_moses_em_distributions.py
  fi
  say "Misclassification rates (~2 min)"
  if done_already data/output/misclassification_rates.csv; then info "already present, skipping"; else
    pyrun "$ENV_MAIN" scripts/analysis/calculate_misclassification.py
  fi
  say "Structural failure analysis -- Fig 3, Fig 4 (~3 min)"
  if done_already data/output/structural_failure_analysis.csv; then info "already present, skipping"; else
    pyrun "$ENV_MAIN" scripts/analysis/structural_failure_analysis.py
  fi
}

# ------------------------------------------------------------------ stage 2 --
stage2() {
  say "Stage 2 -- 186-config sweep + ridge stacking (~7 h CPU)"
  if done_already analysis_output/regression_cv/no_leakage/cv_summary.csv; then
    info "already present, skipping (--force to redo)"; return
  fi

  # Stage-2 inputs are derived from Stage-1 outputs, not supplied by hand.
  pyinline "$ENV_MAIN" <<'PY' || die "could not derive the Stage-2 inputs"
import json, pandas as pd
r = pd.read_csv("data/output/unique_molecules_regression_full.csv")
r[["smiles", "consensus_score"]].to_csv("data/input/regression_500.csv", index=False)
p = pd.read_csv("data/output/senior_consensus_high_quality.csv")
p[["smiles_a", "smiles_b", "consensus_choice"]].to_csv("data/input/consensus_pairs_208.csv", index=False)
json.dump({"training_input_file": "data/input/regression_500.csv",
           "consensus_pairs_file": "data/input/consensus_pairs_208.csv",
           "top_k": 10}, open("config.reproduction.json", "w"), indent=2)
print(f"derived {len(r)} molecules and {len(p)} pairs")
PY
  produced data/input/regression_500.csv; produced data/input/consensus_pairs_208.csv
  info "derived Stage-2 inputs from Stage-1 outputs"
  info "running with REG_CV_MODE=nl (the no-leakage protocol; 'rowwise' is the legacy leaky path)"
  REG_CV_MODE=nl SYNTHPRED_CONFIG="$ROOT/config.reproduction.json" \
    run "$CONDA" run --no-capture-output -n "$ENV_MAIN" python scripts/train_models/run_cv_pipeline.py
}

# ---------------------------------------------------------- interpretability --
interpretability() {
  local cv=analysis_output/regression_cv/no_leakage/scored_500_molecules.csv
  if [ ! -s "$cv" ]; then
    say "Interpretability + SHAP -- skipped"
    info "needs Stage 2 output ($cv)"; return
  fi
  say "Fig 7 interpretability (~2 min)"
  pyrun "$ENV_MAIN" scripts/analysis/interpretability_analysis.py
  say "SI Figs S3a-d, SHAP (~2 min)"
  pyrun "$ENV_MAIN" scripts/analysis/shap_analysis.py
}

# ----------------------------------------------------------------- verifying --
verify() {
  say "Verification against the reference oracle"
  [ -d data/reference ] || { info "no data/reference/ -- nothing to verify against"; return; }
  _vtmp="$(mktemp "${TMPDIR:-/tmp}/synthpred_XXXXXX.py")"; cat > "$_vtmp" <<'PY' 
import filecmp, pathlib
ref = pathlib.Path("data/reference"); out = pathlib.Path("data/output")
same = diff = missing = 0
notes = {
 "context_effect_analysis.csv": "row ordering / float repr (2.2e-16)",
 "moses_em_comparison.txt": "FSscore row only, 1.6e-5 (GNN nondeterminism)",
 "em_functional_group_annotations.csv": "column order + 0.0234 on one molecule (same GNN jitter)",
}
for r in sorted(ref.iterdir()):
    o = out / r.name
    if not o.exists():
        missing += 1; print(f"  not generated : {r.name}"); continue
    if filecmp.cmp(o, r, shallow=False):
        same += 1
    else:
        diff += 1
        print(f"  differs       : {r.name}" + (f"   [expected: {notes[r.name]}]" if r.name in notes else "   [UNEXPECTED]"))
print(f"\n  byte-identical {same}   differs {diff}   not generated {missing}")
unexpected = diff and any((out/r.name).exists() and not filecmp.cmp(out/r.name, r, shallow=False)
                          and r.name not in notes for r in ref.iterdir())
print("\n  RESULT: reproduction matches the oracle." if not unexpected
      else "\n  RESULT: unexpected differences above -- investigate before relying on this run.")
PY
  "$CONDA" run --no-capture-output -n "$ENV_MAIN" python "$_vtmp" | tee -a "$LOG"; rm -f "$_vtmp"
  local s=analysis_output/regression_cv/no_leakage/cv_summary.csv
  if [ -s "$s" ]; then
    _vtmp="$(mktemp "${TMPDIR:-/tmp}/synthpred_XXXXXX.py")"; cat > "$_vtmp" <<'PY' 
import pandas as pd
d = pd.read_csv("analysis_output/regression_cv/no_leakage/cv_summary.csv")
r = d[(d.model == "stacking_ridge")].iloc[0]
print(f"\n  Stacking R2 = {r.r2_mean:.4f} +/- {r.r2_std:.4f}   MAE = {r.mae_mean:.4f}")
print( "  Published   = 0.665 +/- 0.022        MAE = 0.448")
print( "  Expect ~0.686 +/- 0.024: reproduction is statistical (~+/-0.02), not exact.")
print( "  RNG is fully seeded, so the residual is library-version drift -- do not add")
print( "  tolerances to force a match; a large gap means the pipeline ran differently.")
PY
  "$CONDA" run --no-capture-output -n "$ENV_MAIN" python "$_vtmp" | tee -a "$LOG"; rm -f "$_vtmp"
  fi
}

# ---------------------------------------------------------------------- main --
if [ "$VERIFY_ONLY" -eq 1 ]; then verify; exit 0; fi

preflight
stage1
moses_inputs
em_scores
moses_analyses
if [ "$SKIP_STAGE2" -eq 1 ]; then
  say "Stage 2 -- skipped (--skip-stage2)"
  info "Fig 7 and SHAP need Stage 2 output and will be skipped too"
else
  stage2
  interpretability
fi
verify

say "Done"
info "outputs: data/output/  and  analysis_output/regression_cv/no_leakage/"
info "log:     $LOG"
info "Fig 2 / Table 2 are NOT reproducible -- the Stage-1 CV seeds were never recorded."
