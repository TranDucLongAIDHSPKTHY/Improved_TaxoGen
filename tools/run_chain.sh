#!/usr/bin/env bash
# Every step of one dataset, in order. The original TaxoGen is tuned in the
# background while the other methods are tuned; --human-eval also makes the
# annotation forms.
#   bash tools/run_chain.sh configs/dblp.yaml [--human-eval]
set -euo pipefail
C=$1
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=${PY%/Scripts/python}/bin/python   # Linux/macOS venv layout
export PYTHONHASHSEED=0
NAME=$(basename "$C" .yaml)
mkdir -p logs
step() { echo "== $(date '+%F %T') $*"; "$PY" -m taxorec "$@" --config "$C"; }
# tuned methods other than the original TaxoGen, and whether the dataset has recommendation data
read -r has_rec others < <("$PY" -c "
from taxorec.common import load_config
c = load_config('$C')
print('rec' in c, ' '.join('--method ' + m for m in c['taxonomy']['tuning']['grids'] if m != 'taxogen'))" | tr -d '\r')
[ "$has_rec" = True ] && step prepare-rec
step prepare-corpus
step tune-taxonomy --method taxogen > "logs/${NAME}_tune_taxogen.log" 2>&1 &
orig=$!
trap 'kill "$orig" 2>/dev/null || true' EXIT   # a failed step stops the background tuning too
# shellcheck disable=SC2086
step tune-taxonomy $others
step sensitivity
wait $orig
trap - EXIT
step build-taxonomy
step build-taxonomy --all-seeds
step ablation
step eval-taxonomy
[ "${2:-}" = --human-eval ] && step make-human-eval
[ "$has_rec" = True ] && step run-rec
step report
echo "== $(date '+%F %T') chain done"
