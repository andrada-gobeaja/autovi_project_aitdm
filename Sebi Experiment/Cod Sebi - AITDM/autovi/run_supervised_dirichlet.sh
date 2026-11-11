#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SIDE EXPERIMENT: label-skew (Dirichlet) non-IID split, with the starter kit's
# own splitter and classifier app, on a labelled dataset built from the AutoVI
# TEST folders (see autovi/labelled_from_test.py for why, and for the caveats).
#
#   FedAvg on a Dirichlet(ALPHA) split  vs.  FedAvg on an IID split  vs.  centralised
#
#   bash autovi/run_supervised_dirichlet.sh
#   ALPHA=0.1 SEEDS="1 2 3" bash autovi/run_supervised_dirichlet.sh
#
# The three runs share the same held-out test set (same --seed of split_data.py).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${VENV_DIR:-$REPO_ROOT/.venv}/bin/activate"

SIZE="${SIZE:-224}"
ALPHA="${ALPHA:-0.3}"
ROUNDS="${ROUNDS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
SEEDS="${SEEDS:-1}"
CATEGORIES="${CATEGORIES:-tank_screw pipe_staple underbody_pipes}"
DATASET_ROOT="${AUTOVI_DATASET_ROOT:-$REPO_ROOT/../autovi}"
RAW="$REPO_ROOT/data/raw/autovi_test_labelled_$SIZE.npz"
TAG="autovi-sup"
export CLIENT_GPUS="${CLIENT_GPUS:-1}"   # one simulated client at a time on the GPU

if [ ! -f "$RAW" ]; then
    # shellcheck disable=SC2086
    python autovi/labelled_from_test.py --dataset-root "$DATASET_ROOT" --categories $CATEGORIES \
        --size "$SIZE" --out "$RAW"
fi
CLASS_NAMES=""
for category in $CATEGORIES; do
    CLASS_NAMES+="${CLASS_NAMES:+,}$category/good,$category/defective"
done

for strategy in dirichlet iid; do
    partitions="$REPO_ROOT/data/partitions/${TAG//-/_}_$strategy"
    if [ ! -f "$partitions/meta.json" ]; then
        python data_splitting/split_data.py --input "$RAW" --class-names "$CLASS_NAMES" \
            --num-clients 3 --strategy "$strategy" --alpha "$ALPHA" --out "$partitions"
    fi
    # shellcheck disable=SC2086
    python experiments/run_seeds.py --partitions "$partitions" --seeds $SEEDS \
        --config "num-server-rounds=$ROUNDS local-epochs=$LOCAL_EPOCHS" \
        --name "$TAG-fedavg-$strategy"
done

# shellcheck disable=SC2086
python experiments/centralised.py --partitions "$REPO_ROOT/data/partitions/${TAG//-/_}_dirichlet" \
    --epochs $((ROUNDS * LOCAL_EPOCHS)) --seeds $SEEDS --name "$TAG-centralised"

python experiments/summarize.py outputs/experiments/$TAG-* --with-test \
    | tee "outputs/experiments/$TAG-summary.md"
