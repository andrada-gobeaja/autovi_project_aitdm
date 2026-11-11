#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The whole Stage 1 experiment on AutoVI (one client per category):
#   1. FedAvg            (Flower simulation, ROUNDS x LOCAL_EPOCHS)
#   2. centralised       (same student on the pooled images, same number of epochs)
#   3. local-only        (one student per client, same number of epochs)
# each with every seed, evaluated the same way, then a comparison table.
#
#   bash autovi/run_experiments.sh
#   SEEDS="1" ROUNDS=5 LOCAL_EPOCHS=1 OUT_NAME=smoke bash autovi/run_experiments.sh   # quick test
#
# Finished seeds are skipped, so an interrupted run can simply be restarted.
# Settings (environment variables): SIZE, ROUNDS, LOCAL_EPOCHS, SEEDS, CATEGORIES,
# STRATEGY, EXTRA_CONFIG (more run-config for Flower), OUT_NAME, NO_SPRO=1, CLIENT_GPUS,
# PARTITIONS (an existing partitions folder), BASELINES=0 (federated runs only).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${VENV_DIR:-$REPO_ROOT/.venv}/bin/activate"

SIZE="${SIZE:-384}"
ROUNDS="${ROUNDS:-50}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
SEEDS="${SEEDS:-1 2 3}"
CATEGORIES="${CATEGORIES:-tank_screw pipe_staple underbody_pipes}"
STRATEGY="${STRATEGY:-fedavg}"
EXTRA_CONFIG="${EXTRA_CONFIG:-}"
OUT_NAME="${OUT_NAME:-}"                      # e.g. "smoke" -> outputs/autovi/smoke-fedavg, ...
PREFIX="${OUT_NAME:+$OUT_NAME-}"
DATASET_ROOT="${AUTOVI_DATASET_ROOT:-$REPO_ROOT/../autovi}"
PARTITIONS="${PARTITIONS:-$REPO_ROOT/data/partitions/autovi_$SIZE}"
OUT="$REPO_ROOT/outputs/autovi"
EPOCHS=$((ROUNDS * LOCAL_EPOCHS))
SPRO_FLAG=""
[ -n "${NO_SPRO:-}" ] && SPRO_FLAG="--no-spro"
# One simulated client at a time on the GPU (each one asks Ray for a whole GPU).
# Three at once need more than 8 GB of GPU memory; under WSL that does not fail
# but silently falls back to system memory and becomes ~100x slower. The order
# of execution does not change the result. With a bigger GPU: CLIENT_GPUS=0.33
export CLIENT_GPUS="${CLIENT_GPUS:-1}"

if [ ! -f "$PARTITIONS/meta.json" ]; then
    echo ">>> Preparing the partitions in $PARTITIONS"
    # shellcheck disable=SC2086
    python autovi/prepare_data.py --dataset-root "$DATASET_ROOT" --categories $CATEGORIES \
        --size "$SIZE" --out "$PARTITIONS"
fi

FED_DIR="$OUT/$PREFIX$STRATEGY"
mkdir -p "$FED_DIR"

federated() {  # federated SEED
    local seed="$1" run="$FED_DIR/seed$1" start
    if [ -f "$run/results.json" ]; then
        echo ">>> $(basename "$FED_DIR") seed $seed: done already, skipped"
        return
    fi
    if [ -f "$run/final_model.pt" ] && [ -f "$run/metrics.json" ]; then
        echo ">>> $(basename "$FED_DIR") seed $seed: trained already, evaluating only"
    else
        echo ">>> $(basename "$FED_DIR") seed $seed ($ROUNDS rounds x $LOCAL_EPOCHS local epochs)"
        start=$(date +%s)
        SAVE_DIR="$run" APP_DIR=fl_app_autovi bash scripts/run_simulation.sh "$PARTITIONS" \
            "seed=$seed num-server-rounds=$ROUNDS local-epochs=$LOCAL_EPOCHS strategy='$STRATEGY' $EXTRA_CONFIG" \
            > "$FED_DIR/seed$seed.log" 2>&1 || { echo "Run failed, see $FED_DIR/seed$seed.log" >&2; exit 1; }
        echo "    trained in $(( $(date +%s) - start )) s"
    fi
    # shellcheck disable=SC2086
    python autovi/evaluate.py --run "$run" --partitions "$PARTITIONS" $SPRO_FLAG
}

# Seed by seed: after the first one there is already a complete comparison
# (federated vs. centralised vs. local-only); the other seeds add the std.
for seed in $SEEDS; do
    federated "$seed"
    if [ "${BASELINES:-1}" = "0" ]; then   # federated only (e.g. a second split of the same data)
        python autovi/summarize.py "$FED_DIR"
        continue
    fi
    # shellcheck disable=SC2086
    python autovi/baselines.py --mode centralised --partitions "$PARTITIONS" --epochs "$EPOCHS" \
        --seeds "$seed" --name "${PREFIX}centralised" $SPRO_FLAG
    # shellcheck disable=SC2086
    python autovi/baselines.py --mode local --partitions "$PARTITIONS" --epochs "$EPOCHS" \
        --seeds "$seed" --name "${PREFIX}local" $SPRO_FLAG
    python autovi/summarize.py "$FED_DIR" "$OUT/${PREFIX}centralised" "$OUT/${PREFIX}local"
done
