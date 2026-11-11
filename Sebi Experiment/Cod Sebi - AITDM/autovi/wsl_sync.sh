#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Windows + WSL2 workflow: the code is edited in the Windows project folder,
# but it RUNS from a copy in the Linux file system (fast, no spaces in paths,
# as Local/Wsl/README.md asks). This script refreshes that copy.
#
# Run it inside WSL, from wherever the project lives on the Windows disk:
#   bash "/mnt/c/.../Project/AITDM_FLOWER_PROJECT_TUTORIAL/autovi/wsl_sync.sh"
#
# Result (default ~/aitdm, change with AITDM_WORK=...):
#   ~/aitdm/repo                    this repository (.venv, data/, outputs/ are kept)
#   ~/aitdm/autovi_evaluation_code  the official AutoVI evaluation code
#   ~/aitdm/autovi/<category>       links to the dataset folders on the Windows disk
# ---------------------------------------------------------------------------
set -euo pipefail

SRC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$(dirname "$SRC_REPO")"
WORK="${AITDM_WORK:-$HOME/aitdm}"
DATASET_DIR="${AUTOVI_DATASET:-$PROJECT/Dataset(s)}"

mkdir -p "$WORK/repo" "$WORK/autovi"
rsync -a --delete \
    --exclude .git --exclude .venv --exclude /data --exclude /outputs --exclude /logs \
    --exclude __pycache__ --exclude '*.egg-info' \
    "$SRC_REPO/" "$WORK/repo/"

if [ -d "$PROJECT/autovi_evaluation_code" ]; then
    mkdir -p "$WORK/autovi_evaluation_code"
    rsync -a --delete --exclude .git --exclude __pycache__ \
        "$PROJECT/autovi_evaluation_code/" "$WORK/autovi_evaluation_code/"
fi

# One link per category. The zips unpack to <category>/<category>/{train,test,...}
for dir in "$DATASET_DIR"/*/; do
    category="$(basename "$dir")"
    if [ -d "$dir$category/train" ]; then
        ln -sfn "$dir$category" "$WORK/autovi/$category"
    elif [ -d "${dir}train" ]; then
        ln -sfn "${dir%/}" "$WORK/autovi/$category"
    fi
done

echo ">>> Synced to $WORK (categories: $(ls "$WORK/autovi" | tr '\n' ' '))"

# The other direction: results (tables, metrics, models, example figures) back
# to the Windows project folder, next to the repository. Anomaly maps stay in WSL.
if [ -d "$WORK/repo/outputs/autovi" ]; then
    mkdir -p "$PROJECT/results/autovi"
    rsync -a --exclude anomaly_maps "$WORK/repo/outputs/autovi/" "$PROJECT/results/autovi/"
    echo ">>> Results copied to $PROJECT/results/autovi"
fi
# Side experiment (autovi/run_supervised_dirichlet.sh): results and split plots
if compgen -G "$WORK/repo/outputs/experiments/autovi-sup-*" > /dev/null; then
    mkdir -p "$PROJECT/results/autovi_supervised"
    rsync -a "$WORK/repo/outputs/experiments/"autovi-sup-* "$PROJECT/results/autovi_supervised/"
    for dir in "$WORK/repo/data/partitions/"autovi_sup_*/; do
        mkdir -p "$PROJECT/results/autovi_supervised/split_$(basename "$dir")"
        cp "$dir"meta.json "$dir"label_distribution.png \
            "$PROJECT/results/autovi_supervised/split_$(basename "$dir")/" 2>/dev/null || true
    done
    echo ">>> Side-experiment results copied to $PROJECT/results/autovi_supervised"
fi
