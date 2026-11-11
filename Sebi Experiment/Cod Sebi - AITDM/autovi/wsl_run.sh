#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Windows + WSL2: copy the current code to ~/aitdm, run the whole experiment
# there, copy the results back to <Project>/results/autovi. From PowerShell:
#
#   wsl -d Ubuntu -- bash "/mnt/c/.../Project/AITDM_FLOWER_PROJECT_TUTORIAL/autovi/wsl_run.sh"
#
# Settings are passed on to run_experiments.sh, e.g. (inside WSL):
#   SEEDS="1" ROUNDS=5 LOCAL_EPOCHS=1 OUT_NAME=smoke bash ".../autovi/wsl_run.sh"
# Keep the window open and the laptop awake: closing it stops the run, and
# Windows sleep pauses WSL.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${AITDM_WORK:-$HOME/aitdm}"
export PATH="$HOME/.local/bin:$PATH"

bash "$HERE/wsl_sync.sh"
status=0
mkdir -p "$WORK/repo/outputs/autovi"
(cd "$WORK/repo" && bash autovi/run_experiments.sh 2>&1 | tee "$WORK/repo/outputs/autovi/last_run.log") || status=$?
bash "$HERE/wsl_sync.sh" | tail -1
exit "$status"
