#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time setup inside WSL2 (Ubuntu): installs uv if needed (no sudo; it brings
# Python 3.12, because Flower needs 3.11 - 3.13), copies the code to ~/aitdm
# (wsl_sync.sh) and creates the virtual environment with the CUDA build of PyTorch.
#
#   bash "/mnt/c/.../Project/AITDM_FLOWER_PROJECT_TUTORIAL/autovi/wsl_setup.sh"
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${AITDM_WORK:-$HOME/aitdm}"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    echo ">>> Installing uv (Python manager, no sudo)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

bash "$HERE/wsl_sync.sh"

cd "$WORK/repo"
bash scripts/setup_env.sh "$@"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e fl_app_autovi
python - <<'EOF'
import torch, torchvision
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE'}")
# Download the frozen teacher once, so the simulated clients do not all fetch it at once
torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
print("ImageNet ResNet-18 weights cached")
EOF
