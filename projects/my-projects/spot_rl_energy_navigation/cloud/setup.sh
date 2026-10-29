#!/usr/bin/env bash
# Linux RTX GPU bootstrap. Does not install drivers or start a full training run.
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
RUNTIME="${SPOT_RUNTIME_DIR:-$PROJECT/.spot-runtime}"
SKIP_APT=0
SKIP_PROBE=0
CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-apt) SKIP_APT=1 ;;
        --skip-probe) SKIP_PROBE=1 ;;
        --check-only) CHECK_ONLY=1 ;;
        --runtime-dir)
            [[ $# -ge 2 ]] || { echo 'Missing --runtime-dir value.' >&2; exit 2; }
            RUNTIME="$2"; shift ;;
        -h|--help)
            cat <<'HELP'
Usage: bash spot_rl_energy_navigation/cloud/setup.sh [options]
  --runtime-dir PATH  Installation directory (default: PROJECT/.spot-runtime)
  --skip-apt          System libraries are already installed; do not use apt/sudo
  --skip-probe        Install and check CUDA/DINO, but skip the simulator probe
  --check-only        Check package/input files without downloads or installation

Requires Ubuntu 22.04/24.04 x86_64 and a graphics-capable NVIDIA container/host.
Use an RTX GPU (e.g. L40S/RTX 6000 Ada) with a compatible host driver.
The portable package contains the project's exact local weights and artifacts.
HELP
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
command -v python3 >/dev/null || { echo 'python3 is required (choose an Ubuntu/PyTorch template).' >&2; exit 1; }
if [[ "$CHECK_ONLY" == 1 ]]; then
    exec python3 "$HERE/verify.py" --check-bundle
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo 'This setup supports Linux x86_64 only.' >&2; exit 1;
}
if [[ "$(uname -r)" == *[Mm]icrosoft* ]]; then
    echo 'Use a native Linux GPU instance; WSL is outside this setup profile.' >&2; exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
[[ "$ID" == ubuntu && ( "$VERSION_ID" == 22.04 || "$VERSION_ID" == 24.04 ) ]] || {
    echo 'Choose an Ubuntu 22.04 or 24.04 instance/template.' >&2; exit 1;
}
command -v nvidia-smi >/dev/null || { echo 'GPU/driver not exposed: nvidia-smi is missing.' >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Verify exact project inputs BEFORE downloading tens of GB of simulator packages.
python3 "$HERE/verify.py" --check-bundle
mkdir -p "$RUNTIME"
RUNTIME="$(cd "$RUNTIME" && pwd)"
mkdir -p "$RUNTIME/logs"
LOG="$RUNTIME/logs/setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[ERROR] Setup failed at line $LINENO. Fix the reported error and rerun the same command. Log: $LOG" >&2' ERR
rm -f "$RUNTIME/simulator-probe.ok"
echo "[INFO] Installation: $RUNTIME"
export UV_CACHE_DIR="$RUNTIME/cache/uv"
export UV_PYTHON_INSTALL_DIR="$RUNTIME/python"
export PIP_CACHE_DIR="$RUNTIME/cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CONSTRAINT="$HERE/constraints.txt"
export PYTHONNOUSERSITE=1
# Avoid using an unrelated Conda/Python environment supplied by a cloud template.
unset CONDA_PREFIX PYTHONHOME PYTHONPATH || true
if [[ "$SKIP_APT" == 0 ]]; then
    command -v apt-get >/dev/null || { echo 'apt-get missing; use --skip-apt after installing system dependencies.'; exit 1; }
    APT_PREFIX=()
    if [[ "$EUID" != 0 ]]; then
        if ! command -v sudo >/dev/null || ! sudo -n true; then
            echo 'System libraries require root/passwordless sudo; install them manually and use --skip-apt.'; exit 1
        fi
        APT_PREFIX=(sudo -n)
    fi
    ALSA_PACKAGE=libasound2
    if [[ "$VERSION_ID" == 24.04 ]]; then ALSA_PACKAGE=libasound2t64; fi
    "${APT_PREFIX[@]}" apt-get update
    "${APT_PREFIX[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential cmake pkg-config \
        libgl1 libglu1-mesa libegl1 libglib2.0-0 libgomp1 libvulkan1 vulkan-tools \
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 libxcursor1 \
        libsm6 libice6 libxt6 libnss3 "$ALSA_PACKAGE" libatomic1 \
        libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 tmux
fi
command -v curl >/dev/null && command -v git >/dev/null
# Install uv privately; do not change the template's Python or shell profile.
if [[ ! -x "$RUNTIME/bin/uv" ]]; then
    curl --fail --location --retry 3 https://astral.sh/uv/0.9.5/install.sh -o "$RUNTIME/uv-install.sh"
    UV_INSTALL_DIR="$RUNTIME/bin" UV_NO_MODIFY_PATH=1 sh "$RUNTIME/uv-install.sh"
fi
UV="$RUNTIME/bin/uv"
if [[ ! -x "$RUNTIME/venv/bin/python" ]]; then
    "$UV" venv --seed --python 3.11 "$RUNTIME/venv"
fi
export PATH="$RUNTIME/venv/bin:$PATH"
PY="$RUNTIME/venv/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$PY" -m pip install 'pip<26' 'setuptools<81' wheel toml
# Select the CUDA build explicitly BEFORE Sim can pull the default PyPI build.
# Version 2.7.0 alone also matches the cu126 build, which cannot run on RTX 5090.
"$PY" -m pip install 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
    'torchaudio==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
LAB="$RUNTIME/IsaacLab"
if [[ ! -d "$LAB" ]]; then
    git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git "$LAB"
fi
[[ "$(git -C "$LAB" describe --tags --exact-match HEAD)" == v2.3.0 ]] || {
    echo 'Existing IsaacLab checkout is not v2.3.0; use a fresh --runtime-dir.'; exit 1;
}
# The default Kit experience also loads tasks; the RL base package supplies
# h5py/hydra and other dependencies imported transitively by the core framework.
# No optional learning-framework extras are installed for this project's custom PPO.
# Build dependencies must be present because Isaac Lab uses setup.py + toml.
"$PY" -m pip install --no-build-isolation \
    'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com \
    -e "$LAB/source/isaaclab" -e "$LAB/source/isaaclab_assets" \
    -e "$LAB/source/isaaclab_rl" -e "$LAB/source/isaaclab_tasks" \
    -e "$LAB/source/isaaclab_mimic" \
    -r "$HERE/requirements.txt"
"$PY" -m pip check
cat > "$RUNTIME/isaaclab-python.sh" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
RUNTIME="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$RUNTIME/venv/bin:$PATH"
export PYTHONNOUSERSITE=1
unset CONDA_PREFIX PYTHONHOME PYTHONPATH || true
exec bash "$RUNTIME/IsaacLab/isaaclab.sh" "$@"
WRAPPER
chmod +x "$RUNTIME/isaaclab-python.sh"
{
    echo '# Generated by cloud/setup.sh; source this file in a new terminal.'
    printf 'export SPOT_RUNTIME_DIR=%q\n' "$RUNTIME"
    printf 'export GENERAL_MAP_ISAACLAB=%q\n' "$RUNTIME/isaaclab-python.sh"
    printf 'export SPOT_ZED_USD=%q\n' "${SPOT_ZED_USD:-$PROJECT/.cloud_assets/ZED_X.usdc}"
    # Keep $PATH literal: expand it when env.sh is sourced in a new terminal.
    # shellcheck disable=SC2016
    printf 'export PATH=%q:"$PATH"\n' "$RUNTIME/venv/bin"
    echo 'export PYTHONNOUSERSITE=1'
    # Vast.ai containers often use root. This flag only affects Kit's root guard.
    if [[ "$EUID" == 0 ]]; then echo 'export OMNI_KIT_ALLOW_ROOT=1'; fi
} > "$HERE/env.sh"
# shellcheck disable=SC1091
source "$HERE/env.sh"
"$PY" "$HERE/verify.py" --runtime
# Record resolved transitive dependencies and source revision for reproducibility.
"$PY" -m pip freeze > "$RUNTIME/requirements-resolved.txt"
git -C "$LAB" rev-parse HEAD > "$RUNTIME/isaaclab-commit.txt"
if [[ "$SKIP_PROBE" == 0 ]]; then
    echo '[INFO] Simulator probe: cameras, physics, DINO; no PPO training updates.'
    echo '[INFO] On first launch, read and accept the NVIDIA license when prompted.'
    echo '[INFO] Extension/shader initialization can take several minutes.'
    bash "$HERE/run.sh" train_general_big_map_test3 --probe-steps 2
    touch "$RUNTIME/simulator-probe.ok"
    echo '[OK] Setup and simulator probe completed.'
else
    rm -f "$RUNTIME/simulator-probe.ok"
    echo '[OK] Installation and CUDA/DINO checks completed. Simulator has NOT been checked.'
fi
echo "[INFO] Setup log: $LOG"
echo "Training: bash $HERE/run.sh train_general_big_map_test3"
