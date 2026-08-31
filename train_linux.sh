#!/usr/bin/env bash
# CROP-AI training / dataset entrypoint (Linux / WSL / macOS).
# Usage:
#   ./train_linux.sh              # env check + dataset prep
#   ./train_linux.sh hardware
#   ./train_linux.sh dataset
#   ./train_linux.sh train        # Phase 2 vision (refuses if CUDA missing unless CROP_AI_ALLOW_CPU_TRAIN=1)
#   ./train_linux.sh train-risk   # Phase 3 LightGBM/heuristic (CPU ok; --dry-run default-safe)
#   ./train_linux.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export CROP_AI_ROOT="${CROP_AI_ROOT:-$ROOT}"

STAGE="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[FAIL] python3 not found. Set PYTHON_BIN or create .venv."
    exit 1
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FAIL] python not found: $PYTHON_BIN"
  exit 1
fi

echo "CROP-AI SIH26131"
echo "root: $ROOT"
echo "stage: $STAGE"
echo "python: $($PYTHON_BIN --version 2>&1)"

fail() { echo "[FAIL] $*"; exit 1; }
ok() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }

run_hardware() {
  "$PYTHON_BIN" "$ROOT/scripts/check_env.py"
}

run_dataset() {
  "$PYTHON_BIN" "$ROOT/scripts/check_env.py" || fail "environment check failed"
  mkdir -p data/raw data/processed data/splits data/synthetic data/manifests \
           runs/detector runs/classifier runs/segmentation runs/risk runs/fusion runs/deployment \
           models logs
  echo "=== dataset preparation ==="
  "$PYTHON_BIN" -m cropai prepare-dataset
  ok "dataset preparation finished (see data/manifests/)"
}

run_train() {
  "$PYTHON_BIN" "$ROOT/scripts/check_env.py" || fail "environment check failed"
  TRAIN_PY="$ROOT/src/cropai/vision/train.py"
  if [[ ! -f "$TRAIN_PY" ]]; then
    warn "Phase 2 vision training is not in this tree yet (src/cropai/vision/train.py missing)."
    exit 1
  fi
  JOINED="${EXTRA_ARGS[*]-}"
  LIGHT=0
  if [[ "$JOINED" == *"--dry-run"* || "$JOINED" == *"--smoke"* ]]; then
    LIGHT=1
  fi
  CUDA="$("$PYTHON_BIN" - <<'PY'
from cropai.utils.hardware import detect_hardware
h = detect_hardware()
print("1" if h.get("cuda_available") else "0")
PY
)"
  if [[ "$CUDA" != "1" ]]; then
    echo "WARNING:"
    echo "CUDA unavailable."
    echo
    echo "Continuing with CPU fallback where practical."
    if [[ "$LIGHT" != "1" && "${CROP_AI_ALLOW_CPU_TRAIN:-0}" != "1" ]]; then
      echo "Refusing heavy training without CUDA. Use --dry-run / --smoke or set CROP_AI_ALLOW_CPU_TRAIN=1."
      exit 2
    fi
  fi
  "$PYTHON_BIN" "$TRAIN_PY" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

run_train_risk() {
  "$PYTHON_BIN" "$ROOT/scripts/check_env.py" || fail "environment check failed"
  RISK_PY="$ROOT/src/cropai/risk/train.py"
  if [[ ! -f "$RISK_PY" ]]; then
    warn "Phase 3 risk training is not in this tree yet."
    exit 1
  fi
  "$PYTHON_BIN" "$RISK_PY" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
}

run_export() {
  "$PYTHON_BIN" "$ROOT/scripts/check_env.py" || fail "environment check failed"
  JOINED="${EXTRA_ARGS[*]-}"
  if [[ "$JOINED" == *"--execute"* ]]; then
    "$PYTHON_BIN" -m cropai export-models --execute
  else
    "$PYTHON_BIN" -m cropai export-models --dry-run
  fi
}

case "$STAGE" in
  hardware) run_hardware ;;
  dataset) run_dataset ;;
  train) run_train ;;
  train-risk) run_train_risk ;;
  export) run_export ;;
  all)
    run_dataset
    run_train
    ;;
  *)
    echo "Unknown stage: $STAGE"
    echo "Usage: $0 [hardware|dataset|train|train-risk|export|all]"
    exit 2
    ;;
esac

ok "stage '$STAGE' complete"
