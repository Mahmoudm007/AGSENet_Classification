#!/usr/bin/env bash

#SBATCH --job-name=SALIENCE_training
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --ntasks=1
#SBATCH --time=D-HH:MM:SS
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors_/%x-%j.txt

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash main_job.sh [mode] [options]

Modes:
  pipeline    Run the full project pipeline via run_pipeline.py (default)
  train       Train only
  eval        Evaluate a checkpoint
  visualize  Generate validation/test visualizations
  infer       Run inference on an image or folder
  eda         Run exploratory data analysis only
  diagram     Generate the model architecture diagram only
  plots       Plot training metrics only

Options:
  --config PATH          Config YAML path (default: configs/default.yaml)
  --checkpoint PATH      Checkpoint path (default: outputs/checkpoints/best_model.pth)
  --split NAME           Dataset split for eval/visualize: train, val, or test (default: val)
  --input PATH           Input image/folder for infer mode
  --output-csv PATH      Output CSV for infer mode (default: outputs/inference_results.csv)
  --max-images N         Limit images processed by visualize mode
  --export-gradcam       Export Grad-CAM during eval mode
  --gradcam-target NAME  Grad-CAM target: predicted or true (default: predicted)
  --install-deps         Run pip install -r requirements.txt before the selected mode
  --create-venv          Create and use .venv before running
  --venv PATH            Virtual environment path (default: .venv)
  --help                 Show this message

Examples:
  bash main_job.sh
  bash main_job.sh train --config configs/default.yaml
  bash main_job.sh eval --split val --export-gradcam --gradcam-target true
  bash main_job.sh visualize --split val --max-images 25
  bash main_job.sh infer --input path/to/image.jpg --output-csv outputs/predictions.csv
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

MODE="pipeline"
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  MODE="$1"
  shift
fi

CONFIG="${CONFIG:-configs/default.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/checkpoints/best_model.pth}"
SPLIT="${SPLIT:-val}"
INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/inference_results.csv}"
MAX_IMAGES="${MAX_IMAGES:-}"
EXPORT_GRADCAM="${EXPORT_GRADCAM:-0}"
GRADCAM_TARGET="${GRADCAM_TARGET:-predicted}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
CREATE_VENV="${CREATE_VENV:-0}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
PYTHON_BIN_WAS_SET="0"
if [[ -n "$PYTHON_BIN" ]]; then
  PYTHON_BIN_WAS_SET="1"
fi
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --input)
      INPUT_PATH="$2"
      shift 2
      ;;
    --output-csv|--output_csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    --max-images)
      MAX_IMAGES="$2"
      shift 2
      ;;
    --export-gradcam)
      EXPORT_GRADCAM="1"
      shift
      ;;
    --gradcam-target)
      GRADCAM_TARGET="$2"
      shift 2
      ;;
    --install-deps)
      INSTALL_DEPS="1"
      shift
      ;;
    --create-venv)
      CREATE_VENV="1"
      shift
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ "$CREATE_VENV" == "1" && ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_ACTIVATED="0"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # Linux/macOS virtual environment.
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  VENV_ACTIVATED="1"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Git Bash/MSYS virtual environment on Windows.
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
  VENV_ACTIVATED="1"
fi

if [[ -n "${PYTHON_BIN_OVERRIDE:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN_OVERRIDE"
elif [[ "$VENV_ACTIVATED" == "1" && "$PYTHON_BIN_WAS_SET" == "0" ]]; then
  PYTHON_BIN="python"
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p outputs/logs

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_config() {
  require_file "$CONFIG" "config file"
}

if [[ "$INSTALL_DEPS" == "1" ]]; then
  run "$PYTHON_BIN" -m pip install -r requirements.txt
fi

case "$MODE" in
  pipeline|all)
    require_config
    run "$PYTHON_BIN" run_pipeline.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
    ;;

  train)
    require_config
    run "$PYTHON_BIN" train.py --config "$CONFIG" "${EXTRA_ARGS[@]}"
    ;;

  eval|evaluate)
    require_config
    require_file "$CHECKPOINT" "checkpoint"
    EVAL_ARGS=("$PYTHON_BIN" evaluate.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --split "$SPLIT" --gradcam-target "$GRADCAM_TARGET")
    if [[ "$EXPORT_GRADCAM" == "1" ]]; then
      EVAL_ARGS+=(--export-gradcam)
    fi
    run "${EVAL_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;

  visualize|viz)
    require_config
    VIZ_ARGS=(
      "$PYTHON_BIN" visualize.py
      --config "$CONFIG"
      --split "$SPLIT"
      --best-checkpoint "$CHECKPOINT"
      --gradcam-target "$GRADCAM_TARGET"
      --export-image-panels
      --export-encoder-gradcam
      --export-decoder-gradcam
      --export-roc
      --export-pr
      --export-topk
      --export-calibration
    )
    if [[ -n "$MAX_IMAGES" ]]; then
      VIZ_ARGS+=(--max-images "$MAX_IMAGES")
    fi
    run "${VIZ_ARGS[@]}" "${EXTRA_ARGS[@]}"
    ;;

  infer|predict)
    require_config
    require_file "$CHECKPOINT" "checkpoint"
    if [[ -z "$INPUT_PATH" ]]; then
      echo "infer mode requires --input PATH" >&2
      exit 1
    fi
    run "$PYTHON_BIN" infer.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --input "$INPUT_PATH" --output_csv "$OUTPUT_CSV" "${EXTRA_ARGS[@]}"
    ;;

  eda)
    run "$PYTHON_BIN" eda/run_eda.py "${EXTRA_ARGS[@]}"
    ;;

  diagram|architecture)
    run "$PYTHON_BIN" utils/model_diagram.py "${EXTRA_ARGS[@]}"
    ;;

  plots|plot-metrics)
    run "$PYTHON_BIN" utils/plot_metrics.py --log-file outputs/metrics.jsonl --out-dir outputs/plots "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac

echo
echo "Completed mode: $MODE"
