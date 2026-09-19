#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
METHODS="${METHODS:-san ban mutan cross_attention qformer}"
TRANSPORTS="${TRANSPORTS:-none balanced uot}"
SEEDS="${SEEDS:-1105 1106 1107}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/fusion_benchmark/${RUN_ID}}"
TRAIN_CSV="${TRAIN_CSV:-data/gqa_dataset/train.csv}"
DEV_CSV="${DEV_CSV:-data/gqa_dataset/val.csv}"
IMG_PATH="${IMG_PATH:-data/gqa_dataset/images}"
TEXT_MODEL="${TEXT_MODEL:-bert-base-uncased}"
IMAGE_MODEL="${IMAGE_MODEL:-google/vit-base-patch16-224-in21k}"
DIAGNOSTICS="${DIAGNOSTICS:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

case "${DEVICE}" in
  cpu)
    OT_PROFILE="${OT_PROFILE:-configs/ot_cpu.json}"
    ;;
  mps)
    OT_PROFILE="${OT_PROFILE:-configs/ot_mps.json}"
    ;;
  cuda)
    OT_PROFILE="${OT_PROFILE:-configs/ot_gpu.json}"
    ;;
  auto)
    if [[ -z "${OT_PROFILE:-}" ]]; then
      echo "OT_PROFILE must be set when DEVICE=auto" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported DEVICE=${DEVICE}; use cpu, mps, cuda, or auto" >&2
    exit 2
    ;;
esac

fusion_name() {
  local method="$1"
  local transport="$2"
  case "${transport}" in
    none) printf '%s' "${method}" ;;
    uot) printf 'uot_%s' "${method}" ;;
    balanced) printf 'balanced_ot_%s' "${method}" ;;
    *)
      echo "Unsupported transport: ${transport}" >&2
      return 2
      ;;
  esac
}

validate_method() {
  case "$1" in
    san|ban|mutan|cross_attention|qformer) ;;
    *)
      echo "Unsupported method: $1" >&2
      return 2
      ;;
  esac
}

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Run directory already exists; choose a new RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi

TRAIN_HELP="$(${PYTHON_BIN} train.py --help 2>&1)"
missing=()
for method in ${METHODS}; do
  validate_method "${method}"
  for transport in ${TRANSPORTS}; do
    fusion="$(fusion_name "${method}" "${transport}")"
    if [[ "${TRAIN_HELP}" != *"${fusion}"* ]]; then
      missing+=("${fusion}")
    fi
  done
done

if (( ${#missing[@]} )); then
  echo "The following fusion modes are not implemented by train.py:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  echo "Implement the modes in docs/fusion_methods_benchmark_plan.md first." >&2
  exit 2
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Fusion benchmark preflight passed."
  exit 0
fi

mkdir -p "${RUN_ROOT}"
printf '%s\n' \
  "run_id=${RUN_ID}" \
  "device=${DEVICE}" \
  "epochs=${EPOCHS}" \
  "batch_size=${BATCH_SIZE}" \
  "methods=${METHODS}" \
  "transports=${TRANSPORTS}" \
  "seeds=${SEEDS}" \
  "ot_profile=${OT_PROFILE}" \
  "text_model=${TEXT_MODEL}" \
  "image_model=${IMAGE_MODEL}" \
  > "${RUN_ROOT}/benchmark.env"

for method in ${METHODS}; do
  for transport in ${TRANSPORTS}; do
    fusion="$(fusion_name "${method}" "${transport}")"
    for seed in ${SEEDS}; do
      run_dir="${RUN_ROOT}/${method}/${transport}/seed_${seed}"
      model_dir="${run_dir}/model"
      mkdir -p "${run_dir}"

      command=(
        "${PYTHON_BIN}" train.py
        --device "${DEVICE}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --fusion "${fusion}"
        --train_csv_path "${TRAIN_CSV}"
        --dev_csv_path "${DEV_CSV}"
        --img_path "${IMG_PATH}"
        --text_model "${TEXT_MODEL}"
        --image_model "${IMAGE_MODEL}"
        --d_model 384
        --ffn_hidden 1024
        --num_layers 2
        --num_heads 4
        --drop_prob 0.2
        --freeze_answer_embeddings
        --weight_decay 0.05
        --gradient_clip 1.0
        --label_smoothing 0.1
        --early_stopping_patience 8
        --seed "${seed}"
        --model_path "${model_dir}"
      )
      if [[ "${transport}" != "none" ]]; then
        command+=(--ot_profile "${OT_PROFILE}")
      fi
      if [[ "${fusion}" == *"_san" && "${fusion}" != "san" ]]; then
        command+=(
          --ot_san_layers 1
          --ot_san_hidden_dim 128
          --ot_san_dropout 0.2
          --ot_san_gate_init -2.0
        )
      fi
      if [[ "${DIAGNOSTICS}" == "1" ]]; then
        command+=(--diagnostics)
      fi

      printf '%q ' "${command[@]}" > "${run_dir}/command.txt"
      printf '\n' >> "${run_dir}/command.txt"
      echo "Starting method=${method} transport=${transport} seed=${seed}"
      "${command[@]}" 2>&1 | tee "${run_dir}/train.log"
    done
  done
done

"${PYTHON_BIN}" scripts/summarize_fusion_benchmark.py "${RUN_ROOT}"
echo "Benchmark complete: ${RUN_ROOT}"
