#!/bin/bash

set -e

MODEL_PATH="liuhaotian/llava-v1.5-7b"
OPENAI_KEY="foo"
OPENAI_BASE_URL="https://api.openai.com/v1"
CONV_MODE="llama"

MODEL_NAME="${MODEL_PATH##*/}"
SAVE_DIR="./result/${MODEL_NAME}/hb"
ANSWERS_FILE="${SAVE_DIR}/hb_answer"
VD_SAVE_PATH="${SAVE_DIR}/hb_output_vd_model.json"
VS_SAVE_PATH="${SAVE_DIR}/hb_output_vs_model.json"

[ -f .env ] && source .env

IFS=',' read -ra GPUS <<<"${CUDA_VISIBLE_DEVICES:-0}"
NUM_CHUNKS="${#GPUS[@]}"

msg() {
  printf "\033[0;32m==> %s\033[0m\n" "$@"
}

[ -d "log/${MODEL_NAME}" ] || mkdir -p "log/${MODEL_NAME}"
LOG_FILE="log/${MODEL_NAME}/hb_$(date +%y%m%d-%H%M%S).log"

msg "START GENERATE ANSWER"

for IDX in $(seq 0 $((NUM_CHUNKS - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[$IDX]}" python -u ./eval_tinyllava/model_hb.py \
    --model-path "$MODEL_PATH" \
    --image-folder "./HallusionBench/hallusion_bench" \
    --question-file "./HallusionBench/HallusionBench.json" \
    --answers-file "${ANSWERS_FILE}_${NUM_CHUNKS}_${IDX}.jsonl" \
    --num-chunks "$NUM_CHUNKS" \
    --chunk-idx "$IDX" \
    --conv-mode "$CONV_MODE" \
    --temperature 0 &
done

wait

exec > >(tee "${LOG_FILE}.part") 2>&1

echo -n >"${ANSWERS_FILE}.jsonl"

for IDX in $(seq 0 $((NUM_CHUNKS - 1))); do
  cat "${ANSWERS_FILE}_${NUM_CHUNKS}_${IDX}.jsonl" >>"${ANSWERS_FILE}.jsonl"
done

msg "START EVALUATE"

python -u ./eval_tinyllava/eval_hb.py \
  --input_file_name "${ANSWERS_FILE}.jsonl" \
  --api_key "$OPENAI_KEY" \
  --api_base "$OPENAI_BASE_URL" \
  --save_json_path_vd "$VD_SAVE_PATH" \
  --save_json_path_vs "$VS_SAVE_PATH" \
  --load_json

mv "${LOG_FILE}.part" "$LOG_FILE"
