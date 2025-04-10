#!/bin/bash

set -e

MODEL_PATH="liuhaotian/llava-v1.5-7b"
AMBER_IMAGE_DIR="./AMBER/image"
CONV_MODE="llama"

MODEL_NAME="${MODEL_PATH##*/}"
SAVE_DIR="./result/${MODEL_NAME}/amber"
ANSWERS_FILE="${SAVE_DIR}/amber_answer"

[ -f .env ] && source .env

IFS=',' read -ra GPUS <<<"${CUDA_VISIBLE_DEVICES:-0}"
NUM_CHUNKS="${#GPUS[@]}"

msg() {
  printf "\033[0;32m==> %s\033[0m\n" "$@"
}

[ -d "log/${MODEL_NAME}" ] || mkdir -p "log/${MODEL_NAME}"
LOG_FILE="log/${MODEL_NAME}/amber_$(date +%y%m%d-%H%M%S).log"

msg "START GENERATE ANSWER"

for IDX in $(seq 0 $((NUM_CHUNKS - 1))); do
  CUDA_VISIBLE_DEVICES="${GPUS[$IDX]}" python ./eval_tinyllava/model_amber.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$AMBER_IMAGE_DIR" \
    --question-file "./AMBER/data/query/query_all.json" \
    --answers-file "${ANSWERS_FILE}_${NUM_CHUNKS}_${IDX}.jsonl" \
    --num-chunks "$NUM_CHUNKS" \
    --chunk-idx "$IDX" \
    --conv-mode "$CONV_MODE" \
    --temperature 0 &
done

wait

exec > >(tee "${LOG_FILE}.part") 2>&1

# echo -n >"${ANSWERS_FILE}.jsonl"

# for IDX in $(seq 0 $((NUM_CHUNKS - 1))); do
#   cat "${ANSWERS_FILE}_${NUM_CHUNKS}_${IDX}.jsonl" >>"${ANSWERS_FILE}.jsonl"
# done
python ./eval_tinyllava/merge_answer.py \
  --answer-dir "$SAVE_DIR" \
  --num-chunks "$NUM_CHUNKS"


msg "START EVALUATE"

python ./AMBER/inference.py \
  --word_association "./AMBER/data/relation.json" \
  --safe_words "./AMBER/data/safe_words.txt" \
  --inference_data "${ANSWERS_FILE}.jsonl" \
  --annotation "./AMBER/data/annotations.json" \
  --metrics "./AMBER/data/metrics.txt" \
  --similarity_score 0.8 \
  --evaluation_type "a"

mv "${LOG_FILE}.part" "$LOG_FILE"
