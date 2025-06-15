#!/bin/bash

set -e

trap 'kill -- -$$' EXIT

MODEL_PATH="liuhaotian/llava-v1.5-7b"
OPENAI_KEY="foo"
OPENAI_BASE_URL="https://api.openai.com/v1"
COCO_ANNOTATION_PATH="./coco2014/annotations"
CONV_MODE="llava_v1"

[ -f .env ] && source .env

MODEL_NAME="${MODEL_PATH##*/}"
QUESTION_FILE="./RLHF-V/eval/data/obj_halbench_300_with_image.jsonl"
SAVE_DIR="./result/${MODEL_NAME}/obj_hal"
ANSWERS_FILE="${SAVE_DIR}/obj_halbench_answer.jsonl"

msg() {
  printf "\033[0;32m==> %s\033[0m\n" "$@"
}

[ -d "log/${MODEL_NAME}" ] || mkdir -p "log/${MODEL_NAME}"
LOG_FILE="log/${MODEL_NAME}/obj_hal_$(date +%y%m%d-%H%M%S).log"
exec > >(tee "${LOG_FILE}.part") 2>&1

msg "START GENERATE ANSWER"

python -u ./eval_llava/model_obj_hal.py \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --answers-file "$ANSWERS_FILE" \
  --conv-mode "$CONV_MODE" \
  --temperature 0

msg "START EVALUATE"

python -u ./RLHF-V/eval/eval_gpt_obj_halbench.py \
  --coco_path "$COCO_ANNOTATION_PATH" \
  --cap_folder "$SAVE_DIR" \
  --cap_type "$ANSWERS_FILE" \
  --org_folder "$QUESTION_FILE" \
  --openai_key "$OPENAI_KEY" \
  --openai_baseurl "$OPENAI_BASE_URL" --use_gpt

msg "SHOW RESULT"

python -u ./RLHF-V/eval/summarize_gpt_obj_halbench_review.py "$SAVE_DIR" >"$SAVE_DIR/obj_halbench_scores.txt"

echo 'Scores are:'
cat "$SAVE_DIR/obj_halbench_scores.txt"

mv "${LOG_FILE}.part" "$LOG_FILE"
