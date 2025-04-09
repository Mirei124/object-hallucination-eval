#!/bin/bash

set -e

MODEL_PATH="TODO"
OPENAI_KEY="TODO"
COCO_ANNOTATION_PATH="TODO"
CONV_MODE="llama"

MODEL_NAME="${MODEL_PATH##*/}"
QUESTION_FILE="./RLHF-V/eval/data/obj_halbench_300_with_image.jsonl"
SAVE_DIR="./result/${MODEL_NAME}"
ANSWERS_FILE="${SAVE_DIR}/obj_halbench_answer.jsonl"

GREEN='\033[0;32m'
END='\033[0m'

echo -e "\n${GREEN}----> START GENERATE ANSWER <----${END}\n"

python ./eval_tinyllava/model_obj_hal.py \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --answers-file "$ANSWERS_FILE" \
  --conv-mode "$CONV_MODE" \
  --temperature 0

echo -e "\n${GREEN}----> START EVALUATE <----${END}\n"

python ./RLHF-V/eval/eval_gpt_obj_halbench.py \
  --coco_path "$COCO_ANNOTATION_PATH" \
  --cap_folder "$SAVE_DIR" \
  --cap_type "$ANSWERS_FILE" \
  --org_folder "$QUESTION_FILE" \
  --use_gpt \
  --openai_key "$OPENAI_KEY"

echo -e "\n${GREEN}----> SHOW RESULT <----${END}\n"

python ../RLHF-V/eval/summarize_gpt_obj_halbench_review.py "$SAVE_DIR" >"$SAVE_DIR/obj_halbench_scores.txt"

echo 'Scores are:'
cat "$SAVE_DIR/obj_halbench_scores.txt"
