import argparse
import json
import math
import os

import shortuuid
import torch
from PIL import Image
from tinyllava.data import *
from tinyllava.model import *
from tinyllava.utils import *
from tqdm import tqdm


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device="cuda")
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    with open(os.path.expanduser(args.question_file), "r") as fp:
        questions = json.loads(fp.read())
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        has_visual = line["visual_input"] == "1"
        image_file = line["filename"]
        qs = line["question"]
        cur_prompt = qs

        if has_visual:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        msg = Message()
        msg.add_message(qs)

        result = text_processor(msg.messages, mode="eval")
        input_ids = result["input_ids"]
        prompt = result["prompt"]
        input_ids = input_ids.unsqueeze(0).cuda()

        if has_visual:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
            image_tensor = image_processor(image)
            image_tensors = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
        else:
            image_tensors = None
            image_sizes = None
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "category": line["category"],
                    "subcategory": line["subcategory"],
                    "visual_input": line["visual_input"],
                    "set_id": line["set_id"],
                    "figure_id": line["figure_id"],
                    "sample_note": line["sample_note"],
                    "question_id": line["question_id"],
                    "question": line["question"],
                    "gt_answer_details": line["gt_answer_details"],
                    "gt_answer": line["gt_answer"],
                    "model_prediction": outputs,
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
