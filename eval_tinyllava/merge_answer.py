import os
from argparse import ArgumentParser
from itertools import zip_longest


def merge_gather(*lists, fillvalue=None) -> list[str]:
    return [item for sublist in zip_longest(*lists, fillvalue=fillvalue) for item in sublist if item is not fillvalue]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--answer-dir")
    parser.add_argument("--num-chunks", type=int)
    args = parser.parse_args()

    all_list = []
    for idx in range(args.num_chunks):
        with open(os.path.join(args.answer_dir, f"amber_answer_{args.num_chunks}_{idx}.jsonl")) as fp:
            data = fp.readlines()
            all_list.append(data)

    result = merge_gather(*all_list)
    with open(os.path.join(args.answer_dir, "amber_answer.jsonl")) as fp:
        fp.writelines(result)
