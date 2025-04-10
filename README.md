These are some scripts for evaluating hallucination datasets. The datasets support Object HalBench, AMBER, HallusionBench, and the models support LLaVA, TinyLLaVA.

## How to use

1. Install dependences

```bash
pip install jsonlines nltk openai prettytable protobuf spacy
python -c 'import spacy; spacy.cli.download("en_core_web_lg")'
python -c 'import spacy; spacy.cli.download("en_core_web_trf")'
```

2. Create a file named `.env` and change the configuration

```
MODEL_PATH="liuhaotian/llava-v1.5-7b"
OPENAI_KEY="<api_key>"
OPENAI_BASE_URL="https://api.openai.com/v1"
COCO_ANNOTATION_PATH="./coco2014/annotations"
```

2. Start evaluation:

```bash
# for TinyLLaVA:
./script/eval_amber.sh
./script/eval_hb.sh
./script/eval_obj_hal.sh

# for LLaVA:
./script/llava/eval_amber.sh
./script/llava/eval_hb.sh
./script/llava/eval_obj_hal.sh
```

## Object HalBench

$$
\text{response-level hallucination rate} = \frac{|\text{responses with object hallucinations}|}{|\text{responses that introduce COCO objects}|}
$$

$$
\text{mention-level hallucination rate} = \frac{|\text{falsely mentioned COCO objects}|}{|\text{mentioned COCO objects}|}
$$

## AMBER

1. The frequency of hallucinatory objects appearing in the responses:

$$
CHAIR(R) = 1 - \frac{len(R^{\prime}_{obj}\cap A_{obj})}{len(R^{\prime}_{obj})}
$$

$$
\begin{align*}
&A_{obj}\text{ annotated objects list}
\\
&R^{\prime}_{obj}\text{ objects mentioned in the response after filtering}
\end{align*}
$$

2. The object coverage of responses:

$$
Cover(R)=\frac{len(R^{\prime}_{obj} \cap A_{obj})}{len(A_{obj})}
$$

3. The proportion of responses with hallucinations:

$$
Hal(R)=
\begin{cases}
1 \quad if\ CHAIR(R)\ne 0,\\\\
0 \quad otherwise.
\end{cases}
$$

4. Assess whether the hallucinations in MLLMs are similar to those in human cognition:

$$
Cog(R)=\frac{len(R^{\prime}_{obj}\cap H_{obj})}{len(R^{\prime}_{obj})}
$$

## HallusionBench

Visual Dependent questions: questions that do not have an affirmative answer without the visual context.

Visual Supplement questions: questions that can be answered without the visual input; the visual component merely provides supplemental information or corrections.

Metrics: All accuracy, Figure Accuracy, Question Pair Accuracy, Yes Percentage Difference, False Positive Ratio

## Reference

- Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., ... & Chua, T. S. (2024). Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 13807-13816).
- https://github.com/RLHF-V/RLHF-V
- Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., ... & Sang, J. (2023). Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397.
- https://github.com/junyangwang0410/AMBER
- Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., ... & Zhou, T. (2024). Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 14375-14385).
- https://github.com/tianyi-lab/HallusionBench

<!-- vim: set spell ts=4 sw=4 et: -->
