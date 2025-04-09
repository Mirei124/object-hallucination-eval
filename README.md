## Object HalBench

$$
\text{response-level hallucination rate} = \frac{|\text{responses with object hallucinations}|}{|\text{responses that introduce COCO objects}|}
$$

$$
\text{mention-level hallucination rate} = \frac{|\text{falsely mentioned COCO objects}|}{|\text{mentioned COCO objects}|}
$$

### Evaluation

Download en_core_web_trf:

```python
import spacy
spacy.cli.download('en_core_web_trf')
```

Run evaluation:

```bash
bash ./script/eval_obj_hal.sh
```

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

### Evaluation

```bash
bash ./script/eval_amber.sh
```

## Reference

- Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., ... & Chua, T. S. (2024). Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 13807-13816).
- https://github.com/RLHF-V/RLHF-V
- Wang, J., Wang, Y., Xu, G., Zhang, J., Gu, Y., Jia, H., ... & Sang, J. (2023). Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397.
- https://github.com/junyangwang0410/AMBER

<!-- vim: set spell ts=4 sw=4 et: -->
