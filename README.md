## Object HalBench

$\text{response-level hallucination rate} = \frac{|\text{responses with object hallucinations}|}{|\text{responses that introduce COCO objects}|}$

$\text{mention-level hallucination rate} = \frac{|\text{falsely mentioned COCO objects}|}{|\text{mentioned COCO objects}|}$

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

## Reference

- Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., ... & Chua, T. S. (2024). Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 13807-13816).

> vim: set spell ts=4 sw=4 et:
