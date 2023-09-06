# InfoSeek Evaluation Toolkit
- Community Contributed Repository
- [[Project Page]](https://open-vision-language.github.io/infoseek/)[[Paper]](https://arxiv.org/abs/2302.11713)
[[Official Dataset Page]](https://github.com/open-vision-language/infoseek)[[Sibling Project (OVEN)]](https://open-vision-language.github.io/oven/)

<p align="center">
    <img src="assets/infoseek.jpg" width="100%"> <br>
    InfoSeek, A New VQA Benchmark focus on Visual Info-Seeking Questions
</p>

**Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?**

[Yang Chen](https://edchengg.github.io/), [Hexiang Hu](https://www.hexianghu.com/), [Yi Luan](https://luanyi.github.io/), [Haitian Sun](https://scholar.google.com/citations?user=opSHsTQAAAAJ&hl=en), [Soravit Changpinyo](https://schangpi.github.io/), [Alan Ritter](http://aritter.github.io/index.html) and [Ming-Wei Chang](https://mingweichang.org/).

## Release
- [6/7] We are releasing [InfoSeek Dataset](#infoseek-dataset) and evaluation script. 

## InfoSeek Dataset

To download image snapshot, please refer to [OVEN](https://github.com/edchengg/oven_eval).

To download annotations, please run the bash script "download_infoseek_jsonl.sh" inside the infoseek_data folder (from Google Drive).

## Evaluation Script
Run evaluation on InfoSeek validation set:
```python
python run_evaluation.py

# ====Example====
# ===BLIP2 instruct Flan T5 XXL===
# val final score: 8.06
# val unseen question score: 8.89
# val unseen entity score: 7.38
# ===BLIP2 pretrain Flan T5 XXL===
# val final score: 12.51
# val unseen question score: 12.74
# val unseen entity score: 12.28
```

## Starting Code
Run BLIP2 zero-shot inference:
```python
python run_blip2_infoseek.py --split val
```

## Acknowledgement
If you find InfoSeek useful for your your research and applications, please cite using this BibTeX:
```
@article{chen2023infoseek,
  title={Can Pre-trained Vision and Language Models Answer Visual Information-Seeking Questions?},
  author={Chen, Yang and Hu, Hexiang and Luan, Yi and Sun, Haitian and Changpinyo, Soravit and Ritter, Alan and Chang, Ming-Wei},
  journal={arXiv preprint arXiv:2302.11713},
  year={2023}
}
```
