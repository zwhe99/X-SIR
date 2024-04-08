<div align="center">
  <img src="assert/logo.png" alt="Logo" width="200">
</div>

<div align="center">
  <h2 align="center">💧 X-SIR: A text watermark that survives translation</h2>
  <a href="https://arxiv.org/abs/2402.14007" style="display: inline-block; text-align: center;">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2402.14007-b31b1b.svg?style=flat">
  </a>
  <a href="https://img.shields.io/badge/python-3.10-blue.svg" style="display: inline-block; text-align: center;">
      <img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg">
  </a>
</div>


**Implementaion of our [paper](https://arxiv.org/abs/2402.14007):**

```
Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models
```

🔥 **News**

* **[Apr 8, 2024]**: New repo released!

**Conda environment**

Tested on the following environment, but it should work on other versions.

- python 3.10.10
- pytorch
- `pip3 install -r requirements.txt`

**Overview**

* `src_watermark` implements three text watermarking methods (`x-sir`, `sir` and `kgw`) with a unified interface.
* `attack` contains two watermarking removal methods: *paraphrase* and *translation*
* Scripts:
  * `gen.py`: generate text with watermark
  * `detect.py`: compute z-score for given texts
  * `eval_detection.py`: calculate AUC, TPR, and F1 for watermark detection
  * You can use `--help` to see full usage of these scripts.
* Supported models:
  * `meta-llama/Llama-2-7b-hf`
  * `baichuan-inc/Baichuan2-7B-Base`
  * `baichuan-inc/Baichuan-7B`
  * `mistralai/Mistral-7B-v0.1`
* Supported languages: English (En), German (De), French (Fr), Chinese (Zh), Japanese (Ja)
* You can learn how to extend the model and language in [from-scratch.md](from-scratch.md).



### Usage (No attack)

**Generate text with watermark**

```shell
MODEL_NAME=baichuan-inc/Baichuan-7B
MODEL_ABBR=baichuan-7b
TRANSFORM_MODEL=data/model/transform_model_x-sbert_10K.pth
MAPPING_FILE=data/mapping/xsir/300_mapping_$MODEL_ABBR.json

WATERMARK_METHOD_FLAG="--watermark_method xsir  --transform_model $TRANSFORM_MODEL --embedding_model paraphrase-multilingual-mpnet-base-v2 --mapping_file $MAPPING_FILE"

python3 gen.py \
    --base_model $MODEL_NAME \
    --fp16 \
    --batch_size 32 \
    --input_file data/dataset/mc4/mc4.en.jsonl \
    --output_file gen/$MODEL_ABBR/xsir/mc4.en.mod.jsonl \
    --WATERMARK_METHOD_FLAG
```

**Compute the z-scores**

```shell
# Compute z-score for human-written text
python3 detect.py \
    --base_model $MODEL_NAME \
    --detect_file data/dataset/mc4/mc4.en.jsonl \
    --output_file gen/$MODEL_ABBR/xsir/mc4.en.hum.z_score.jsonl \
    $WATERMARK_METHOD_FLAG

# Compute z-score for watermarked text
python3 detect.py \
    --base_model $MODEL_NAME \
    --detect_file gen/$MODEL_ABBR/xsir/mc4.en.mod.jsonl \
    --output_file gen/$MODEL_ABBR/xsir/mc4.en.mod.z_score.jsonl \
    $WATERMARK_METHOD_FLAG
```

**Evaluation**

```shell
python3 eval_detection.py \
	--hm_zscore gen/$MODEL_ABBR/xsir/mc4.en.hum.z_score.jsonl \
	--wm_zscore gen/$MODEL_ABBR/xsir/mc4.en.mod.z_score.jsonl

AUC: 0.994

TPR@FPR=0.1: 0.994
TPR@FPR=0.01: 0.862

F1@FPR=0.1: 0.955
F1@FPR=0.01: 0.921
```



### Usage (With attack)

Here we test the watermark after translating to other languages (De, Fr, Zh, Ja).

##### Preparation

We use ChatGPT to perform paraphrase and translation. Therefore:

* Set you openai api key: `export OPENAI_API_KEY=xxxx`
* You may also want to modify the RPMs and TPMs in `attach/const.py`

**Translation**

```shell
TGT_LANGS=("de" "fr" "zh" "ja")
for TGT_LANG in "${TGT_LANGS[@]}"; do
    python3 attack/translate.py \
        --input_file gen/$MODEL_ABBR/xsir/mc4.en.mod.jsonl \
        --output_file gen/$MODEL_ABBR/xsir/mc4.en-$TGT_LANG.mod.jsonl \
        --model gpt-3.5-turbo-1106 \
        --src_lang en \
        --tgt_lang $TGT_LANG
done
```

**Compute the z-scores**

```shell
for TGT_LANG in "${TGT_LANGS[@]}"; do
    python3 detect.py \
        --base_model $MODEL_NAME \
        --detect_file gen/$MODEL_ABBR/xsir/mc4.en-$TGT_LANG.mod.jsonl \
        --output_file gen/$MODEL_ABBR/xsir/mc4.en-$TGT_LANG.mod.z_score.jsonl \
        $WATERMARK_METHOD_FLAG
done
```

**Evaluation**

```shell
for TGT_LANG in "${TGT_LANGS[@]}"; do
    echo "En->$TGT_LANG"
    python3 eval_detection.py \
        --hm_zscore gen/$MODEL_ABBR/xsir/mc4.en.hum.z_score.jsonl \
        --wm_zscore gen/$MODEL_ABBR/xsir/mc4.en-$TGT_LANG.mod.z_score.jsonl
done

En->de
AUC: 0.769

TPR@FPR=0.1: 0.318
TPR@FPR=0.01: 0.060

F1@FPR=0.1: 0.450
F1@FPR=0.01: 0.112

En->fr
AUC: 0.810

TPR@FPR=0.1: 0.354
TPR@FPR=0.01: 0.046

F1@FPR=0.1: 0.488
F1@FPR=0.01: 0.087

En->zh
AUC: 0.905

TPR@FPR=0.1: 0.702
TPR@FPR=0.01: 0.182

F1@FPR=0.1: 0.781
F1@FPR=0.01: 0.305

En->ja
AUC: 0.911

TPR@FPR=0.1: 0.696
TPR@FPR=0.01: 0.112

F1@FPR=0.1: 0.775
F1@FPR=0.01: 0.200
```

## Acknowledgement

This work can not be done without the help of the following repos:

- SIR: [https://github.com/THU-BPM/Robust_Watermark](https://github.com/THU-BPM/Robust_Watermark)
- KGW: [https://github.com/jwkirchenbauer/lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)

## Citation

```ruby
@article{he2024can,
  title={Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models},
  author={He, Zhiwei and Zhou, Binglin and Hao, Hongkun and Liu, Aiwei and Wang, Xing and Tu, Zhaopeng and Zhang, Zhuosheng and Wang, Rui},
  journal={arXiv preprint arXiv:2402.14007},
  year={2024}
}
```
