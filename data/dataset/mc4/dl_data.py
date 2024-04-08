import json
import jieba
from opencc import OpenCC
from datasets import load_dataset
from tqdm import tqdm

T2S = OpenCC('t2s')
N = 500

for lang in ["en", "zh", "de", "fr", "ja"]:
    bar = tqdm(total=N, desc=f"Processing {lang}")
    ds = load_dataset("mc4", lang, streaming=True, split="validation")
    prompts = []
    responses = []
    for s in ds:
        text = s["text"]
        if lang == "zh":
            text = T2S.convert(text)
        tokens = jieba.cut(text)
        tokens = list(tokens)
        if 195 <= len(tokens) <= 205:
            split_index = int(len(tokens) * 0.1)

            front_tokens = tokens[:split_index]
            back_tokens = tokens[split_index:]

            prompt = "".join(front_tokens)
            response = "".join(back_tokens)

            prompts.append(prompt)
            responses.append(response)
            bar.update(1)
            if len(prompts) == N:
                break

    assert len(prompts) == N
    with open(f"mc4.{lang}.jsonl", "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")