import torch

def get_wp(watermark_type, key):
    from . import (
        WatermarkLogitsProcessor,
        Delta_Reweight,
        Gamma_Reweight,
        PrevN_ContextCodeExtractor,
    )

    if watermark_type == "delta":
        rw = Delta_Reweight()
    elif watermark_type == "gamma":
        rw = Gamma_Reweight()
    else:
        raise ValueError(f"Unknown watermark type: {watermark_type}")
    wp = WatermarkLogitsProcessor(key, rw, PrevN_ContextCodeExtractor(5))
    return wp


def r_llr_score(Model, Tokenizer, texts, dist_qs, watermark_type, key, **kwargs):
    from . import RobustLLR_Score_Batch_v2

    score = RobustLLR_Score_Batch_v2.from_grid([0.0], dist_qs)

    wp = get_wp(watermark_type, key)
    wp.ignore_history = True

    # cache = load_model(model_str)

    #inputs = cache["tokenizer"](texts, return_tensors="pt", padding=True)
    inputs = Tokenizer(texts, return_tensors="pt", padding=True)

    from transformers import GenerationConfig

    #model = cache["generator"].model
    model = Model
    input_ids = inputs["input_ids"][..., :-1].to(model.device)
    attention_mask = inputs["attention_mask"][..., :-1].to(model.device)
    labels = inputs["input_ids"][..., 1:].to(model.device)
    labels_mask = inputs["attention_mask"][..., 1:].to(model.device)
    generation_config = GenerationConfig.from_model_config(model.config)
    logits_processor = model._get_logits_processor(
        generation_config,
        input_ids_seq_length=input_ids.shape[-1],
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=[],
    )
    logits_warper = model._get_logits_warper(generation_config)

    # print("input_ids: ", input_ids)
    # print("attention_mask: ", attention_mask)
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask, 
                    use_cache=False,
                    #do_sample=True,
                    #max_new_tokens=512,
                    #eos_token_id=Tokenizer.eos_token_id,
                    #no_repeat_ngram_size=5,
                    #temperature=0.7
                    )
    logits = outputs.logits
    old_logits = torch.clone(logits)
    new_logits = torch.clone(logits)
    for i in range(logits.size(1)):
        pre = input_ids[:, : i + 1]
        t = logits[:, i]
        t = logits_processor(pre, t)
        t = logits_warper(pre, t)
        old_logits[:, i] = t
        new_logits[:, i] = wp(pre, t)
    llr, max_llr, min_llr = score.score(old_logits, new_logits)
    query_ids = labels
    unclipped_scores = torch.gather(llr, -1, query_ids.unsqueeze(-1)).squeeze(-1)
    # scores : [batch_size, input_ids_len, query_size]
    scores = torch.clamp(unclipped_scores.unsqueeze(-1), min_llr, max_llr)
    return labels, labels_mask, scores * labels_mask.unsqueeze(-1)


def show_r_llr_score(
    Model,
    Tokenizer,
    text,
    compute_range=(None, None),
    merge_till_displayable=True,
    **kwargs,
):
    n = 10
    dist_qs = [float(i) / n for i in range(n + 1)]

    #labels, _, scores = r_llr_score(model_str, [text], dist_qs=dist_qs, **kwargs)
    labels, _, scores = r_llr_score(Model, Tokenizer, [text], dist_qs=dist_qs, **kwargs)
    import numpy as np

    labels = np.array(labels[0].cpu())
    #print(type(scores[0]))
    scores = np.array(scores[0].to(torch.float).cpu())
    if compute_range[0] is None:
        compute_range = (0, compute_range[1])
    if compute_range[1] is None:
        compute_range = (compute_range[0], len(labels))
    scores[: compute_range[0], :] = 0
    scores[compute_range[1] :, :] = 0
    sum_scores = np.sum(scores, axis=0)
    best_index = np.argmax(sum_scores)
    res = sum_scores[best_index]

    if res >= 1000:
        res = 1000

    return float(res)

class Detector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = tokenizer.eos_token

    def detect(self, text):
        return {
            "z_score": show_r_llr_score(
                Model=self.model,
                Tokenizer=self.tokenizer,
                text=text,
                show_latex=False,
                watermark_type="delta",
                key=b"42"
            )
        }