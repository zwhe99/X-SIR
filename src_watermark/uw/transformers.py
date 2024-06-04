#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor

from .base import AbstractReweight, AbstractContextCodeExtractor, AbstractScore


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        private_key: any,
        reweight: AbstractReweight,
        context_code_extractor: AbstractContextCodeExtractor,
        ignore_history=False,
    ):
        self.private_key = private_key
        self.reweight = reweight
        self.context_code_extractor = context_code_extractor
        self.ignore_history = ignore_history
        self.cc_history = set()

    def __repr__(self):
        return f"WatermarkLogitsProcessor({repr(self.private_key)}, {repr(self.reweight)}, {repr(self.context_code_extractor)}, {repr(self.ignore_history)})"

    def get_rng_seed(self, context_code: any) -> any:
        if not self.ignore_history:
            self.cc_history.add(context_code)
        import hashlib

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.private_key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    def reset_history(self):
        self.cc_history = set()

    def _get_codes(self, input_ids: LongTensor):
        batch_size = input_ids.size(0)
        context_codes = [
            self.context_code_extractor.extract(input_ids[i]) for i in range(batch_size)
        ]

        mask, seeds = zip(
            *[
                (context_code in self.cc_history, self.get_rng_seed(context_code))
                for context_code in context_codes
            ]
        )
        return mask, seeds

    def _core(self, input_ids: LongTensor, scores: FloatTensor):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device)
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, scores.size(1)
        )
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scores)
        return mask, reweighted_scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        mask, reweighted_scores = self._core(input_ids, scores)
        if self.ignore_history:
            return reweighted_scores
        else:
            return torch.where(mask[:, None], scores, reweighted_scores)

    def get_score(
        self,
        labels: LongTensor,
        old_logits: FloatTensor,
        new_logits: FloatTensor,
        scorer,
    ) -> FloatTensor:
        from unbiased_watermark import (
            RobustLLR_Score_Batch_v1,
            RobustLLR_Score_Batch_v2,
        )

        if isinstance(scorer, RobustLLR_Score_Batch_v1):
            all_scores = scorer.score(old_logits, new_logits)
            query_ids = labels.unsqueeze(-1).expand(
                tuple(-1 for _ in range(decoder_input_ids.ndim))
                + (all_scores.size(-2),)
            )
            #  scores: [batch_size, query_size]
            scores = torch.gather(all_scores, -1, query_ids.unsqueeze(-1)).squeeze(-1)
        elif isinstance(scorer, RobustLLR_Score_Batch_v2):
            llr, max_llr, min_llr = scorer.score(old_logits, new_logits)
            query_ids = labels
            unclipped_scores = torch.gather(llr, -1, query_ids.unsqueeze(-1)).squeeze(
                -1
            )
            #  scores: [batch_size, query_size]
            scores = torch.clamp(unclipped_scores.unsqueeze(-1), min_llr, max_llr)
        return scores

    def get_la_score(
        self,
        input_ids: LongTensor,
        labels: LongTensor,
        vocab_size: int,
    ) -> FloatTensor:
        assert "get_la_score" in dir(
            self.reweight
        ), "Reweight does not support likelihood agnostic detection"
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)
        all_scores = self.reweight.get_la_score(watermark_code)
        scores = torch.gather(all_scores, -1, labels.unsqueeze(-1)).squeeze(-1)
        scores = torch.logical_not(mask).float() * scores
        return scores


def get_score(
    text: str,
    watermark_processor: WatermarkLogitsProcessor,
    score: AbstractScore,
    model,
    tokenizer,
    temperature=0.2,
    prompt: str = "",
    **kwargs,
) -> tuple[FloatTensor, int]:
    input_ids = tokenizer.encode(text)
    prompt_len = len(tokenizer.encode(prompt))
    input_ids = torch.tensor(input_ids, device=model.device).unsqueeze(0)
    outputs = model(input_ids)
    logits = (
        torch.cat(
            [torch.zeros_like(outputs.logits[:, :1]), outputs.logits[:, :-1]],
            dim=1,
        )
        / temperature
    )
    new_logits = torch.clone(logits)
    for i in range(logits.size(1)):
        if i == prompt_len:
            watermark_processor.reset_history()
        if i == 0:
            watermark_processor.reset_history()
            continue
        new_logits[:, i] = watermark_processor(input_ids[:, :i], logits[:, i])
    all_scores = score.score(logits, new_logits)
    if input_ids.ndim + 2 == all_scores.ndim:
        # score is RobustLLR_Score_Batch
        input_ids = input_ids.unsqueeze(-1).expand(
            tuple(-1 for _ in range(input_ids.ndim)) + (all_scores.size(-2),)
        )
    scores = torch.gather(all_scores, -1, input_ids.unsqueeze(-1)).squeeze(-1)
    return scores[0], prompt_len
