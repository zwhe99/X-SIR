#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch
from torch import FloatTensor, LongTensor
from torch.nn import functional as F


class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random(
        cls,
        #  rng: Union[torch.Generator, list[torch.Generator]],
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        """When rng is a list, it should have the same length as the batch size."""
        pass


class AbstractReweight(ABC):
    watermark_code_type: type[AbstractWatermarkCode]

    @abstractmethod
    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        pass


class AbstractScore(ABC):
    @abstractmethod
    def score(self, p_logits: FloatTensor, q_logits: FloatTensor) -> FloatTensor:
        """p is the original distribution, q is the distribution after reweighting."""
        pass


class LLR_Score(AbstractScore):
    def score(self, p_logits: FloatTensor, q_logits: FloatTensor) -> FloatTensor:
        return F.log_softmax(q_logits, dim=-1) - F.log_softmax(p_logits, dim=-1)


class AbstractContextCodeExtractor(ABC):
    @abstractmethod
    def extract(self, context: LongTensor) -> any:
        """Should return a context code `c` which will be used to initialize a torch.Generator."""
        pass
