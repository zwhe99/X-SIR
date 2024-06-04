#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from torch import FloatTensor, LongTensor
from .base import AbstractContextCodeExtractor


@dataclass
class All_ContextCodeExtractor(AbstractContextCodeExtractor):
    def extract(self, context: LongTensor) -> any:
        return context.detach().cpu().numpy().tobytes()


@dataclass
class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    """Extracts the last n tokens in the context"""

    n: int

    def extract(self, context: LongTensor) -> any:
        return context[-self.n :].detach().cpu().numpy().tobytes()
