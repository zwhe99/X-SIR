#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class Gamma_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: LongTensor):
        self.shuffle = shuffle
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return cls(shuffle)


class Gamma_Reweight(AbstractReweight):
    watermark_code_type = Gamma_WatermarkCode

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __repr__(self):
        if self.gamma == 1.0:
            return "Gamma_Reweight()"
        else:
            return f"Gamma_Reweight(gamma={self.gamma})"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        """
        \textbf{$\gamma$-reweight:}
        Let the watermark code space $\mathcal{E}$ be the set of all bijective function between symbol set $\Sigma$ and a set of number $[\abs{\Sigma}]=\{1,\dots,\abs{\Sigma}\}$, where $\abs{\Sigma}$ is the size of symbol set $\Sigma$.
        Essentially, any watermark code $E$ is an indexing function for symbol set $\Sigma$, and also assign an order on $\Sigma$. Let $P_E$ be the uniform probability on $\mathcal{E}$, it would be easy to sample a watermark code $E$ by randomly shuffle the symbol list.

        Assume the original distribution is $P_T(t)\in\Delta_\Sigma,\forall t\in\Sigma$.
        We interpret watermark code $E:\Sigma\to[\abs{\Sigma}]$ as a indexing function and we introduce parameter $\gamma$ to control the strength of watermark.
        % Use the hash of
        % $E$ as a pseudorandom number seed and sample a random permutation $\sigma:\Sigma\to N$.
        Then we construct auxiliary functions
        % $F_I(i)=P_{t\sim P_T}(E(t)\leq i),$
        $F_I(i)=\sum_{t\in\Sigma} \mathbf{1}(E(t)\leq i) P_T(t),$
        $F_S(s)=\begin{cases}(1-\gamma)s & s\leq\frac{1}{2}\\-\gamma+(1+\gamma)s ~~~& s>\frac{1}{2}\end{cases},$
        $F_{I'}(i)=F_S(F_I(i)).$
        The new distribution is given by $P_{T'}(t)=F_{I'}(E(t))-F_{I'}(E(t)-1)$.
        """
        # s_ means shuffled
        s_p_logits = torch.gather(p_logits, -1, code.shuffle)
        s_log_cumsum = torch.logcumsumexp(s_p_logits, dim=-1)
        # normalize the log_cumsum to force the last element to be 0
        s_log_cumsum = s_log_cumsum - s_log_cumsum[..., -1:]
        s_cumsum = torch.exp(s_log_cumsum)
        s_p = F.softmax(s_p_logits, dim=-1)

        #  _t = torch.cat([torch.zeros_like(s_cumsum[..., :1]), s_cumsum], dim=-1)
        boundary = torch.argmax((s_cumsum > 1 / 2).to(torch.int), dim=-1, keepdim=True)
        p_boundary = torch.gather(s_p, -1, boundary)
        portion_in_right = (torch.gather(s_cumsum, -1, boundary) - 1 / 2) / p_boundary
        portion_in_right = torch.clamp(portion_in_right, 0, 1)

        s_all_portion_in_right = (s_cumsum > 1 / 2).type_as(p_logits)
        s_all_portion_in_right.scatter_(-1, boundary, portion_in_right)
        s_shift_logits = torch.log(
            (1 - self.gamma) * (1 - s_all_portion_in_right)
            + (1 + self.gamma) * s_all_portion_in_right
        )
        shift_logits = torch.gather(s_shift_logits, -1, code.unshuffle)
        return p_logits + shift_logits
        #
        #  hi = cumsum
        #  lo = torch.cat([torch.zeros_like(cumsum[..., :1]), cumsum[..., :-1]], dim=-1)
        #
        #  s_p_logits = torch.gather(p_logits, -1, code.shuffle)
        #  cumsum = torch.cumsum(F.softmax(s_p_logits, dim=-1), dim=-1)
        #  portion_in_left = cumsum < 1 / 2
        #  reweighted_cumsum = torch.where(
        #      cumsum < 1 / 2,
        #      (1 - self.gamma) * cumsum,
        #      -self.gamma + (1 + self.gamma) * cumsum,
        #  )
        #  suffled_rewighted_p = torch.diff(
        #      reweighted_cumsum,
        #      dim=-1,
        #      prepend=torch.zeros_like(reweighted_cumsum[..., :1]),
        #  )
        #  rewighted_p = torch.gather(suffled_rewighted_p, -1, code.unshuffle)
        #  reweighted_p_logits = torch.log(rewighted_p)
        #  return reweighted_p_logits
