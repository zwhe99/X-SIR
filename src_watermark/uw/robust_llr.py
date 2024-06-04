#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torch import FloatTensor
from torch.nn import functional as F

from . import AbstractScore


def get_max_llr(
    p_logits: np.ndarray,
    q_logits: np.ndarray,
    dist_p_logits: float,
    dist_q_logits: float,
) -> tuple[float, set]:
    llr = [
        (i, q_logits[i] - p_logits[i]) for i in range(len(p_logits))
    ]  # log likelihood ratio with index
    llr.sort(key=lambda x: x[1], reverse=True)
    max_set = set()
    sum_q_logits = -np.inf
    sum_p_logits = -np.inf

    def lowest_llr():
        if sum_q_logits < dist_q_logits:
            return -np.inf
        modified_q_logits = sum_q_logits + np.log(
            1 - np.exp(dist_q_logits - sum_q_logits)
        )
        modified_p_logits = np.logaddexp(sum_p_logits, dist_p_logits)
        return modified_q_logits - modified_p_logits

    for i in range(len(p_logits)):
        if max_set:
            if llr[i][1] < lowest_llr():
                break
        max_set.add(llr[i][0])
        sum_q_logits = np.logaddexp(sum_q_logits, q_logits[llr[i][0]])
        sum_p_logits = np.logaddexp(sum_p_logits, p_logits[llr[i][0]])

    return lowest_llr(), max_set


def get_min_llr(
    p_logits: np.ndarray,
    q_logits: np.ndarray,
    dist_p_logits: float,
    dist_q_logits: float,
) -> tuple[float, set]:
    lowest_neg_llr, min_set = get_max_llr(
        q_logits, p_logits, dist_q_logits, dist_p_logits
    )
    return -lowest_neg_llr, min_set


def safe_ln(x):
    if x <= 0:
        return -np.inf
    return np.log(x)


class RobustLLR_Score(AbstractScore):
    def __init__(self, dist_p: float, dist_q: float):
        assert dist_p >= 0 and dist_q >= 0
        self.dist_p_logits = safe_ln(dist_p)
        self.dist_q_logits = safe_ln(dist_q)

    def _score(self, p_logits: np.ndarray, q_logits: np.ndarray) -> tuple[float, float]:
        max_llr, max_set = get_max_llr(
            p_logits, q_logits, self.dist_p_logits, self.dist_q_logits
        )
        min_llr, min_set = get_min_llr(
            p_logits, q_logits, self.dist_p_logits, self.dist_q_logits
        )
        if max_set.intersection(min_set) or max_llr <= min_llr:
            return (0, 0)
        else:
            return (max_llr, min_llr)

    def score(
        self, p_logits: FloatTensor, q_logits: FloatTensor, n_workers=None
    ) -> FloatTensor:
        q_logits = F.log_softmax(q_logits, dim=-1)
        p_logits = F.log_softmax(p_logits, dim=-1)
        llr = q_logits - p_logits
        _q_logits = q_logits.detach().cpu().numpy()
        _p_logits = p_logits.detach().cpu().numpy()
        if len(_q_logits.shape) == 1:
            max_llr, min_llr = self._score(_p_logits, _q_logits)
            llr = torch.clamp(llr, min_llr, max_llr)
            return llr
        else:
            from concurrent.futures import ProcessPoolExecutor

            ns, d = _q_logits.shape[:-1], _q_logits.shape[-1]
            _q_logits_flat = _q_logits.reshape(-1, d)
            _p_logits_flat = _p_logits.reshape(-1, d)
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                rs_flat = list(
                    executor.map(self._score, _p_logits_flat, _q_logits_flat)
                )
            rs = np.reshape(rs_flat, (*ns, 2))
            max_llr = torch.tensor(rs[..., 0], device=llr.device, dtype=llr.dtype)
            min_llr = torch.tensor(rs[..., 1], device=llr.device, dtype=llr.dtype)
            llr = torch.clamp(llr, min_llr.unsqueeze(-1), max_llr.unsqueeze(-1))
            return llr
