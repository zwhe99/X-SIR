#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import numpy.typing as npt
import torch
from torch import FloatTensor
from torch.nn import functional as F


class RobustLLR_Score_Batch_Base:
    def __init__(self, batch_query):
        self.batch_query = batch_query

    def query_size(self):
        return len(self.batch_query)

    @classmethod
    def from_grid(cls, dist_ps: npt.ArrayLike, dist_qs: npt.ArrayLike):
        dist_ps = np.array(dist_ps)
        dist_qs = np.array(dist_qs)
        assert dist_ps.ndim == 1
        assert dist_qs.ndim == 1
        assert np.all(dist_ps >= 0) and np.all(dist_qs >= 0)
        with np.errstate(divide="ignore"):
            dist_p_logs = np.log(dist_ps)
            dist_q_logs = np.log(dist_qs)
        dist_p_logs.sort()
        dist_q_logs.sort()
        batch_query = [(d_p_l, d_q_l) for d_p_l in dist_p_logs for d_q_l in dist_q_logs]
        return cls(batch_query)


class RobustLLR_Score_Batch_v1(RobustLLR_Score_Batch_Base):
    @torch.no_grad()
    def score(self, p_logits: FloatTensor, q_logits: FloatTensor) -> FloatTensor:
        """dim -2 in result is for different queries, dim -1 is for different tokens"""
        q_logits = F.log_softmax(q_logits, dim=-1).unsqueeze(-2)
        p_logits = F.log_softmax(p_logits, dim=-1).unsqueeze(-2)
        dist_p_logs = torch.tensor(
            [dist_p_log for dist_p_log, dist_q_log in self.batch_query],
            device=p_logits.device,
            dtype=p_logits.dtype,
        ).unsqueeze(-1)
        dist_q_logs = torch.tensor(
            [dist_q_log for dist_p_log, dist_q_log in self.batch_query],
            device=p_logits.device,
            dtype=p_logits.dtype,
        ).unsqueeze(-1)

        max_llr = get_max_llr_v1(p_logits, q_logits, dist_p_logs, dist_q_logs)
        min_llr = -get_max_llr_v1(q_logits, p_logits, dist_q_logs, dist_p_logs)
        trivial_pos = max_llr < min_llr

        llr = safe_minus(q_logits, p_logits)
        r_llr = torch.where(
            trivial_pos,
            torch.tensor(0.0, device=p_logits.device),
            torch.clamp(llr, min_llr, max_llr),
        )
        return r_llr


@torch.no_grad()
def safe_minus(q_logits, p_logits):
    # they can be -inf, but not inf and nan
    #  return torch.where(
    #      torch.isneginf(q_logits),
    #      q_logits,
    #      q_logits - p_logits,
    #  )
    #  use nan_to_num_ instead of where to save memory
    #  return torch.nan_to_num(q_logits - p_logits, nan=float("-inf"))
    llr = q_logits - p_logits
    llr.nan_to_num_(nan=0.0)
    return llr


@torch.no_grad()
def get_max_llr_v1(
    # shape = (..., 1, vocab_size)
    p_logits: FloatTensor,
    q_logits: FloatTensor,
    # shape = (query_size, 1)
    dist_p_logs: FloatTensor,
    dist_q_logs: FloatTensor,
):
    """require large memeory, but avoid loop in python"""

    # shape = (..., 1, vocab_size)
    llr = safe_minus(q_logits, p_logits)
    # shape = (..., 1, vocab_size)
    sort_index = torch.argsort(llr, dim=-1, descending=True)
    del llr

    p_logits = p_logits.gather(-1, sort_index)
    q_logits = q_logits.gather(-1, sort_index)
    del sort_index

    # shape = (..., 1, vocab_size)
    llr = safe_minus(q_logits, p_logits)

    # shape = (..., 1, vocab_size)
    sum_q_logits = torch.logcumsumexp(q_logits, dim=-1)
    sum_p_logits = torch.logcumsumexp(p_logits, dim=-1)
    del q_logits
    del p_logits

    # shape = (..., query_size, vocab_size)
    modified_q_logits = torch.where(
        sum_q_logits <= dist_q_logs,
        torch.tensor(
            float("-inf"), device=sum_q_logits.device, dtype=sum_q_logits.dtype
        ),
        sum_q_logits + torch.log(-torch.expm1(dist_q_logs - sum_q_logits)),
    )
    del sum_q_logits
    modified_p_logits = torch.logaddexp(sum_p_logits, dist_p_logs)
    del sum_p_logits

    # shape = (..., query_size, vocab_size)
    modified_llr = safe_minus(modified_q_logits, modified_p_logits)
    del modified_p_logits
    del modified_q_logits

    # pad left modified_llr with -inf
    # modified_llr : [..., query_size, vocab_size+1]
    modified_llr = F.pad(modified_llr, (1, 0), value=float("-inf"))
    #  get index by argmax
    # cut_index : [..., query_size]
    cut_index = torch.where(
        torch.any(llr < modified_llr[..., :-1], dim=-1),
        torch.argmax((llr < modified_llr[..., :-1]).to(torch.int), dim=-1),
        torch.tensor(modified_llr.shape[-1] - 1, device=modified_llr.device),
    )
    # shape = (..., query_size, 1)
    return modified_llr.gather(-1, cut_index.unsqueeze(-1))


class RobustLLR_Score_Batch_v2(RobustLLR_Score_Batch_Base):
    @torch.no_grad()
    def score(
        self, p_logits: FloatTensor, q_logits: FloatTensor
    ) -> (FloatTensor, FloatTensor, FloatTensor):
        """
        return (llr, max_llr, min_llr)
        llr: [batch_size, seq_len, vocab_size]
        max_llr: [batch_size, seq_len, query_size]
        min_llr: [batch_size, seq_len, query_size]
        """
        q_logits = F.log_softmax(q_logits, dim=-1)
        p_logits = F.log_softmax(p_logits, dim=-1)

        max_llr = get_max_llr_v2(p_logits, q_logits, self.batch_query)
        min_llr = -get_max_llr_v2(
            q_logits, p_logits, [(q, p) for p, q in self.batch_query]
        )
        trivial_pos = max_llr < min_llr
        max_llr = torch.where(
            trivial_pos, torch.tensor(0.0, device=max_llr.device), max_llr
        )
        min_llr = torch.where(
            trivial_pos, torch.tensor(0.0, device=min_llr.device), min_llr
        )

        llr = safe_minus(q_logits, p_logits)
        return llr, max_llr, min_llr


@torch.no_grad()
def get_max_llr_v2(
    # shape = (..., vocab_size)
    p_logits: FloatTensor,
    q_logits: FloatTensor,
    batch_query: list[tuple[float, float]],
):
    """require large memeory, but avoid loop in python"""

    # shape = (..., vocab_size)
    llr = safe_minus(q_logits, p_logits)
    # shape = (..., vocab_size)
    try:
        sort_index = torch.argsort(llr, dim=-1, descending=True)
    except torch.cuda.OutOfMemoryError as e:
        #  use cpu instead
        sort_index = torch.argsort(llr.cpu(), dim=-1, descending=True).to(llr.device)
    del llr

    p_logits = p_logits.gather(-1, sort_index)
    q_logits = q_logits.gather(-1, sort_index)
    del sort_index

    # shape = (..., vocab_size)
    llr = safe_minus(q_logits, p_logits)

    # shape = (..., vocab_size)
    sum_q_logits = torch.logcumsumexp(q_logits, dim=-1)
    sum_p_logits = torch.logcumsumexp(p_logits, dim=-1)
    del q_logits
    del p_logits

    max_llrs = []
    for dist_p_log, dist_q_log in batch_query:
        # shape = (..., vocab_size)
        modified_q_logits = torch.where(
            sum_q_logits <= dist_q_log,
            torch.tensor(
                float("-inf"), device=sum_q_logits.device, dtype=sum_q_logits.dtype
            ),
            sum_q_logits + torch.log(-torch.expm1(dist_q_log - sum_q_logits)),
        )
        modified_p_logits = torch.logaddexp(
            sum_p_logits,
            torch.tensor(
                dist_p_log, device=sum_p_logits.device, dtype=sum_p_logits.dtype
            ),
        )

        # shape = (..., vocab_size)
        modified_llr = safe_minus(modified_q_logits, modified_p_logits)
        del modified_p_logits
        del modified_q_logits

        # pad left modified_llr with -inf
        # shape = (..., vocab_size+1)
        modified_llr = F.pad(modified_llr, (1, 0), value=float("-inf"))
        #  get index by argmax
        # shape = (..., )
        cut_index = torch.where(
            torch.any(llr < modified_llr[..., :-1], dim=-1),
            torch.argmax((llr < modified_llr[..., :-1]).to(torch.int), dim=-1),
            torch.tensor(modified_llr.shape[-1] - 1, device=modified_llr.device),
        )
        # shape = (..., 1)
        max_llrs.append(modified_llr.gather(-1, cut_index.unsqueeze(-1)))
    # shape = (..., query_size)
    max_llr = torch.cat(max_llrs, dim=-1)
    del max_llrs
    return max_llr
