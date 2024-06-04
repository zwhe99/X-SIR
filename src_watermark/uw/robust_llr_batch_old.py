import numpy as np
import numpy.typing as npt
import torch
from torch import FloatTensor
from torch.nn import functional as F

from dataclasses import dataclass, field


@dataclass
class GetMaxLLRLayout:
    p_logits: np.ndarray
    q_logits: np.ndarray
    llr: list[tuple[int, float]] = field(init=False)

    def __post_init__(self):
        llr = [
            (i, self.q_logits[i] - self.p_logits[i]) for i in range(len(self.p_logits))
        ]  # log likelihood ratio with index
        llr.sort(key=lambda x: x[1], reverse=True)
        self.llr = llr

    def to_get_min(self):
        return GetMaxLLRLayout(p_logits=self.q_logits, q_logits=self.p_logits)


@dataclass(frozen=True)
class GetMaxLLRQuery:
    dist_p_log: float
    dist_q_log: float

    def can_leverage(self, other: "GetMaxLLRQuery") -> bool:
        """if return true, then cache for query `other` can be used to reduce computation of query `self`"""
        return (
            self.dist_p_log >= other.dist_p_log and self.dist_q_log >= other.dist_q_log
        )

    def to_get_min(self):
        return GetMaxLLRQuery(dist_p_log=self.dist_q_log, dist_q_log=self.dist_p_log)


@dataclass
class GetMaxLLRQueryCache:
    sum_q_logits: float = -np.inf
    sum_p_logits: float = -np.inf
    i: int = 0  # index of first point that is not in max_set, also length of max_set
    lowest_llr: float = -np.inf


def lowest_llr(query: GetMaxLLRQuery, query_cache: GetMaxLLRQueryCache) -> float:
    if query_cache.sum_q_logits <= query.dist_q_log:
        return -np.inf
    s = query.dist_q_log - query_cache.sum_q_logits
    modified_q_logits = query_cache.sum_q_logits + np.log(-np.expm1(s))
    modified_p_logits = np.logaddexp(query_cache.sum_p_logits, query.dist_p_log)
    return modified_q_logits - modified_p_logits


import copy


def get_max_llr_core(
    layout: GetMaxLLRLayout,
    query: GetMaxLLRQuery,
    initial_query_cache: GetMaxLLRQueryCache,
) -> GetMaxLLRQueryCache:
    query_cache = copy.deepcopy(initial_query_cache)
    query_cache.lowest_llr = lowest_llr(query, query_cache)

    while query_cache.i < len(layout.llr):
        if layout.llr[query_cache.i][1] < query_cache.lowest_llr:
            break
        query_cache.sum_q_logits = np.logaddexp(
            query_cache.sum_q_logits, layout.q_logits[layout.llr[query_cache.i][0]]
        )
        query_cache.sum_p_logits = np.logaddexp(
            query_cache.sum_p_logits, layout.p_logits[layout.llr[query_cache.i][0]]
        )
        query_cache.i += 1
        query_cache.lowest_llr = lowest_llr(query, query_cache)
    return query_cache


@dataclass
class GetMaxLLRBatchQuery:
    query_list: list[GetMaxLLRQuery]
    candidates: dict[GetMaxLLRQuery, set[GetMaxLLRQuery]]

    def to_get_min(self):
        return GetMaxLLRBatchQuery(
            query_list=[q.to_get_min() for q in self.query_list],
            candidates={
                q.to_get_min(): {c.to_get_min() for c in self.candidates[q]}
                for q in self.query_list
            },
        )


@dataclass
class GetMaxLLRCache:
    layout: GetMaxLLRLayout
    query_cache: dict[GetMaxLLRQuery, GetMaxLLRQueryCache]


def get_max_llr_batch(
    layout: GetMaxLLRLayout,
    batch_query: GetMaxLLRBatchQuery,
) -> GetMaxLLRCache:
    cache = GetMaxLLRCache(layout, dict())
    for query in batch_query.query_list:
        candidates = batch_query.candidates[query]
        candidates = [c for c in candidates if query.can_leverage(c)]
        candidate_caches = [
            cache.query_cache[c] for c in candidates if c in cache.query_cache
        ]
        best_cache = max(
            candidate_caches, key=lambda c: c.i, default=GetMaxLLRQueryCache()
        )
        cache.query_cache[query] = get_max_llr_core(layout, query, best_cache)
    return cache


class RobustLLR_Score_Batch_v0:
    def __init__(self, batch_query: GetMaxLLRBatchQuery):
        self.batch_query = batch_query

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
        batch_query = GetMaxLLRBatchQuery(
            query_list=[
                GetMaxLLRQuery(d_p_l, d_q_l)
                for d_p_l in dist_p_logs
                for d_q_l in dist_q_logs
            ],
            candidates={
                GetMaxLLRQuery(dist_p_logs[i_d_p_l], dist_q_logs[i_d_q_l],): {
                    GetMaxLLRQuery(dist_p_logs[i_d_p_l - a], dist_q_logs[i_d_q_l - b])
                    for a, b in [(0, 1), (1, 0)]
                    if i_d_p_l - a >= 0 and i_d_q_l - b >= 0
                }
                for i_d_p_l in range(len(dist_p_logs))
                for i_d_q_l in range(len(dist_q_logs))
            },
        )
        return cls(batch_query)

    def _score(self, p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
        """output shape is (len(self.batch_query.query_list),len(p_logits))"""
        layout = GetMaxLLRLayout(p_logits, q_logits)
        max_llr_cache = get_max_llr_batch(layout, self.batch_query)
        min_llr_cache = get_max_llr_batch(
            layout.to_get_min(), self.batch_query.to_get_min()
        )
        rs = []
        for query, max_llr, min_llr in [
            (
                q,
                max_llr_cache.query_cache[q].lowest_llr,
                -min_llr_cache.query_cache[q.to_get_min()].lowest_llr,
            )
            for q in self.batch_query.query_list
        ]:
            if max_llr <= min_llr:
                rs.append((0, 0))
            else:
                rs.append((max_llr, min_llr))
        return np.array(rs)

    def score(
        self, p_logits: FloatTensor, q_logits: FloatTensor, n_workers=None
    ) -> FloatTensor:
        """dim -2 in result is for different queries, dim -1 is for different tokens"""
        q_logits = F.log_softmax(q_logits, dim=-1)
        p_logits = F.log_softmax(p_logits, dim=-1)
        llr = q_logits - p_logits
        _q_logits = q_logits.detach().cpu().numpy()
        _p_logits = p_logits.detach().cpu().numpy()
        if len(_q_logits.shape) == 1:
            rs = torch.tensor(
                self._score(_p_logits, _q_logits), device=llr.device, dtype=llr.dtype
            )
            llr = torch.clamp(
                llr.unsqueeze(-2), rs[:, 1].unsqueeze(-1), rs[:, 0].unsqueeze(-1)
            )
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
            rs = np.reshape(rs_flat, (*ns, len(self.batch_query.query_list), 2))
            rs = torch.tensor(rs, device=llr.device, dtype=llr.dtype)
            llr = torch.clamp(
                llr.unsqueeze(-2), rs[..., 1].unsqueeze(-1), rs[..., 0].unsqueeze(-1)
            )
            return llr
