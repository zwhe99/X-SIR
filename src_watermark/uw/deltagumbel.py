import torch
from torch import FloatTensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


def get_gumbel_variables(rng, vocab_size):
    u = torch.rand((vocab_size,), generator=rng, device=rng.device)  # ~ Unif(0, 1)
    e = -torch.log(u)  # ~ Exp(1)
    g = -torch.log(e)  # ~ Gumbel(0, 1)
    return u, e, g


class DeltaGumbel_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, g: FloatTensor):
        self.g = g

    @classmethod
    def from_random(
        cls,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            g = torch.stack(
                [get_gumbel_variables(rng[i], vocab_size)[2] for i in range(batch_size)]
            )
        else:
            g = get_gumbel_variables(rng, vocab_size)[2]
        return cls(g)


class DeltaGumbel_Reweight(AbstractReweight):
    watermark_code_type = DeltaGumbel_WatermarkCode

    def __repr__(self):
        return f"DeltaGumbel_Reweight()"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        assert isinstance(code, DeltaGumbel_WatermarkCode)
        index = torch.argmax(p_logits + code.g, dim=-1)
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device)
            == index.unsqueeze(-1),
            torch.full_like(p_logits, 0),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits

    def get_la_score(self, code):
        """likelihood agnostic score"""
        import math

        return torch.tensor(math.log(2)) - torch.exp(-code.g)
