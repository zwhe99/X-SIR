import torch
from torch import FloatTensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class Delta_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, u: FloatTensor):
        assert torch.all(u >= 0) and torch.all(u <= 1)
        self.u = u

    @classmethod
    def from_random(
        cls,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            u = torch.stack(
                [
                    torch.rand((), generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            u = torch.rand((), generator=rng, device=rng.device)
        return cls(u)


class Delta_Reweight(AbstractReweight):
    watermark_code_type = Delta_WatermarkCode

    def __repr__(self):
        return f"Delta_Reweight()"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        cumsum = torch.cumsum(F.softmax(p_logits, dim=-1), dim=-1)
        index = torch.searchsorted(cumsum, code.u[..., None], right=True)
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device) == index,
            torch.full_like(p_logits, 0),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits
