import torch

from typing import Any
from typing import Optional


class RecurrentContextsManager:
    def __init__(self):
        self.contexts = {}

    def init_ctx_with_id(self, cid: str, shape: torch.Size, device: Any, dtype=torch.float32,
                         init_with: Any = 'zeros') -> None:
        if init_with == 'zeros':
            ctx = torch.zeros(shape, dtype=dtype, device=device)
        elif init_with == 'randn':
            ctx = torch.randn(shape, dtype=dtype, device=device)
        elif isinstance(init_with, torch.Tensor):
            ctx = torch.clone(init_with)
        else:
            raise RuntimeError('init_ctx_with_id argument should be either "zeros", "randn" or a tensor.')
        self.contexts[cid] = ctx

    def get_ctx(self, cid: str) -> Optional[torch.Tensor]:
        return self.contexts.get(cid, None)

    def set_ctx(self, cid: str, ctx: torch.Tensor) -> None:
        self.contexts[cid] = ctx

    def del_ctx(self, cid: str) -> None:
        del self.contexts[cid]
