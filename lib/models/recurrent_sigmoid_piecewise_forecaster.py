from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

from torch import nn
import torch

from typing import Optional
from typing import Tuple


class RSPForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, dtype=torch.float32, *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        inp_size = inp_len + hidden_size + 1  # + 1 comes from previous error
        self.ctx_shape = torch.Size([hidden_size])
        self.sigm_lin = nn.Linear(inp_size, hidden_size, dtype=dtype)
        self.w_ctx_cand_lin = nn.Linear(inp_size, hidden_size, dtype=dtype)
        self.out_linear = nn.Linear(hidden_size, 1, dtype=dtype)

    def get_ctx_shape(self) -> Optional[torch.Size]:
        return self.ctx_shape

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 prev_errors: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        inp = torch.cat((inp, ctx, prev_errors), dim=1)
        z = torch.sigmoid(self.sigm_lin(inp))
        ctx_cand = self.w_ctx_cand_lin(inp)
        new_ctx = (1 - z) * ctx + z * ctx_cand
        return self.out_linear(new_ctx), new_ctx
