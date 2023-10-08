from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

from torch import nn
import torch

from typing import Any
from typing import Optional
from typing import Tuple


class RSPForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, use_out_linear=True, dtype=torch.float32,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        inp_size = inp_len + hidden_size
        self.ctx_shape = torch.Size([hidden_size])
        self.sigm_lin = nn.Linear(inp_size, hidden_size, dtype=dtype)
        self.w_ctx_cand_lin = nn.Linear(inp_size, hidden_size, dtype=dtype)
        self.use_out_linear = use_out_linear
        if use_out_linear:
            self.out_linear = nn.Linear(inp_len + hidden_size, 1, dtype=dtype)

    def get_ctx_shape(self) -> Optional[torch.Size]:
        return self.ctx_shape

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        orig_inp = inp
        inp = torch.cat((inp, ctx), dim=-1)
        z = torch.sigmoid(self.sigm_lin(inp))
        ctx_cand = self.w_ctx_cand_lin(inp)
        new_ctx = (1 - z) * ctx + z * ctx_cand
        out = new_ctx
        if self.use_out_linear:
            out = self.out_linear(torch.cat((orig_inp, new_ctx), dim=-1))
        return out, new_ctx


class StackedRSPForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, num_cells: int, dtype=torch.float32,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        cells = []
        for i in range(num_cells):
            if i == 0:
                cells.append(RSPForecaster(ctx_manager, inp_len, hidden_size, False, dtype))
            else:
                cells.append(RSPForecaster(ctx_manager, hidden_size, hidden_size, False, dtype))
        self.cells = nn.ModuleList(cells)
        self.out_linear = nn.Linear(inp_len + hidden_size, 1, dtype=dtype)

    def init_context(self, batch_size: int, device: Any, dtype=torch.float32, init_with: Any = 'zeros'):
        for cell in self.cells:
            cell.init_context(batch_size, device, dtype, init_with)

    def delete_context(self):
        for cell in self.cells:
            cell.delete_context()

    def set_mode(self, mode):
        for cell in self.cells:
            cell.set_mode(mode)

    def set_sample_index(self, i):
        for cell in self.cells:
            cell.set_sample_index(i)

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out = inp
        for cell in self.cells:
            out = cell(out)
        return self.out_linear(torch.cat((inp, out), dim=-1)), None
