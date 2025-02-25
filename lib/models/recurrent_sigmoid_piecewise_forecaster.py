from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any
from typing import Optional
from typing import Tuple


class RSPForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, use_out_linear=False,
                 normalize_ctx=False, dtype=torch.float32,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        inp_size = inp_len + hidden_size
        self.ctx_shape = torch.Size([hidden_size])
        self.sigm_lin = nn.Linear(inp_size, hidden_size, dtype=dtype)
        self.w_ctx_cand_lin = nn.Linear(inp_size, hidden_size, dtype=dtype)
        self.use_out_linear = use_out_linear
        if use_out_linear:
            self.out_linear = nn.Linear(inp_len + hidden_size, 1, dtype=dtype)
        self.normalize_ctx = normalize_ctx

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
        if self.normalize_ctx:
            new_ctx = F.normalize(new_ctx, dim=-1)
        out = new_ctx
        if self.use_out_linear:
            out = self.out_linear(torch.cat((orig_inp, new_ctx), dim=-1))
        return out, new_ctx


class LSTMForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, use_out_linear=True,
                 normalize_ctx=False, dtype=torch.float32,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        self.hidden_size = hidden_size
        self.ctx_shape = torch.Size([2 * hidden_size])
        self.lstm_cell = nn.LSTMCell(inp_len, hidden_size, dtype=dtype)
        self.use_out_linear = use_out_linear
        if use_out_linear:
            self.out_linear = nn.Linear(inp_len + 2 * hidden_size, 1, dtype=dtype)
        self.normalize_ctx = normalize_ctx

    def get_ctx_shape(self) -> Optional[torch.Size]:
        return self.ctx_shape

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_state, cell_state = ctx[..., :self.hidden_size], ctx[..., self.hidden_size:]
        new_hidden, new_cell = self.lstm_cell(inp, (hidden_state, cell_state))
        new_ctx = torch.cat((new_hidden, new_cell), dim=-1)
        if self.normalize_ctx:
            new_ctx = F.normalize(new_ctx, dim=-1)        
        out = new_hidden
        if self.use_out_linear:
            out = self.out_linear(torch.cat((inp, new_ctx), dim=-1))
        return out, new_ctx


class GRUForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, use_out_linear=True,
                 normalize_ctx=False, dtype=torch.float32,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        self.hidden_size = hidden_size
        self.ctx_shape = torch.Size([hidden_size])
        self.gru_cell = nn.GRUCell(inp_len, hidden_size, dtype=dtype)
        self.use_out_linear = use_out_linear
        if use_out_linear:
            self.out_linear = nn.Linear(inp_len + hidden_size, 1, dtype=dtype)
        self.normalize_ctx = normalize_ctx

    def get_ctx_shape(self) -> Optional[torch.Size]:
        return self.ctx_shape

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_ctx = self.gru_cell(inp, ctx)
        if self.normalize_ctx:
            new_ctx = F.normalize(new_ctx, dim=-1)
        out = new_ctx
        if self.use_out_linear:
            out = self.out_linear(torch.cat((inp, new_ctx), dim=-1))
        return out, new_ctx


class StackedRCellsForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 inp_len: int, hidden_size: int, num_cells: int, dtype=torch.float32,
                 CellType=RSPForecaster,
                 cell_kwargs={},
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        cells = []
        for i in range(num_cells):
            if i == 0:
                cells.append(CellType(ctx_manager, inp_len, hidden_size, use_out_linear=False, dtype=dtype, **cell_kwargs))
            else:
                cells.append(CellType(ctx_manager, hidden_size, hidden_size, use_out_linear=False, dtype=dtype, **cell_kwargs))
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
