from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

import torch

from typing import Any
from typing import Optional
from typing import Tuple


class RecurrentForecasterWithCorrectionBlock(PRForecaster):
    def __init__(self,
                 ctx_manager: RecurrentContextsManager,
                 baseline_model: PRForecaster,
                 correction_block: PRForecaster,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        self.baseline = baseline_model
        self.corrector = correction_block

    def init_context(self, batch_size: int, device: Any, dtype=torch.float32, init_with: Any = 'zeros'):
        self.baseline.init_context(batch_size, device, dtype, init_with)
        self.corrector.init_context(batch_size, device, dtype, init_with)

    def delete_context(self):
        self.baseline.delete_context()
        self.corrector.delete_context()

    def _forward(
            self,
            inp: torch.Tensor,
            ctx: Optional[torch.Tensor],
            prev_errors: torch.Tensor,
            baseline_no_grad: bool = True,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if baseline_no_grad:
            with torch.no_grad():
                baseline_forecast = self.baseline.forward(inp, prev_errors)
        else:
            baseline_forecast = self.baseline.forward(inp, prev_errors)
        correction = self.corrector(torch.cat((inp, baseline_forecast), dim=1), prev_errors)
        return baseline_forecast + correction, None
