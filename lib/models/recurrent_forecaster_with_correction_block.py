from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

from torch.nn.parameter import Parameter
import torch
import torch.nn as nn

from typing import Any
from typing import Optional
from typing import Tuple


class RecurrentForecasterWithCorrectionBlock(PRForecaster):
    def __init__(self,
                 ctx_manager: RecurrentContextsManager,
                 baseline_model: PRForecaster,
                 correction_block: PRForecaster,
                 correction_scaling: str,
                 inp_len: int,
                 *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        self.baseline = baseline_model
        self.corrector = correction_block
        self.correction_scaling = correction_scaling
        if correction_scaling == 'constrained':
            self.correction_scale_b = Parameter(data=torch.tensor(-1.))
            self.correction_scale_w = Parameter(data=torch.tensor(1.))
        elif correction_scaling == 'unconstrained':
            self.correction_scaler = nn.Sequential(nn.Linear(inp_len + 4, 1), nn.Sigmoid())

    def init_context(self, batch_size: int, device: Any, dtype=torch.float32, init_with: Any = 'zeros') -> None:
        self.baseline.init_context(batch_size, device, dtype, init_with)
        self.corrector.init_context(batch_size, device, dtype, init_with)

    def delete_context(self):
        self.baseline.delete_context()
        self.corrector.delete_context()

    def set_mode(self, mode):
        self.baseline.set_mode(mode)
        self.corrector.set_mode(mode)

    def set_sample_index(self, i):
        self.baseline.set_sample_index(i)
        self.corrector.set_sample_index(i)

    def _forward(
            self,
            inp: torch.Tensor,
            ctx: Optional[torch.Tensor],
            *args,
            baseline_no_grad: bool = True,
            **kwargs
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        if baseline_no_grad:
            with torch.no_grad():
                baseline_forecast = self.baseline.forward(inp)
        else:
            baseline_forecast = self.baseline.forward(inp)
        previous_baseline_errors, previous_actual_errors, prev_correction = args[:3]
        full_inp = torch.cat((inp, previous_baseline_errors, previous_actual_errors, prev_correction,
                              baseline_forecast), dim=-1)
        correction = self.corrector(full_inp)
        # TODO: scale based on previous baseline error and actual error + correction value as well - the smaller the
        #  actual error (given correction was big enough) - the better job our corrector did/is doing
        if self.correction_scaling == 'constrained':
            correction_sigmoid_exponent = (self.correction_scale_b +
                                           torch.abs(self.correction_scale_w) * torch.abs(previous_baseline_errors))
            correction_scale = 1. / (1 + torch.exp(-correction_sigmoid_exponent))
        elif self.correction_scaling == 'unconstrained':
            corrector_inp = torch.cat((inp, torch.abs(previous_baseline_errors), torch.abs(previous_actual_errors),
                                       prev_correction, baseline_forecast), dim=-1)
            correction_scale = self.correction_scaler(corrector_inp)
        else:
            correction_scale = 1.0
        correction = correction_scale * correction
        return (baseline_forecast, correction), None
