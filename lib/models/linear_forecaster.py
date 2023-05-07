from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

from torch import nn
import torch
import torch.linalg

from typing import Optional
from typing import Tuple


class LinearForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager, inp_len: int, dtype=torch.float32,
                 bias: bool = True, *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        self.predictor = nn.Linear(inp_len, 1, dtype=dtype, bias=bias)

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 prev_errors: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.predictor(inp), None

    @classmethod
    def find_optimal_least_squares_forecaster(cls,
                                              ctx_manager: RecurrentContextsManager,
                                              train_inp: torch.Tensor,
                                              train_targ: torch.Tensor,
                                              bias: bool = True) -> 'LinearForecaster':
        """
        :param ctx_manager: context manager, ignored
        :param train_inp: shape NxSxM, N - number of samples; S - number of sequences; M - number of features in sample
        :param train_targ: shape NxSx1
        :param bias: enable/disable bias
        :return: optimal in terms of least squares error forecaster
        """
        N, S, M = train_inp.shape
        train_inp = train_inp.reshape(-1, M)
        dtype = train_inp.dtype
        device = train_inp.device
        if bias:
            train_inp = torch.cat((train_inp, torch.ones((N * S, 1), dtype=dtype, device=device)), dim=1)
        X, Y = train_inp, train_targ.reshape(-1, 1)
        weights = torch.linalg.lstsq(X, Y).solution
        forecaster = LinearForecaster(ctx_manager, M, dtype=dtype, bias=bias)
        if not bias:
            forecaster.predictor.weight.data.copy_(torch.t(weights))
        else:
            forecaster.predictor.weight.data.copy_(torch.t(weights[:-1, :]))
            forecaster.predictor.bias.data.copy_(weights[-1, :])
        return forecaster
