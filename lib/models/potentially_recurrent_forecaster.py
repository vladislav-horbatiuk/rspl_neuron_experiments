from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

from torch import nn
import torch

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Tuple
from uuid import uuid4


class PRForecaster(nn.Module, ABC):
    MODE_TRAIN = 0
    MODE_TEST = 1

    def __init__(self, ctx_manager: RecurrentContextsManager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ctx_manager = ctx_manager
        self.cid = str(uuid4())
        self.sample_index = 0
        self.mode = self.MODE_TRAIN

    def set_sample_index(self, i):
        self.sample_index = i

    def set_mode(self, mode):
        self.mode = mode

    # noinspection PyMethodMayBeStatic
    def get_ctx_shape(self) -> Optional[torch.Size]:
        return None

    def init_context(self, batch_size: int, device: Any, dtype=torch.float32, init_with: Any = 'zeros') -> None:
        baseline_shape = self.get_ctx_shape()
        if baseline_shape is None:
            return
        ctx_shape = torch.Size([batch_size] + list(baseline_shape))
        self.ctx_manager.init_ctx_with_id(self.cid, ctx_shape, device, dtype, init_with)

    def delete_context(self) -> None:
        if self.get_ctx_shape() is not None:
            self.ctx_manager.del_ctx(self.cid)

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ctx = self.ctx_manager.get_ctx(self.cid)
        out, new_ctx = self._forward(inp, ctx, *args, **kwargs)
        self.ctx_manager.set_ctx(self.cid, new_ctx)
        return out

    @abstractmethod
    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 *args,
                 **kwargs) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Run the model on input tensor using a current context tensor; return forecast and new context tensors.

        If the specific model does not use recurrent context - just return None as a second output and ignore ctx
        input.

        :param inp: input sample tensor, shape SxM: S - number of sequences, M - number of input features in a sample.
        :param ctx: context tensor, shape SxC1x...xCn: S - number of sequences, C1x...xCn - shape returned from
        get_ctx_shape. If the get_ctx_shape method returns None - context tensor input will also be None.
        :param args: any additional positional arguments specific to your model
        :param kwargs: any additional keyword arguments specific to your model
        :return: tuple <forecast output, new context tensor>; forecast output is "scheme-dependent" - in a basic scheme
        it's just a Sx1 forecasts' tensor, but generally it can be anything as long as corresponding "forecast scheme"
        can properly handle it; new context tensor shape (if provided) should be SxC1x...xCn, or you can just return
        None if no recurrent context is used for your model.
        """
        pass
