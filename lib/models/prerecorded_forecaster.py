from typing import Optional, Tuple, Any

from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.utils.recurrent_contexts_manager import RecurrentContextsManager

import torch


class PrerecordedForecaster(PRForecaster):
    def __init__(self, ctx_manager: RecurrentContextsManager,
                 train_targets: torch.Tensor, test_targets: torch.Tensor, *args, **kwargs):
        super().__init__(ctx_manager, *args, **kwargs)
        self.train_targets = train_targets
        self.test_targets = test_targets

    def _forward(self,
                 inp: torch.Tensor,
                 ctx: Optional[torch.Tensor],
                 *args,
                 **kwargs) -> Tuple[Any, Optional[torch.Tensor]]:
        targets = self.train_targets if self.mode == self.MODE_TRAIN else self.test_targets
        return targets[self.sample_index], None
