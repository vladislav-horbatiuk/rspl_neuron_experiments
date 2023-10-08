from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.models.recurrent_forecaster_with_correction_block import RecurrentForecasterWithCorrectionBlock

import torch


def run_on_inputs_with_targets(
        model: PRForecaster,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        setup_ctxs: bool = True,
        delete_ctxs: bool = True,
        model_mode: int = PRForecaster.MODE_TRAIN,
        *forward_args,
        **forward_kwargs
) -> torch.Tensor:
    num_samples, num_sequences = inputs.shape[:2]
    forecasts = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    if setup_ctxs:
        model.init_context(num_sequences, device=inputs.device, dtype=inputs.dtype)
    model.set_mode(model_mode)
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        model.set_sample_index(i)
        forecast = model(inp, *forward_args, **forward_kwargs)
        forecasts[i, :] = forecast
    if delete_ctxs:
        model.delete_context()
    return forecasts


def run_mod_with_correction_block_on_inputs_with_targets(
        model: RecurrentForecasterWithCorrectionBlock,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        setup_ctxs: bool = True,
        delete_ctxs: bool = True,
        model_mode: int = PRForecaster.MODE_TRAIN,
        *forward_args,
        **forward_kwargs
) -> torch.Tensor:
    num_samples, num_sequences = inputs.shape[:2]
    forecasts = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    corrections = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    previous_baseline_errors = torch.zeros(num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    previous_actual_errors = torch.zeros(num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    if setup_ctxs:
        model.init_context(num_sequences, device=inputs.device, dtype=inputs.dtype)
    model.set_mode(model_mode)
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        model.set_sample_index(i)
        baseline_forecast, correction = model(inp, previous_baseline_errors, previous_actual_errors,
                                              *forward_args, **forward_kwargs)
        full_forecast = baseline_forecast + correction
        forecasts[i, :] = full_forecast
        corrections[i, :] = correction
        with torch.no_grad():
            previous_baseline_errors = target - baseline_forecast
            previous_actual_errors = target - full_forecast
    if delete_ctxs:
        model.delete_context()
    return forecasts, corrections
