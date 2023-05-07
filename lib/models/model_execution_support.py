from lib.models.potentially_recurrent_forecaster import PRForecaster

import torch


def run_on_inputs_with_targets(
        model: PRForecaster,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        setup_ctxs: bool = True,
        delete_ctxs: bool = True,
        prev_errors_no_grad: bool = True,
        *forward_args,
        **forward_kwargs
) -> torch.Tensor:
    num_samples, num_sequences = inputs.shape[:2]
    forecasts = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    previous_errors = torch.zeros(num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    if setup_ctxs:
        model.init_context(num_sequences, device=inputs.device, dtype=inputs.dtype)
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        forecast = model(inp, previous_errors, *forward_args, **forward_kwargs)
        if prev_errors_no_grad:
            with torch.no_grad():
                previous_errors = target - forecast
        else:
            previous_errors = target - forecast
        forecasts[i, :] = forecast
    if delete_ctxs:
        model.delete_context()
    return forecasts
