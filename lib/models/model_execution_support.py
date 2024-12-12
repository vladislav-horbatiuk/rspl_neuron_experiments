from __future__ import annotations

import time
from typing import Callable
from typing import Tuple

from lib.models.linear_forecaster import LinearForecaster
from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.models.recurrent_forecaster_with_correction_block import RecurrentForecasterWithCorrectionBlock

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lib.utils.recurrent_contexts_manager import RecurrentContextsManager


def run_on_inputs_with_targets(
        model: PRForecaster,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        setup_ctxs: bool = True,
        delete_ctxs: bool = False,
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
        delete_ctxs: bool = False,
        model_mode: int = PRForecaster.MODE_TRAIN,
        *forward_args,
        **forward_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_samples, num_sequences = inputs.shape[:2]
    forecasts = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    corrections = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    previous_baseline_errors = torch.zeros(num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    previous_actual_errors = torch.zeros(num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    prev_correction = torch.zeros(num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    if setup_ctxs:
        model.init_context(num_sequences, device=inputs.device, dtype=inputs.dtype)
    model.set_mode(model_mode)
    for i, (inp, target) in enumerate(zip(inputs, targets)):
        model.set_sample_index(i)
        baseline_forecast, curr_correction = model(inp, previous_baseline_errors.detach(), previous_actual_errors.detach(),
                                                   prev_correction.detach(), *forward_args, **forward_kwargs)
        full_forecast = baseline_forecast + curr_correction
        forecasts[i, :] = full_forecast
        corrections[i, :] = curr_correction
        with torch.no_grad():
            previous_baseline_errors = target - baseline_forecast
            previous_actual_errors = target - full_forecast
            prev_correction = curr_correction
    if delete_ctxs:
        model.delete_context()
    return forecasts, corrections


def sw(t: torch.Tensor, size: int) -> torch.Tensor:
    return t.unfold(dimension=1, size=size, step=1)


def get_train_test_data_from_ts(ts: torch.Tensor,
                                inp_size: int,
                                num_val_points: int,
                                device='cpu',
                                dtype=torch.float32):
    N = ts.shape[0]
    train_size = N - num_val_points
    train_ts = ts[:train_size]
    test_ts = ts[train_size - inp_size:]
    test_ts_size = test_ts.shape[0]
    return (
        sw(train_ts[:-1].view(1, train_size - 1), inp_size).permute(1, 0, 2).to(device, dtype=dtype),
        train_ts[inp_size:][:, None, None].to(device, dtype=dtype),
        sw(test_ts[:-1].view(1, test_ts_size - 1), inp_size).permute(1, 0, 2).to(device, dtype=dtype),
        test_ts[inp_size:][:, None, None].to(device, dtype=dtype)
    )


def fit_linear_predictor(ts, inp_size, num_val_points,
                         forecast_horizon, device='cpu', dtype=torch.float32):
    train_inp, train_targ, val_inp, val_targ = get_train_test_data_from_ts(
        ts, inp_size, num_val_points, device, dtype)
    with torch.no_grad():
        lin_mod = LinearForecaster.find_optimal_least_squares_forecaster(RecurrentContextsManager(),
                                                                         train_inp, train_targ, bias=True)
        lin_train_out = run_on_inputs_with_targets(lin_mod, train_inp, train_targ,
                                                   setup_ctxs=False, delete_ctxs=False)
        lin_val_out = run_on_inputs_with_targets(lin_mod, val_inp, val_targ,
                                                 setup_ctxs=False, delete_ctxs=False)
        curr_inp = torch.cat((val_inp[-1, :, 1:], val_targ[-1, :, :]), dim=-1)
        lin_forecasts = []
        for i in range(forecast_horizon):
            lin_mod_out = lin_mod(curr_inp)
            lin_forecasts.append(lin_mod_out.squeeze().item())
            curr_inp = torch.cat((curr_inp[:, 1:], lin_mod_out), dim=-1)
        return train_inp, train_targ, val_inp, val_targ, lin_train_out, lin_val_out, lin_forecasts, lin_mod


def train_and_forecast(model_constructor: Callable[[], PRForecaster],
            train_inp: torch.Tensor,
            train_targ: torch.Tensor,
            val_inp: torch.Tensor,
            val_targ: torch.Tensor,
            lin_train_out: torch.Tensor,
            lin_val_out: torch.Tensor,
            lin_mod: LinearForecaster,
            forecast_horizon: int,
            num_attempts: int = 3,
            max_epochs: int = 1000,
            min_epochs: int = 100,
            max_epochs_wo_improvement: int = 10,
            lr: float = 0.001,
            wd: float = 0,
            do_print: bool = True) -> list:
    criterion = nn.MSELoss()
    with torch.no_grad():
        model_train_targ = (train_targ - lin_train_out).detach()
        maxmindist = model_train_targ.max() - model_train_targ.min()
        model_train_targ = model_train_targ / maxmindist
        mean = model_train_targ.mean()
        model_train_targ = model_train_targ - mean
        #     mean, std = model_train_targ.mean(), model_train_targ.std()
        #     model_train_targ = (model_train_targ - mean) / std
        #     plt.figure()
        #     plt.plot(range(model_train_targ.shape[0]), model_train_targ.squeeze().detach().numpy())
        #     plt.show()
        model_val_targ = (val_targ - lin_val_out).detach()
        #     model_val_targ = (model_val_targ - mean) / std
        model_val_targ = model_val_targ / maxmindist - mean

    best_val_loss_per_attempt = [None] * num_attempts

    epoch_times = []
    for attempt_idx in range(num_attempts):
        best_val_loss = None
        model = model_constructor()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        # begin to train
        epochs_wo_improv = 0
        for i in range(max_epochs):
            start = time.perf_counter_ns()
            if do_print:
                print('Epoch: ', i + 1)
            optimizer.zero_grad()
            train_out = run_on_inputs_with_targets(
                model, train_inp, model_train_targ, baseline_no_grad=True, model_mode=PRForecaster.MODE_TRAIN)
            loss = criterion(train_out, model_train_targ)
            rescaled_loss = (train_targ - ((train_out + mean) * maxmindist + lin_train_out)).square().mean().item()
            if do_print:
                print(f'Train loss: {rescaled_loss}')
            loss.backward()
            optimizer.step()
            end = time.perf_counter_ns()
            epoch_times.append((end - start) / 1_000_000_000)
            with torch.no_grad():
                val_out = run_on_inputs_with_targets(
                    model, val_inp, model_val_targ, setup_ctxs=False, baseline_no_grad=True, model_mode=PRForecaster.MODE_TEST)
                loss = criterion(val_out, model_val_targ)
                rescaled_loss = (val_targ - ((val_out + mean) * maxmindist + lin_val_out)).square().mean().item()
                if do_print:
                    print(f'Test loss: {rescaled_loss}')
                val_loss = loss.item()
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_per_attempt[attempt_idx] = best_val_loss
                    torch.save(model.state_dict(), f'best_chkpt_attempt{attempt_idx}.pt')
                    epochs_wo_improv = 0
                elif i >= min_epochs:
                    epochs_wo_improv += 1
                    if epochs_wo_improv == max_epochs_wo_improvement:
                        model.delete_context()
                        break
                model.delete_context()
    print(f'Avg epoch time: {np.array(epoch_times).mean()}')
    with torch.no_grad():
        print(f'Val losses for all attempts: {best_val_loss_per_attempt}')
        best_attempt = np.argmin(best_val_loss_per_attempt)
        model = model_constructor()
        print(f'Loading model at path: best_chkpt_attempt{best_attempt}.pt')
        model.load_state_dict(torch.load(f'best_chkpt_attempt{best_attempt}.pt'))
        full_inp = torch.cat((train_inp, val_inp), dim=0)
        full_targ = torch.cat((train_targ, val_targ), dim=0)
        _ = run_on_inputs_with_targets(model, full_inp, full_targ, setup_ctxs=True,
                                       delete_ctxs=False, model_mode=PRForecaster.MODE_TEST)
        curr_inp = torch.cat((val_inp[-1, :, 1:], val_targ[-1, :, :]), dim=-1)
        predictions = []
        for i in range(forecast_horizon):
            lin_mod_out = lin_mod(curr_inp)
            mod_out = model(curr_inp)
            comb_out = lin_mod_out + (mod_out + mean) * maxmindist
            predictions.append(comb_out.squeeze().item())
            curr_inp = torch.cat((curr_inp[:, 1:], comb_out), dim=-1)
        return predictions