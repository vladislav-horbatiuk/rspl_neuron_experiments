from __future__ import annotations

from typing import Callable

from lib.models.linear_forecaster import LinearForecaster
from lib.models.potentially_recurrent_forecaster import PRForecaster
from lib.models.recurrent_forecaster_with_correction_block import RecurrentForecasterWithCorrectionBlock

import torch
import torch.nn as nn
import torch.optim as optim

from lib.utils.recurrent_contexts_manager import RecurrentContextsManager


def run_on_inputs(
        model: PRForecaster,
        inputs: torch.Tensor,
        setup_ctxs: bool = True,
        delete_ctxs: bool = False,
        *forward_args,
        **forward_kwargs
) -> torch.Tensor:
    num_samples, num_sequences = inputs.shape[:2]
    forecasts = torch.zeros(num_samples, num_sequences, 1, dtype=inputs.dtype, device=inputs.device)
    if setup_ctxs:
        model.init_context(num_sequences, device=inputs.device, dtype=inputs.dtype)
    for i, inp in enumerate(inputs):
        forecast = model(inp, *forward_args, **forward_kwargs)
        forecasts[i, :] = forecast
    if delete_ctxs:
        model.delete_context()
    return forecasts


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
        lin_train_out = run_on_inputs(lin_mod, train_inp, setup_ctxs=False)
        lin_val_out = run_on_inputs(lin_mod, val_inp, setup_ctxs=False)
        curr_inp = torch.cat((val_inp[-1, :, 1:], val_targ[-1, :, :]), dim=-1)
        lin_forecasts = []
        for _ in range(forecast_horizon):
            lin_mod_out = lin_mod(curr_inp)
            lin_forecasts.append(lin_mod_out.squeeze().item())
            curr_inp = torch.cat((curr_inp[:, 1:], lin_mod_out), dim=-1)
        return train_inp, train_targ, val_inp, val_targ, lin_train_out, lin_val_out, lin_forecasts, lin_mod


def train_and_forecast(model_constructor: Callable[[], PRForecaster],
            out_dir: str,
            train_inp: torch.Tensor,
            train_targ: torch.Tensor,
            val_inp: torch.Tensor,
            val_targ: torch.Tensor,
            lin_train_out: torch.Tensor,
            lin_val_out: torch.Tensor,
            lin_mod: LinearForecaster,
            forecast_horizon: int,
            max_epochs: int = 1000,
            min_epochs: int = 100,
            max_epochs_wo_improvement: int = 20,
            lr: float = 0.001,
            wd: float = 5e-4,
            ) -> list:
    with torch.no_grad():
        model_train_targ = (train_targ - lin_train_out).detach()
        maxmindist = model_train_targ.max() - model_train_targ.min()
        model_train_targ = model_train_targ / maxmindist
        mean = model_train_targ.mean()
        model_train_targ = model_train_targ - mean
        model_val_targ = (val_targ - lin_val_out).detach()
        model_val_targ = model_val_targ / maxmindist - mean

    criterion = nn.MSELoss()
    best_val_loss = None
    model = model_constructor()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    epochs_wo_improv = 0
    for i in range(max_epochs):
        optimizer.zero_grad()
        train_out = run_on_inputs(model, train_inp)
        loss = criterion(train_out, model_train_targ)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_out = run_on_inputs(model, val_inp, setup_ctxs=False)
            val_loss = criterion(val_out, model_val_targ).item()
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{out_dir}/best_chkpt.pt')
                epochs_wo_improv = 0
            elif i >= min_epochs:
                epochs_wo_improv += 1
                if epochs_wo_improv == max_epochs_wo_improvement:
                    model.delete_context()
                    break
                model.delete_context()
    with torch.no_grad():
        model = model_constructor()
        model.load_state_dict(torch.load(f'{out_dir}/best_chkpt.pt'))
        full_inp = torch.cat((train_inp, val_inp), dim=0)
        _ = run_on_inputs(model, full_inp)
        curr_inp = torch.cat((val_inp[-1, :, 1:], val_targ[-1, :, :]), dim=-1)
        predictions = []
        for i in range(forecast_horizon):
            lin_mod_out = lin_mod(curr_inp)
            mod_out = model(curr_inp)
            comb_out = lin_mod_out + (mod_out + mean) * maxmindist
            predictions.append(comb_out.squeeze().item())
            curr_inp = torch.cat((curr_inp[:, 1:], comb_out), dim=-1)
        model.delete_context()
        return predictions