import torch
from Evaluation.metrics import safe_r2, safe_mape
from Data.preprocessing import assert_finite
from Evaluation.comparison_plots import save_error_plots
import torch.nn as nn
import torch
import copy
import math
import time 
import json
from pathlib import Path
import sys
from tqdm.auto import tqdm

sys.path.append("../src")



from Evaluation.metrics import (
    METRIC_STANDARD_MODIFIED,
    _finalize_r2_tracker,
    _init_r2_tracker,
    _update_r2_tracker,
    normalize_metric_standard,
    safe_mape,
    safe_r2,
)
from Evaluation.comparison_plots import save_error_plots
from Data.preprocessing import assert_finite
from Training.experiment_runner import unpack_model_output
from output_layout import logs_directory, train_history_directory

loss_fn = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


_ADAPTATIVE_LR_METRIC_MODES = {
    "loss": "min",
    "mse": "min",
    "rmse": "min",
    "mae": "min",
    "mape": "min",
    "r2": "max",
    "r2_batch_mean": "max",
}


def resolve_adaptative_lr_metric(metric):
    """Normalize a validation metric and return its ReduceLROnPlateau mode."""
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("adaptative_lr_metric must be a non-empty string.")

    normalized = metric.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "r_2": "r2",
        "r2_score": "r2",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
        "mean_absolute_error": "mae",
        "mean_absolute_percentage_error": "mape",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return normalized, _ADAPTATIVE_LR_METRIC_MODES[normalized]
    except KeyError as exc:
        available = ", ".join(_ADAPTATIVE_LR_METRIC_MODES)
        raise ValueError(
            f"Unsupported adaptative_lr_metric={metric!r}. Available metrics: {available}."
        ) from exc


def adaptative_lr_metric_value(val_metrics, metric):
    """Return the scalar validation value monitored by ReduceLROnPlateau."""
    normalized, _mode = resolve_adaptative_lr_metric(metric)
    if normalized == "rmse":
        if "rmse" in val_metrics:
            return float(val_metrics["rmse"])
        mse = float(val_metrics["mse"])
        return math.sqrt(max(mse, 0.0))
    return float(val_metrics[normalized])


def _format_progress_metric(value):
    """Format train progress metrics compactly for tqdm postfixes."""
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.4g}"


_MODIFIED_MONITOR_METRICS = frozenset({"mse", "rmse", "mae", "r2", "r2_batch_mean"})


def resolve_training_metric_monitors(metric_standard, adaptative_lr_metric):
    """Resolve scheduler and early-stopping metrics without changing legacy runs."""
    metric_standard = normalize_metric_standard(metric_standard)
    requested_metric, requested_mode = resolve_adaptative_lr_metric(adaptative_lr_metric)
    scheduler_metric, scheduler_mode = requested_metric, requested_mode
    early_stopping_metric, early_stopping_mode = "loss", "min"

    if metric_standard == METRIC_STANDARD_MODIFIED:
        # Loss and MAPE are intentionally not threshold-filtered. Fall back to
        # modified RMSE so every patience-based decision uses eligible targets.
        if requested_metric not in _MODIFIED_MONITOR_METRICS:
            scheduler_metric, scheduler_mode = "rmse", "min"
        early_stopping_metric, early_stopping_mode = scheduler_metric, scheduler_mode

    return {
        "requested_metric": requested_metric,
        "requested_mode": requested_mode,
        "scheduler_metric": scheduler_metric,
        "scheduler_mode": scheduler_mode,
        "early_stopping_metric": early_stopping_metric,
        "early_stopping_mode": early_stopping_mode,
    }


def _finite_target_scale_threshold(metric_threshold):
    """Validate a threshold after an optional scaler transform."""
    try:
        threshold = float(metric_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric_threshold in target scale must be finite.") from exc
    if not math.isfinite(threshold):
        raise ValueError("metric_threshold in target scale must be finite.")
    return threshold


def node_standard_deviation_target(y_true):
    """Return the population standard deviation across nodes for each lead day.

    Targets are expected in the model layout ``[batch, lead_day, node]``.  The
    zero correction intentionally matches the population standard deviation
    used by the saved cross-node dispersion diagnostics.
    """
    if not isinstance(y_true, torch.Tensor):
        raise TypeError("y_true must be a torch.Tensor.")
    if y_true.ndim != 3:
        raise ValueError(
            "y_true must have shape [batch, lead_day, node] to compute node standard deviation."
        )
    if y_true.shape[-1] < 1:
        raise ValueError("y_true must contain at least one node.")
    return torch.std(y_true, dim=-1, correction=0)


def _new_metric_accumulator(device_local):
    return {
        "mse_batch_sum": torch.zeros((), device=device_local),
        "mae_batch_sum": torch.zeros((), device=device_local),
        "squared_error_sum": torch.zeros((), dtype=torch.float64, device=device_local),
        "absolute_error_sum": torch.zeros((), dtype=torch.float64, device=device_local),
        "selected_count": torch.zeros((), dtype=torch.float64, device=device_local),
        "r2_batch_sum": torch.zeros((), device=device_local),
        "r2_batch_count": 0,
        "steps": 0,
        "r2_tracker": None,
    }


def _update_metric_accumulator(
    accumulator,
    y_true,
    y_pred,
    *,
    metric_standard,
    metric_threshold,
):
    modified = metric_standard == METRIC_STANDARD_MODIFIED
    if accumulator["r2_tracker"] is None:
        accumulator["r2_tracker"] = _init_r2_tracker(
            y_true.shape[1],
            device=y_true.device,
            undefined_as_nan=modified,
        )

    metric_mask = None
    if modified:
        metric_mask = (
            (y_true > metric_threshold)
            & torch.isfinite(y_true)
            & torch.isfinite(y_pred)
        )
    _update_r2_tracker(accumulator["r2_tracker"], y_true, y_pred, metric_mask=metric_mask)

    if modified:
        selected_count = metric_mask.sum()
        if selected_count.item() > 0:
            selected_true = y_true[metric_mask]
            selected_pred = y_pred[metric_mask]
            residual = (selected_true - selected_pred).to(torch.float64)
            accumulator["squared_error_sum"] += residual.square().sum()
            accumulator["absolute_error_sum"] += residual.abs().sum()
            accumulator["selected_count"] += selected_count.to(torch.float64)
            selected_true_64 = selected_true.to(torch.float64)
            batch_ss_tot = (selected_true_64 - selected_true_64.mean()).square().sum()
            if batch_ss_tot.item() > 1e-8:
                batch_r2 = 1.0 - residual.square().sum() / batch_ss_tot
                accumulator["r2_batch_sum"] += batch_r2.to(
                    accumulator["r2_batch_sum"].dtype
                )
                accumulator["r2_batch_count"] += 1
    else:
        accumulator["mse_batch_sum"] += torch.mean((y_pred - y_true).square()).detach()
        accumulator["mae_batch_sum"] += torch.mean(torch.abs(y_pred - y_true)).detach()
        accumulator["r2_batch_sum"] += safe_r2(
            y_true.reshape(-1), y_pred.reshape(-1)
        ).detach()
        accumulator["r2_batch_count"] += 1
    accumulator["steps"] += 1


def _finalize_metric_accumulator(accumulator, metric_standard):
    r2_tracker = accumulator["r2_tracker"]
    if r2_tracker is None:
        r2_metrics = {
            "global": float("nan"),
            "per_step": [],
            "count": 0,
            "count_per_step": [],
        }
    else:
        r2_metrics = _finalize_r2_tracker(r2_tracker)

    if metric_standard == METRIC_STANDARD_MODIFIED:
        count = int(accumulator["selected_count"].detach().cpu().item())
        if count:
            mse = float((accumulator["squared_error_sum"] / count).detach().cpu().item())
            mae = float((accumulator["absolute_error_sum"] / count).detach().cpu().item())
        else:
            mse = float("nan")
            mae = float("nan")
    else:
        steps = max(accumulator["steps"], 1)
        mse = float((accumulator["mse_batch_sum"] / steps).detach().cpu().item())
        mae = float((accumulator["mae_batch_sum"] / steps).detach().cpu().item())

    r2_batch_count = accumulator["r2_batch_count"]
    r2_batch_mean = (
        float((accumulator["r2_batch_sum"] / r2_batch_count).detach().cpu().item())
        if r2_batch_count
        else float("nan")
    )
    return {
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)) if math.isfinite(mse) else float("nan"),
        "mae": mae,
        "r2": r2_metrics["global"],
        "r2_by_step": r2_metrics["per_step"],
        "r2_batch_mean": r2_batch_mean,
        "metric_target_count": r2_metrics["count"],
        "metric_target_count_by_step": r2_metrics["count_per_step"],
    }



def eval_with_loader_stable(
    model,
    loader,
    criterion,
    use_amp,
    amp_device,
    amp_dtype,
    debug_checks=False,
    metric_standard=None,
    metric_threshold=0.0,
    metric_threshold_mm=None,
    learn_std=False,
):
    """Evaluate a loader using standard or threshold-filtered regression metrics.

    ``metric_threshold`` is already expressed in the target tensor scale. The
    optional ``metric_threshold_mm`` value is output metadata used by the run
    artifacts and does not participate in tensor comparisons.
    """
    metric_standard = normalize_metric_standard(metric_standard)
    metric_threshold = _finite_target_scale_threshold(metric_threshold)
    device_local = next(model.parameters()).device
    model.eval()
    total_loss = torch.zeros((), device=device_local)
    if learn_std:
        total_prediction_loss = torch.zeros((), device=device_local)
        total_std_loss = torch.zeros((), device=device_local)
    total_mape = torch.zeros((), device=device_local)
    steps = 0
    loss_fn = criterion if criterion is not None else nn.MSELoss()
    metric_accumulator = _new_metric_accumulator(device_local)

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device_local, non_blocking=True)
            yb = yb.to(device_local, non_blocking=True)
            if debug_checks:
                assert_finite(f"eval_xb_batch{batch_idx}", xb)
                assert_finite(f"eval_yb_batch{batch_idx}", yb)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                if learn_std:
                    output = model(xb)
                    pred, pred_std = unpack_model_output(output)
                    if pred_std is None:
                        raise ValueError(
                            "learn_std=True requires the model to return (forecast, predicted_std)."
                        )
                    target_std = node_standard_deviation_target(yb)
                    if pred_std.shape != target_std.shape:
                        raise ValueError(
                            "predicted_std must have shape [batch, lead_day]. "
                            f"Expected {tuple(target_std.shape)}, got {tuple(pred_std.shape)}."
                        )
                    if debug_checks:
                        assert_finite(f"eval_pred_batch{batch_idx}", pred)
                        assert_finite(f"eval_pred_std_batch{batch_idx}", pred_std)
                        assert_finite(f"eval_target_std_batch{batch_idx}", target_std)
                    prediction_loss = loss_fn(pred, yb)
                    std_loss = nn.functional.mse_loss(pred_std, target_std)
                    loss = prediction_loss + std_loss
                else:
                    pred = model(xb)
                    if debug_checks:
                        assert_finite(f"eval_pred_batch{batch_idx}", pred)
                    loss = loss_fn(pred, yb)

            if debug_checks and (not torch.isfinite(loss)):
                raise RuntimeError(f"non-finite eval loss on batch={batch_idx}")

            _update_metric_accumulator(
                metric_accumulator,
                yb,
                pred,
                metric_standard=metric_standard,
                metric_threshold=metric_threshold,
            )

            total_loss += loss.detach()
            if learn_std:
                total_prediction_loss += prediction_loss.detach()
                total_std_loss += std_loss.detach()
            total_mape += safe_mape(yb, pred, eps=1e-3).detach()
            steps += 1

    regression_metrics = _finalize_metric_accumulator(metric_accumulator, metric_standard)
    metrics = {
        "loss": (total_loss / max(steps, 1)).item(),
        "mape": (total_mape / max(steps, 1)).item(),
        **regression_metrics,
        "metric_standard": metric_standard,
        "metric_threshold_mm": (
            float(metric_threshold_mm)
            if metric_threshold_mm is not None
            else (metric_threshold if metric_standard == METRIC_STANDARD_MODIFIED else None)
        ),
        "metric_threshold_target_scale": (
            metric_threshold if metric_standard == METRIC_STANDARD_MODIFIED else None
        ),
    }
    if learn_std:
        metrics.update(
            {
                "prediction_loss": (total_prediction_loss / max(steps, 1)).item(),
                "std_loss": (total_std_loss / max(steps, 1)).item(),
            }
        )
    return metrics


def train_stable(
    model,
    train_loader,
    val_loader,
    window_size,
    horizon,
    hidden_dim,
    epochs,
    lr,
    weight_decay,
    patience,
    criterion=None,
    run_dir=None,
    adj_lr_factor=0.25,
    max_norm=0.8,
    clamp_logits=6.0,
    use_amp=None,
    debug_checks=False,
    loss_name=None,
    loss_metadata=None,
    warm_up=0,
    adaptative_lr_metric="loss",
    metric_standard=None,
    metric_threshold=0.0,
    metric_threshold_mm=None,
    learn_std=False,
):
    if warm_up < 0:
        raise ValueError("warm_up must be non-negative.")
    metric_standard = normalize_metric_standard(metric_standard)
    metric_threshold = _finite_target_scale_threshold(metric_threshold)

    device_local = "cuda" if torch.cuda.is_available() else "cpu"
    model = copy.deepcopy(model).to(device_local)

    if isinstance(criterion, nn.Module):
        criterion = criterion.to(device_local)
    loss_fn = criterion if criterion is not None else nn.MSELoss()

    base_params, adj_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "a_logits" in name:
            adj_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": lr})
    if adj_params:
        # Adjacency uses an explicit anchor loss below. Applying Adam's
        # ordinary weight decay here would instead pull absolute/log-residual
        # parameters towards zero without documenting the selected graph prior.
        param_groups.append(
            {
                "params": adj_params,
                "lr": max(lr * adj_lr_factor, 1e-5),
                "weight_decay": 0.0,
            }
        )

    adjacency_anchor = getattr(model, "adjacency_anchor_loss", None)
    has_adjacency_anchor = bool(adj_params) and callable(adjacency_anchor)

    monitors = resolve_training_metric_monitors(
        metric_standard,
        adaptative_lr_metric
    )
    requested_adaptative_lr_metric = monitors["requested_metric"]
    adaptative_lr_metric = monitors["scheduler_metric"]
    adaptative_lr_mode = monitors["scheduler_mode"]
    early_stopping_metric = monitors["early_stopping_metric"]
    early_stopping_mode = monitors["early_stopping_mode"]
    optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode=adaptative_lr_mode,
                                                           factor=0.1,
                                                           patience=patience//2)

    use_amp = (device_local == "cuda") if use_amp is None else use_amp
    amp_device = "cuda" if device_local == "cuda" else "cpu"
    amp_dtype = torch.float16 if amp_device == "cuda" else torch.bfloat16
    scaler = torch.amp.GradScaler(amp_device, enabled=(use_amp and amp_device == "cuda"))
    tqdm.write(
        f"train_stable: use_amp={use_amp} adj_lr_factor={adj_lr_factor} "
        f"adaptative_lr_metric={adaptative_lr_metric} mode={adaptative_lr_mode} "
        f"early_stopping_metric={early_stopping_metric} metric_standard={metric_standard}"
    )

    best_model = copy.deepcopy(model.state_dict())
    best_monitor_value = float("inf") if early_stopping_mode == "min" else float("-inf")
    best_val_loss = None
    counter = 0
    has_post_warm_up_checkpoint = False
    monitor_updates = 0

    history = {
        "train_loss": [], "train_mse": [], "train_mae": [], "train_mape": [],
        "train_r2": [], "train_r2_batch_mean": [], "train_r2_by_step": [],
        "train_metric_target_count": [], "train_metric_target_count_by_step": [],
        "val_loss": [], "val_mse": [], "val_mae": [], "val_mape": [],
        "val_r2": [], "val_r2_batch_mean": [], "val_r2_by_step": [],
        "val_metric_target_count": [], "val_metric_target_count_by_step": [],
        "epoch_time": [], "train_adjacency_anchor_loss": [],
    }
    if learn_std:
        history.update(
            {
                "train_prediction_loss": [],
                "train_std_loss": [],
                "val_prediction_loss": [],
                "val_std_loss": [],
            }
        )

    progress_bar = tqdm(
        range(epochs),
        total=epochs,
        desc="Training",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )
    progress_bar.set_postfix(
        {
            "train_loss": "nan",
            "val_loss": "nan",
            "train_mse": "nan",
            "val_mse": "nan",
            "train_r2": "nan",
            "val_r2": "nan",
        }
    )

    for ep in progress_bar:
        t0 = time.perf_counter()
        model.train()
        train_loss = torch.zeros((), device=device_local)
        train_adjacency_anchor_loss = torch.zeros((), device=device_local)
        if learn_std:
            train_prediction_loss = torch.zeros((), device=device_local)
            train_std_loss = torch.zeros((), device=device_local)
        train_mape = torch.zeros((), device=device_local)
        train_metric_accumulator = _new_metric_accumulator(device_local)
        steps = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device_local, non_blocking=True)
            yb = yb.to(device_local, non_blocking=True)
            if debug_checks:
                assert_finite(f"train_xb_epoch{ep}_batch{batch_idx}", xb)
                assert_finite(f"train_yb_epoch{ep}_batch{batch_idx}", yb)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                if learn_std:
                    output = model(xb)
                    pred, pred_std = unpack_model_output(output)
                    if pred_std is None:
                        raise ValueError(
                            "learn_std=True requires the model to return (forecast, predicted_std)."
                        )
                    target_std = node_standard_deviation_target(yb)
                    if pred_std.shape != target_std.shape:
                        raise ValueError(
                            "predicted_std must have shape [batch, lead_day]. "
                            f"Expected {tuple(target_std.shape)}, got {tuple(pred_std.shape)}."
                        )
                    if debug_checks:
                        assert_finite(f"train_pred_epoch{ep}_batch{batch_idx}", pred)
                        assert_finite(f"train_pred_std_epoch{ep}_batch{batch_idx}", pred_std)
                        assert_finite(f"train_target_std_epoch{ep}_batch{batch_idx}", target_std)
                    prediction_loss = loss_fn(pred, yb)
                    std_loss = nn.functional.mse_loss(pred_std, target_std)
                    loss = prediction_loss + std_loss
                else:
                    pred = model(xb)
                    if debug_checks:
                        assert_finite(f"train_pred_epoch{ep}_batch{batch_idx}", pred)
                    loss = loss_fn(pred, yb)

                adjacency_anchor_loss = loss.new_zeros(())
                if has_adjacency_anchor:
                    adjacency_anchor_loss = adjacency_anchor()
                    if adjacency_anchor_loss.ndim != 0:
                        raise ValueError("adjacency_anchor_loss() must return a scalar tensor.")
                    # Match the conventional 0.5 * weight_decay * ||theta||²
                    # form, but center it at the saved adjacency prior.
                    loss = loss + float(weight_decay) * adjacency_anchor_loss

            if debug_checks and (not torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss epoch={ep} batch={batch_idx}")

            scaler.scale(loss).backward()

            if use_amp:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            if debug_checks:
                for name, param in model.named_parameters():
                    if param.grad is not None and (not torch.isfinite(param.grad).all()):
                        raise RuntimeError(f"non-finite gradient in {name} epoch={ep} batch={batch_idx}")

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "a_logits" in name:
                        param.clamp_(-clamp_logits, clamp_logits)

            _update_metric_accumulator(
                train_metric_accumulator,
                yb,
                pred,
                metric_standard=metric_standard,
                metric_threshold=metric_threshold,
            )

            train_loss += loss.detach()
            train_adjacency_anchor_loss += adjacency_anchor_loss.detach()
            if learn_std:
                train_prediction_loss += prediction_loss.detach()
                train_std_loss += std_loss.detach()
            train_mape += safe_mape(yb, pred, eps=1e-3).detach()
            steps += 1

        train_metrics = _finalize_metric_accumulator(
            train_metric_accumulator,
            metric_standard,
        )

        history["train_loss"].append((train_loss / max(steps, 1)).item())
        history["train_adjacency_anchor_loss"].append(
            (train_adjacency_anchor_loss / max(steps, 1)).item()
        )
        if learn_std:
            history["train_prediction_loss"].append(
                (train_prediction_loss / max(steps, 1)).item()
            )
            history["train_std_loss"].append((train_std_loss / max(steps, 1)).item())
        history["train_mse"].append(train_metrics["mse"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_mape"].append((train_mape / max(steps, 1)).item())
        history["train_r2"].append(train_metrics["r2"])
        history["train_r2_batch_mean"].append(train_metrics["r2_batch_mean"])
        history["train_r2_by_step"].append(train_metrics["r2_by_step"])
        history["train_metric_target_count"].append(train_metrics["metric_target_count"])
        history["train_metric_target_count_by_step"].append(
            train_metrics["metric_target_count_by_step"]
        )

        val_metrics = eval_with_loader_stable(
            model,
            val_loader,
            criterion=criterion,
            use_amp=use_amp,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            debug_checks=debug_checks,
            metric_standard=metric_standard,
            metric_threshold=metric_threshold,
            metric_threshold_mm=metric_threshold_mm,
            learn_std=learn_std,
        )
        # Validation is still recorded during warm-up for diagnostics, but it
        # must not influence any patience-based decision.  Monitoring starts
        # only after ``warm_up`` completed epochs, so the first eligible
        # validation establishes the early-stopping baseline.
        warm_up_finished = ep + 1 > warm_up
        scheduler_value = adaptative_lr_metric_value(val_metrics, adaptative_lr_metric)
        if warm_up_finished and math.isfinite(scheduler_value):
            scheduler.step(scheduler_value)
        history["val_loss"].append(val_metrics["loss"])
        if learn_std:
            history["val_prediction_loss"].append(val_metrics["prediction_loss"])
            history["val_std_loss"].append(val_metrics["std_loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_mape"].append(val_metrics["mape"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_r2_batch_mean"].append(val_metrics["r2_batch_mean"])
        history["val_r2_by_step"].append(val_metrics["r2_by_step"])
        history["val_metric_target_count"].append(val_metrics["metric_target_count"])
        history["val_metric_target_count_by_step"].append(
            val_metrics["metric_target_count_by_step"]
        )

        history["epoch_time"].append(time.perf_counter() - t0)

        progress_bar.set_postfix(
            {
                "train_loss": _format_progress_metric(history["train_loss"][-1]),
                "val_loss": _format_progress_metric(history["val_loss"][-1]),
                "train_mse": _format_progress_metric(history["train_mse"][-1]),
                "val_mse": _format_progress_metric(history["val_mse"][-1]),
                "train_r2": _format_progress_metric(history["train_r2"][-1]),
                "val_r2": _format_progress_metric(history["val_r2"][-1]),
            }
        )

        monitor_value = adaptative_lr_metric_value(val_metrics, early_stopping_metric)
        if warm_up_finished and math.isfinite(monitor_value):
            monitor_updates += 1
            improved = (
                monitor_value < best_monitor_value
                if early_stopping_mode == "min"
                else monitor_value > best_monitor_value
            )
            if improved:
                best_monitor_value = monitor_value
                best_val_loss = history["val_loss"][-1]
                counter = 0
                best_model = copy.deepcopy(model.state_dict())
                has_post_warm_up_checkpoint = True
            else:
                counter += 1

        if warm_up_finished and counter > patience:
            model.load_state_dict(best_model)
            tqdm.write("Early stopping")
            break

    # If warm-up covers the run, or no finite modified monitor exists, no
    # checkpoint is eligible; in that case keep the last trained model.
    if has_post_warm_up_checkpoint:
        model.load_state_dict(best_model)

    summary = {
        "train_start_date"                  : train_loader.dataset.start_date if hasattr(train_loader.dataset, "start_date") else None,
        "train_end_date"                    : train_loader.dataset.end_date if hasattr(train_loader.dataset, "end_date") else None,
        "val_start_date"                    : val_loader.dataset.start_date if hasattr(val_loader.dataset, "start_date") else None,
        "val_end_date"                      : val_loader.dataset.end_date if hasattr(val_loader.dataset, "end_date") else None,
        "hidden_dim"                        : hidden_dim,
        "train_period"                      : window_size,
        "horizon"                           : horizon,
        "device"                            : device_local,
        "epochs_requested"                  : epochs,
        "epochs_ran"                        : len(history["epoch_time"]),
        "lr"                                : lr,
        "adj_lr_factor"                     : adj_lr_factor,
        "weight_decay"                      : weight_decay,
        "adjacency_anchor_strength"         : float(weight_decay) if has_adjacency_anchor else 0.0,
        "last_train_adjacency_anchor_loss"  : (
            history["train_adjacency_anchor_loss"][-1]
            if history["train_adjacency_anchor_loss"]
            else None
        ),
        "patience"                          : patience,
        "warm_up"                           : warm_up,
        "adaptative_lr_metric"              : requested_adaptative_lr_metric,
        "adaptative_lr_metric_effective"    : adaptative_lr_metric,
        "adaptative_lr_mode"                : adaptative_lr_mode,
        "early_stopping_metric"             : early_stopping_metric,
        "early_stopping_mode"               : early_stopping_mode,
        "best_early_stopping_metric"        : best_monitor_value if has_post_warm_up_checkpoint else None,
        "patience_monitor_updates"          : monitor_updates,
        "metric_standard"                   : metric_standard,
        "metric_threshold_mm"               : float(metric_threshold_mm) if metric_threshold_mm is not None else None,
        "metric_threshold_target_scale"     : metric_threshold if metric_standard == METRIC_STANDARD_MODIFIED else None,
        "batch_size"                        : train_loader.batch_size,
        "loss"                              : loss_name or (loss_metadata or {}).get("loss") or "mse",
        "loss_metadata"                     : loss_metadata or {},
        "best_val_loss"                     : best_val_loss if has_post_warm_up_checkpoint else None,
        "last_train_rmse"                   : math.sqrt(max(history["train_mse"][-1], 0.0)) if history["train_mse"] and math.isfinite(history["train_mse"][-1]) else None,
        "last_val_rmse"                     : math.sqrt(max(history["val_mse"][-1], 0.0)) if history["val_mse"] and math.isfinite(history["val_mse"][-1]) else None,
        "last_val_mse"                      : history["val_mse"][-1] if history["val_mse"] else None,
        "last_val_mae"                      : history["val_mae"][-1] if history["val_mae"] else None,
        "last_val_mape"                     : history["val_mape"][-1] if history["val_mape"] else None,
        "last_train_r2"                     : history["train_r2"][-1] if history["train_r2"] else None,
        "last_train_r2_batch_mean"          : history["train_r2_batch_mean"][-1] if history["train_r2_batch_mean"] else None,
        "last_train_r2_by_step"             : history["train_r2_by_step"][-1] if history["train_r2_by_step"] else None,
        "last_val_r2"                       : history["val_r2"][-1] if history["val_r2"] else None,
        "last_val_r2_batch_mean"            : history["val_r2_batch_mean"][-1] if history["val_r2_batch_mean"] else None,
        "last_val_r2_by_step"               : history["val_r2_by_step"][-1] if history["val_r2_by_step"] else None,
        "last_train_metric_target_count"    : history["train_metric_target_count"][-1] if history["train_metric_target_count"] else None,
        "last_val_metric_target_count"      : history["val_metric_target_count"][-1] if history["val_metric_target_count"] else None,
        "r2_definition"                     : (
            "global_over_targets_above_metric_threshold"
            if metric_standard == METRIC_STANDARD_MODIFIED
            else "global_over_all_targets_in_split"
        ),
        "total_time_s"                      : float(sum(history["epoch_time"])),
        "stability_patch"                   : True,
        #"cell_clip"                         : model.cell_0.cell_clip,
    }
    if learn_std:
        summary.update(
            {
                "learn_std": True,
                "std_loss_function": "mse",
                "last_train_prediction_loss": (
                    history["train_prediction_loss"][-1]
                    if history["train_prediction_loss"]
                    else None
                ),
                "last_train_std_loss": (
                    history["train_std_loss"][-1]
                    if history["train_std_loss"]
                    else None
                ),
                "last_val_prediction_loss": (
                    history["val_prediction_loss"][-1]
                    if history["val_prediction_loss"]
                    else None
                ),
                "last_val_std_loss": (
                    history["val_std_loss"][-1]
                    if history["val_std_loss"]
                    else None
                ),
            }
        )

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        save_error_plots(
            str(train_history_directory(run_dir, create=True)),
            history["train_mse"], history["val_mse"],
            history["train_mae"], history["val_mae"],
            history["train_r2"],  history["val_r2"],
            metric_standard=metric_standard,
            metric_threshold_mm=metric_threshold_mm,
        )

        logs_dir = logs_directory(run_dir, create=True)
        torch.save(history, logs_dir / "hist.pt")
        torch.save(model.state_dict(), logs_dir / "model_state_dict.pt")
        with open(logs_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return model, history, summary









def eval_with_loader(model, loader, criterion, use_amp, amp_device, amp_dtype):
    model.eval()
    total_loss, total_mse, total_mae, total_mape, total_r2 = 0.0, 0.0, 0.0, 0.0, 0.0
    steps = 0
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    loss_fn = criterion if criterion is not None else nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            assert_finite(f"eval_xb_batch{batch_idx}", xb)
            assert_finite(f"eval_yb_batch{batch_idx}", yb)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                pred = model(xb)
                assert_finite(f"eval_pred_batch{batch_idx}", pred)
                loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite eval loss on batch={batch_idx}")

            total_loss += loss.item()
            total_mse += mse_fn(pred, yb).item()
            total_mae += mae_fn(pred, yb).item()
            total_mape += safe_mape(yb, pred, eps=1e-3).item()
            total_r2 += safe_r2(yb.reshape(-1), pred.reshape(-1)).item()
            steps += 1

    return {
        "loss": total_loss / max(steps, 1),
        "mse": total_mse / max(steps, 1),
        "mae": total_mae / max(steps, 1),
        "mape": total_mape / max(steps, 1),
        "r2": total_r2 / max(steps, 1),
    }


def train_batched_only(model, train_loader, val_loader, train_period, hidden_dim, epochs, lr, weight_decay, patience, criterion=None, run_dir=None):
    model = copy.deepcopy(model).to(device)
    #print(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn = nn.MSELoss()
    mae_fn = nn.L1Loss()
    loss_fn = criterion if criterion is not None else nn.MSELoss()

    # Estável para debug: AMP desligado
    use_amp = False
    amp_device = "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"train_batched_only: use_amp={use_amp}")

    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    counter = 0

    history = {
        "train_loss": [], "train_mse": [], "train_mae": [], "train_mape": [], "train_r2": [],
        "val_loss": [], "val_mse": [], "val_mae": [], "val_mape": [], "val_r2": [], "epoch_time": []
    }

    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        train_mae = 0.0
        train_mape = 0.0
        train_r2 = 0.0
        steps = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            assert_finite(f"train_xb_epoch{ep}_batch{batch_idx}", xb)
            assert_finite(f"train_yb_epoch{ep}_batch{batch_idx}", yb)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                pred = model(xb)
                assert_finite(f"train_pred_epoch{ep}_batch{batch_idx}", pred)
                loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss epoch={ep} batch={batch_idx}")

            scaler.scale(loss).backward()
            
            

            for name, param in model.named_parameters():
                #if param.grad is not None:
                    #print(name, param.grad.abs().mean())
                if param.grad is not None and (not torch.isfinite(param.grad).all()):
                    raise RuntimeError(f"non-finite gradient in {name} epoch={ep} batch={batch_idx}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_mse += mse_fn(pred, yb).item()
            train_mae += mae_fn(pred, yb).item()
            train_mape += safe_mape(yb, pred, eps=1e-3).item()
            train_r2 += safe_r2(yb.reshape(-1), pred.reshape(-1)).item()
            steps += 1

        history["train_loss"].append(train_loss / max(steps, 1))
        history["train_mse"].append(train_mse / max(steps, 1))
        history["train_mae"].append(train_mae / max(steps, 1))
        history["train_mape"].append(train_mape / max(steps, 1))
        history["train_r2"].append(train_r2 / max(steps, 1))

        val_metrics = eval_with_loader(model, val_loader, criterion=criterion, use_amp=use_amp, amp_device=amp_device, amp_dtype=amp_dtype)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_mape"].append(val_metrics["mape"])
        history["val_r2"].append(val_metrics["r2"])

        history["epoch_time"].append(time.perf_counter() - t0)

        print(
            f"epoch={ep+1}/{epochs}" 
            f"train_loss={history['train_loss'][-1]:e}/"
            f"val_loss={history['val_loss'][-1]:e}/"
            f"train_mae={history['train_mae'][-1]:e}/"
            f"val_mae={history['val_mae'][-1]:e}/" 
            f"val_mape={history['val_mape'][-1]:e}/"
            f"val_r2={history['val_r2'][-1]:e}"
        )

        if history["val_loss"][-1] < best_val_loss:
            best_val_loss = history["val_loss"][-1]
            counter = 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            counter += 1

        if counter > patience:
            model.load_state_dict(best_model)
            print("Early Stopping")
            break

    model.load_state_dict(best_model)

    summary = {
        "hiddem_dim"        : hidden_dim,
        'train_period'      : train_period,
        "device"            : device,
        "epochs_requested"  : epochs,
        "epochs_ran"        : len(history["epoch_time"]),
        "lr"                : lr,
        "weight_decay"      : weight_decay,
        "patience"          : patience,
        "batch_size"        : train_loader.batch_size,
        "best_val_loss"     : best_val_loss,
        "last_val_mse"      : history["val_mse"][-1] if history["val_mse"] else None,
        "last_val_mae"      : history["val_mae"][-1] if history["val_mae"] else None,
        "last_val_mape"     : history["val_mape"][-1] if history["val_mape"] else None,
        "last_val_r2"       : history["val_r2"][-1] if history["val_r2"] else None,
        "total_time_s"      : float(sum(history["epoch_time"])),
    }

    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        save_error_plots(
            str(train_history_directory(run_dir, create=True)),
            history["train_mse"], history["val_mse"],
            history["train_mae"], history["val_mae"],
            history["train_r2"],  history["val_r2"],
        )

        logs_dir = logs_directory(run_dir, create=True)
        torch.save(history, logs_dir / "hist.pt")
        torch.save(model.state_dict(), logs_dir / "model_state_dict.pt")
        with open(logs_dir / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return model, history, summary
