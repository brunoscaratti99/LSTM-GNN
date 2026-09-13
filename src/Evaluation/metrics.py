import math

import numpy as np
import torch
import torch.nn as nn


METRIC_STANDARD_MODIFIED = "modified"


def normalize_metric_standard(metric_standard):
    """Return the canonical metric policy (``None`` or ``"modified"``)."""
    if metric_standard is None:
        return None
    if isinstance(metric_standard, str) and metric_standard.strip().lower() == METRIC_STANDARD_MODIFIED:
        return METRIC_STANDARD_MODIFIED
    raise ValueError("metric_standard must be None or 'modified'.")


def validate_metric_threshold(metric_threshold):
    """Validate and return a non-negative precipitation threshold in millimetres."""
    if isinstance(metric_threshold, bool):
        raise ValueError("metric_threshold must be a finite non-negative number in mm.")
    try:
        threshold = float(metric_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("metric_threshold must be a finite non-negative number in mm.") from exc
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("metric_threshold must be a finite non-negative number in mm.")
    return threshold


def metric_threshold_in_target_scale(metric_threshold, target_scaler=None):
    """Convert an mm threshold to the target scale used by the model."""
    threshold_mm = validate_metric_threshold(metric_threshold)
    if target_scaler is None:
        return threshold_mm
    transformed = np.asarray(
        target_scaler.transform(np.asarray([[threshold_mm]], dtype=float)),
        dtype=float,
    ).reshape(-1)
    if transformed.size != 1 or not np.isfinite(transformed[0]):
        raise ValueError("Could not transform metric_threshold to the model target scale.")
    return float(transformed[0])


def numpy_regression_metrics(
    y_true,
    y_pred,
    *,
    metric_standard=None,
    metric_threshold=0.0,
):
    """Compute metrics, optionally only where ``y_true > metric_threshold``.

    ``metric_threshold`` must use the same scale as ``y_true``. Physical-scale
    report callers therefore pass the configured threshold directly in mm.
    """
    metric_standard = normalize_metric_standard(metric_standard)
    threshold = validate_metric_threshold(metric_threshold)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Metric shape mismatch: actual={actual.shape}, predicted={predicted.shape}."
        )

    mask = np.isfinite(actual) & np.isfinite(predicted)
    if metric_standard == METRIC_STANDARD_MODIFIED:
        mask &= actual > threshold
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "mse": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "count": 0,
        }

    selected_actual = actual[mask]
    selected_predicted = predicted[mask]
    residual = selected_actual - selected_predicted
    squared_error_sum = float(np.sum(residual**2))
    mse = squared_error_sum / count
    absolute_error = float(np.sum(np.abs(residual))) / count
    centered = selected_actual - float(np.mean(selected_actual))
    total_sum_of_squares = float(np.sum(centered**2))
    r2 = (
        np.nan
        if total_sum_of_squares <= 1e-12
        else float(1.0 - squared_error_sum / total_sum_of_squares)
    )
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": absolute_error,
        "r2": r2,
        "bias": float(np.mean(selected_predicted - selected_actual)),
        "count": count,
    }


def _binary_roc_auc(actual_rain: np.ndarray, prediction_scores: np.ndarray) -> float:
    """Return ROC AUC from binary labels and continuous scores without sklearn.

    Average ranks are assigned to tied scores.  AUC is undefined when the
    selected data contain only one class, in which case ``NaN`` is returned.
    """
    labels = np.asarray(actual_rain, dtype=bool).reshape(-1)
    scores = np.asarray(prediction_scores, dtype=float).reshape(-1)
    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; tied values receive their average rank.
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = float(ranks[labels].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def numpy_rain_classification_metrics(
    y_true,
    y_pred,
    *,
    threshold=0.0,
):
    """Classify precipitation as rain/no-rain and calculate test metrics.

    ``threshold`` must be finite and use the same scale as the inputs. A target
    or forecast is classified as rain only when it is strictly greater than the
    threshold. The returned matrix uses the conventional layout ``[[TN, FP],
    [FN, TP]]``: rows are actual no-rain/rain and columns are predicted
    no-rain/rain. The helper permits negative thresholds so it can operate on
    standardized target tensors; runner configuration still validates the
    physical millimetre threshold as non-negative.
    """
    if isinstance(threshold, bool):
        raise ValueError("Classification threshold must be a finite number.")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Classification threshold must be a finite number.") from exc
    if not math.isfinite(threshold):
        raise ValueError("Classification threshold must be a finite number.")
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Classification shape mismatch: actual={actual.shape}, predicted={predicted.shape}."
        )

    finite = np.isfinite(actual) & np.isfinite(predicted)
    selected_actual = actual[finite]
    selected_predicted = predicted[finite]
    actual_rain = selected_actual > threshold
    predicted_rain = selected_predicted > threshold

    true_negative = int(np.count_nonzero(~actual_rain & ~predicted_rain))
    false_positive = int(np.count_nonzero(~actual_rain & predicted_rain))
    false_negative = int(np.count_nonzero(actual_rain & ~predicted_rain))
    true_positive = int(np.count_nonzero(actual_rain & predicted_rain))
    count = int(selected_actual.size)

    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = (
        float(true_positive / predicted_positive)
        if predicted_positive
        else float("nan")
    )
    recall = (
        float(true_positive / actual_positive)
        if actual_positive
        else float("nan")
    )
    accuracy = float((true_positive + true_negative) / count) if count else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "auc": _binary_roc_auc(actual_rain, selected_predicted),
        "n_classification_targets": count,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }



def safe_r2(y_true, y_pred, eps=1e-8):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + eps))


def safe_mape(y_true, y_pred, eps=1e-3):
    return (torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)).mean() * 100.0


def combined_loss(y_pred, y_true,alpha=0.5):
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    return alpha*mse(y_pred, y_true)+(1-alpha)*mae(y_pred, y_true)


def weighted_mse_loss(
    y_pred,
    y_true,
    extreme_quantile=0.9,
    extreme_weight=10.0,
    is_log=False,
    eps=1e-6
):
    """
    Weighted MSE loss that emphasizes extreme rainfall, compatible with both
    log-transformed and raw targets.

    Args:
        y_pred: torch.Tensor, model predictions
        y_true: torch.Tensor, true values
        extreme_quantile: float, quantile to define "extreme rainfall" (default 0.9)
        extreme_weight: float, weight multiplier for extreme rainfall (default 5.0)
        is_log: bool, whether y_true (and y_pred) are log-transformed (default True)
        eps: small number to prevent log(0)
    """
    # Compute threshold in original scale
    if is_log:
        y_true_orig = torch.expm1(y_true)
        y_pred_orig = torch.expm1(y_pred)
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    threshold = torch.quantile(y_true_orig, extreme_quantile)

    # Create weights: extreme rainfall gets high weight
    weights = torch.ones_like(y_true)
    weights[y_true_orig > threshold] = extreme_weight

    # Weighted MSE (compute on original scale if is_log)
    if is_log:
        # compute loss in log space (optional: can use pred vs true in log)
        loss = torch.mean(weights * (y_pred - y_true) ** 2)
    else:
        loss = torch.mean(weights * (y_pred - y_true) ** 2)

    return loss



def _init_r2_tracker(horizon, device=None, undefined_as_nan=False):
    device = device if device is not None else torch.device("cpu")
    return {
        "undefined_as_nan": bool(undefined_as_nan),
        "global": {
            "ss_res": torch.zeros((), dtype=torch.float64, device=device),
            "sum_y": torch.zeros((), dtype=torch.float64, device=device),
            "sum_y2": torch.zeros((), dtype=torch.float64, device=device),
            "count": torch.zeros((), dtype=torch.float64, device=device),
        },
        "per_step": {
            "ss_res": torch.zeros(horizon, dtype=torch.float64, device=device),
            "sum_y": torch.zeros(horizon, dtype=torch.float64, device=device),
            "sum_y2": torch.zeros(horizon, dtype=torch.float64, device=device),
            "count": torch.zeros(horizon, dtype=torch.float64, device=device),
        },
    }


def _update_r2_tracker(tracker, y_true, y_pred, metric_mask=None):
    y_true_local = y_true.detach().to(torch.float64)
    y_pred_local = y_pred.detach().to(torch.float64)

    if metric_mask is None:
        mask = torch.ones_like(y_true_local, dtype=torch.bool)
    else:
        mask = metric_mask.detach().to(device=y_true_local.device, dtype=torch.bool)
        if mask.shape != y_true_local.shape:
            raise ValueError(
                f"metric_mask shape must match targets: mask={mask.shape}, target={y_true_local.shape}."
            )
    mask = mask & torch.isfinite(y_true_local) & torch.isfinite(y_pred_local)
    safe_true = torch.where(mask, y_true_local, torch.zeros_like(y_true_local))
    safe_pred = torch.where(mask, y_pred_local, torch.zeros_like(y_pred_local))

    diff = safe_true - safe_pred
    tracker["global"]["ss_res"] += diff.square().sum()
    tracker["global"]["sum_y"] += safe_true.sum()
    tracker["global"]["sum_y2"] += safe_true.square().sum()
    tracker["global"]["count"] += mask.sum(dtype=torch.float64)

    y_true_step = safe_true.reshape(safe_true.shape[0], safe_true.shape[1], -1)
    y_pred_step = safe_pred.reshape(safe_pred.shape[0], safe_pred.shape[1], -1)
    mask_step = mask.reshape(mask.shape[0], mask.shape[1], -1)

    tracker["per_step"]["ss_res"] += (y_true_step - y_pred_step).square().sum(dim=(0, 2))
    tracker["per_step"]["sum_y"] += y_true_step.sum(dim=(0, 2))
    tracker["per_step"]["sum_y2"] += y_true_step.square().sum(dim=(0, 2))
    tracker["per_step"]["count"] += mask_step.sum(dim=(0, 2), dtype=torch.float64)


def _finalize_r2_tracker(tracker, eps=1e-8):
    global_observations = tracker["global"]["count"]
    global_count = global_observations.clamp_min(1.0)
    global_mean = tracker["global"]["sum_y"] / global_count
    global_ss_tot = tracker["global"]["sum_y2"] - global_count * (global_mean ** 2)
    global_r2 = 1.0 - (tracker["global"]["ss_res"] / (global_ss_tot + eps))

    step_observations = tracker["per_step"]["count"]
    step_count = step_observations.clamp_min(1.0)
    step_mean = tracker["per_step"]["sum_y"] / step_count
    step_ss_tot = tracker["per_step"]["sum_y2"] - step_count * step_mean.square()
    step_r2 = 1.0 - (tracker["per_step"]["ss_res"] / (step_ss_tot + eps))

    if tracker.get("undefined_as_nan", False):
        global_r2_exact = 1.0 - (tracker["global"]["ss_res"] / global_ss_tot.clamp_min(eps))
        step_r2_exact = 1.0 - (tracker["per_step"]["ss_res"] / step_ss_tot.clamp_min(eps))
        global_r2 = torch.where(
            (global_observations > 0) & (global_ss_tot > eps),
            global_r2_exact,
            torch.full_like(global_r2, float("nan")),
        )
        step_r2 = torch.where(
            (step_observations > 0) & (step_ss_tot > eps),
            step_r2_exact,
            torch.full_like(step_r2, float("nan")),
        )

    return {
        "global": float(global_r2.detach().cpu().item()),
        "per_step": [float(v) for v in step_r2.detach().cpu().tolist()],
        "count": int(global_observations.detach().cpu().item()),
        "count_per_step": [int(v) for v in step_observations.detach().cpu().tolist()],
    }


def make_weighted_mse_loss_standardized(
    y_train,
    extreme_quantile=0.90,
    extreme_weight=10.0,
    normalize_weights=True,
    eps=1e-8,
):
    """
    Weighted MSE para targets já padronizados com StandardScaler.

    Ideia:
    - define o limiar de extremo uma única vez, usando apenas y_train
    - aplica peso maior aos valores acima desse limiar
    - calcula a loss no espaço padronizado

    Args:
        y_train: targets de treino já normalizados/padronizados
        extreme_quantile: quantil que define evento extremo
        extreme_weight: peso aplicado aos extremos
        normalize_weights: se True, mantém média dos pesos ~= 1
        eps: estabilidade numérica
    """
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    z_threshold = torch.quantile(y_train.reshape(-1), extreme_quantile).detach()

    def weighted_mse_loss(y_pred, y_true):
        threshold = z_threshold.to(device=y_true.device, dtype=y_true.dtype)

        extreme_mask = y_true >= threshold
        weights = torch.where(
            extreme_mask,
            torch.full_like(y_true, extreme_weight),
            torch.ones_like(y_true),
        )

        if normalize_weights:
            weights = weights / weights.mean().clamp_min(eps)

        loss = weights * (y_pred - y_true).pow(2)
        return loss.mean()

    weighted_mse_loss.z_threshold = float(z_threshold.cpu())
    weighted_mse_loss.extreme_quantile = extreme_quantile
    weighted_mse_loss.extreme_weight = extreme_weight
    weighted_mse_loss.normalize_weights = normalize_weights

    return weighted_mse_loss
