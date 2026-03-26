import torch
import torch.nn as nn



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
    is_log=True,
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



def _init_r2_tracker(horizon):
    return {
        "global": {"ss_res": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "count": 0},
        "per_step": {
            "ss_res": torch.zeros(horizon, dtype=torch.float64),
            "sum_y": torch.zeros(horizon, dtype=torch.float64),
            "sum_y2": torch.zeros(horizon, dtype=torch.float64),
            "count": torch.zeros(horizon, dtype=torch.float64),
        },
    }


def _update_r2_tracker(tracker, y_true, y_pred):
    y_true_cpu = y_true.detach().to(torch.float64).cpu()
    y_pred_cpu = y_pred.detach().to(torch.float64).cpu()

    diff = y_true_cpu - y_pred_cpu
    tracker["global"]["ss_res"] += diff.square().sum().item()
    tracker["global"]["sum_y"] += y_true_cpu.sum().item()
    tracker["global"]["sum_y2"] += y_true_cpu.square().sum().item()
    tracker["global"]["count"] += y_true_cpu.numel()

    y_true_step = y_true_cpu.reshape(y_true_cpu.shape[0], y_true_cpu.shape[1], -1)
    y_pred_step = y_pred_cpu.reshape(y_pred_cpu.shape[0], y_pred_cpu.shape[1], -1)

    tracker["per_step"]["ss_res"] += (y_true_step - y_pred_step).square().sum(dim=(0, 2))
    tracker["per_step"]["sum_y"] += y_true_step.sum(dim=(0, 2))
    tracker["per_step"]["sum_y2"] += y_true_step.square().sum(dim=(0, 2))
    tracker["per_step"]["count"] += torch.full(
        (y_true_step.shape[1],),
        y_true_step.shape[0] * y_true_step.shape[2],
        dtype=torch.float64,
    )


def _finalize_r2_tracker(tracker, eps=1e-8):
    global_count = max(tracker["global"]["count"], 1)
    global_mean = tracker["global"]["sum_y"] / global_count
    global_ss_tot = tracker["global"]["sum_y2"] - global_count * (global_mean ** 2)
    global_r2 = 1.0 - (tracker["global"]["ss_res"] / (global_ss_tot + eps))

    step_count = tracker["per_step"]["count"].clamp_min(1.0)
    step_mean = tracker["per_step"]["sum_y"] / step_count
    step_ss_tot = tracker["per_step"]["sum_y2"] - step_count * step_mean.square()
    step_r2 = 1.0 - (tracker["per_step"]["ss_res"] / (step_ss_tot + eps))

    return {
        "global": float(global_r2),
        "per_step": [float(v) for v in step_r2.tolist()],
    }