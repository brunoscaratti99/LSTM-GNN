"""Shared helpers for the source-level experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from Models.model import GLSTM_v2, GraphTemporalTransformer_v1, NodewiseLSTM


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Configuration captured for one run."""

    start_date: str
    end_date: str
    state: str
    max_stations: int | None
    include_precipitation: bool
    include_temperature: bool
    include_specific_humidity: bool
    include_wind: bool
    include_vertical_velocity: bool
    window_size: int
    forecast_horizon: int
    train_ratio: float
    val_ratio: float
    normalize_features: bool
    feature_scaler: str
    normalize_target: bool
    target_scaler: str
    model_type: str
    k_neighbors: int
    hidden_dim: int
    lstm_layers: int
    learn_adj: bool
    lock_topology: bool
    dropout: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    adj_lr_factor: float
    max_grad_norm: float
    loss: str
    loss_quantiles: float | list[float] | tuple[float, ...] | torch.Tensor
    loss_quantile_weights: float | list[float] | tuple[float, ...] | torch.Tensor | str
    loss_quantile_max_weight: float
    random_seed: int
    plot_station_name: str | None
    use_daily_cache: bool
    warm_up: int = 0
    adaptative_lr_metric: str = "loss"
    metric_standard: str | None = None
    metric_threshold: float = 1.0
    learn_std: bool = False
    learn_self_att: bool = False
    station_similarity: str = "gaussian"
    station_similarity_sigma_km: float = 100.0
    empty_graph: bool = False


def print_status(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message)


def select_stations(stations: dict[str, list[str]], max_stations: int | None) -> dict[str, list[str]]:
    if max_stations is None:
        return stations
    if max_stations < 1:
        raise ValueError("MAX_STATIONS must be positive or None.")
    return dict(list(stations.items())[:max_stations])


def resolve_catalog_path(catalog_path: str | Path) -> Path:
    path = Path(catalog_path)
    if path.exists():
        return path

    matches = sorted(path.parent.glob(path.name))
    if matches:
        return matches[0]

    fallback_matches = sorted(path.parent.glob("Catalogo*.csv"))
    if fallback_matches:
        return fallback_matches[0]

    raise FileNotFoundError(f"Station catalog not found: {path}")


def run_directory(output_root: Path, sweep_name: str | None) -> Path:
    name = sweep_name or f"glstm_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_model(
    model_type: str,
    n_stations: int,
    n_features: int,
    edge_index: torch.Tensor,
    config: ExperimentRunConfig,
    *,
    edge_weight: torch.Tensor | None = None,
    share_glstm_adjacency: bool = True,
) -> torch.nn.Module:
    model_type = model_type.lower()
    if config.empty_graph:
        if model_type != "glstm":
            raise ValueError("EMPTY_GRAPH=True is only supported when MODEL_TYPE='glstm'.")
        if config.learn_std:
            raise ValueError("EMPTY_GRAPH=True requires LEARN_STD=False.")
        if config.learn_self_att:
            raise ValueError("EMPTY_GRAPH=True requires LEARN_SELF_ATT=False.")
        return NodewiseLSTM(
            N=n_stations,
            in_channels=n_features,
            hidden_size=config.hidden_dim,
            out_channels=config.forecast_horizon,
            lstm_layers=config.lstm_layers,
            dropout=config.dropout,
        )
    if model_type == "glstm":
        if config.learn_self_att and not config.learn_adj:
            raise ValueError("learn_self_att=True requires learn_adj=True.")
        return GLSTM_v2(
            N=n_stations,
            edge_index=edge_index,
            edge_weight=edge_weight,
            in_channels=n_features,
            hidden_size=config.hidden_dim,
            out_channels=config.forecast_horizon,
            lstm_layers=config.lstm_layers,
            learn_adj=config.learn_adj,
            learn_self_att=config.learn_self_att,
            lock_topology=config.lock_topology,
            dropout=config.dropout,
            share_adjacency=share_glstm_adjacency,
            learn_std=config.learn_std,
        )
    if model_type == "transformer":
        if config.learn_std:
            raise ValueError("LEARN_STD=True is only supported when MODEL_TYPE='glstm'.")
        if config.learn_self_att:
            raise ValueError("LEARN_SELF_ATT=True is only supported when MODEL_TYPE='glstm'.")
        return GraphTemporalTransformer_v1(
            N=n_stations,
            edge_index=edge_index,
            edge_weight=edge_weight,
            in_channels=n_features,
            hidden_size=config.hidden_dim,
            out_channels=config.forecast_horizon,
            learn_adj=config.learn_adj,
            lock_topology=config.lock_topology,
            dropout=config.dropout,
            max_days=config.window_size,
            squeeze_output=True,
        )
    raise ValueError("MODEL_TYPE must be 'glstm' or 'transformer'.")


class QuantileMSELoss(nn.Module):
    """Weighted MSE with target bins defined by train-only quantiles."""

    def __init__(self, thresholds: torch.Tensor, weights: torch.Tensor, normalize_weights: bool = True, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("thresholds", thresholds.detach().float())
        self.register_buffer("weights", weights.detach().float())
        self.normalize_weights = normalize_weights
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        thresholds = self.thresholds.to(device=y_true.device, dtype=y_true.dtype)
        weights = self.weights.to(device=y_true.device, dtype=y_true.dtype)
        bin_idx = torch.bucketize(y_true.detach(), thresholds)
        sample_weights = weights[bin_idx]
        if self.normalize_weights:
            sample_weights = sample_weights / sample_weights.mean().clamp_min(self.eps)
        return (sample_weights * (y_pred - y_true).pow(2)).mean()


def _as_float_tuple(values, name: str) -> tuple[float, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a float, list, tuple, or torch.Tensor.")
    if isinstance(values, torch.Tensor):
        raw_values = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, (float, int)):
        raw_values = [values]
    else:
        try:
            raw_values = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a float, list, tuple, or torch.Tensor.") from exc

    if not raw_values:
        raise ValueError(f"{name} must contain at least one value.")
    return tuple(float(value) for value in raw_values)


def _quantile_weights_from_train(
    y_train: torch.Tensor,
    thresholds: torch.Tensor,
    quantile_weights: float | list[float] | tuple[float, ...] | torch.Tensor | str,
    max_weight: float,
) -> torch.Tensor:
    n_bins = thresholds.numel() + 1
    if isinstance(quantile_weights, str):
        if quantile_weights.lower() != "auto":
            raise ValueError("LOSS_QUANTILE_WEIGHTS must be 'auto' or a tuple/list of bin weights.")
        bins = torch.bucketize(y_train.reshape(-1), thresholds)
        counts = torch.bincount(bins, minlength=n_bins).float().clamp_min(1.0)
        weights = counts.sum() / counts
        weights = weights / weights.mean().clamp_min(1e-8)
        return weights.clamp(max=max_weight)

    weights = torch.as_tensor(_as_float_tuple(quantile_weights, "LOSS_QUANTILE_WEIGHTS"), dtype=torch.float32)
    if weights.numel() != n_bins:
        raise ValueError(
            f"LOSS_QUANTILE_WEIGHTS must have {n_bins} values for {thresholds.numel()} quantiles. "
            f"Got {weights.numel()}."
        )
    return weights.clamp(max=max_weight)


def resolve_loss_function(
    loss: str,
    y_train: torch.Tensor,
    quantiles: float | list[float] | tuple[float, ...] | torch.Tensor = (0.5, 0.75, 0.9, 0.95),
    quantile_weights: float | list[float] | tuple[float, ...] | torch.Tensor | str = "auto",
    max_quantile_weight: float = 20.0,
) -> tuple[nn.Module, dict]:
    """Resolve the configured training loss and return traceable metadata."""
    loss_name = loss.lower().strip().replace("-", "_")
    if loss_name in {"mse", "mean_squared_error"}:
        return nn.MSELoss(), {"loss": "mse"}
    if loss_name in {"mae", "l1", "mean_absolute_error"}:
        return nn.L1Loss(), {"loss": "mae"}
    if loss_name in {"huber", "smooth_l1", "smooth_l1_loss"}:
        return nn.SmoothL1Loss(), {"loss": "huber"}
    if loss_name not in {"quantile_mse", "quantile_weighted_mse", "weighted_quantile_mse"}:
        raise ValueError("LOSS must be one of: 'mse', 'mae', 'huber', 'quantile_mse'.")

    quantiles = _as_float_tuple(quantiles, "LOSS_QUANTILES")
    if any(q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError("LOSS_QUANTILES values must be between 0 and 1.")

    y_train_flat = y_train.detach().float().reshape(-1).cpu()
    thresholds = torch.quantile(y_train_flat, torch.tensor(quantiles, dtype=torch.float32))
    thresholds, _ = torch.sort(thresholds)
    weights = _quantile_weights_from_train(
        y_train_flat,
        thresholds,
        quantile_weights=quantile_weights,
        max_weight=max_quantile_weight,
    )

    metadata = {
        "loss": "quantile_mse",
        "loss_quantiles": [float(q) for q in quantiles],
        "loss_quantile_thresholds": [float(v) for v in thresholds.tolist()],
        "loss_quantile_weights": [float(v) for v in weights.tolist()],
        "loss_quantile_weights_mode": quantile_weights if isinstance(quantile_weights, str) else "manual",
        "loss_quantile_max_weight": float(max_quantile_weight),
        "loss_weight_normalization": "batch_mean",
    }
    return QuantileMSELoss(thresholds, weights), metadata


def unpack_model_output(
    output,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize the legacy tensor and optional GLSTM standard-deviation output."""
    if isinstance(output, torch.Tensor):
        return output, None
    if not isinstance(output, tuple):
        raise TypeError("Model output must be a Tensor or a tuple of two Tensors.")
    if len(output) != 2:
        raise ValueError("Tuple model output must contain exactly forecast and standard deviation.")
    forecast, predicted_std = output
    if not isinstance(forecast, torch.Tensor) or not isinstance(predicted_std, torch.Tensor):
        raise TypeError("Tuple model output must contain exactly two Tensors.")
    return forecast, predicted_std


def collect_model_predictions(
    model: torch.nn.Module,
    data_loader,
    return_std: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    std_predictions = []
    with torch.no_grad():
        for xb, _yb in data_loader:
            forecast, predicted_std = unpack_model_output(model(xb.to(device)))
            predictions.append(forecast.detach().cpu())
            if return_std:
                if predicted_std is None:
                    raise ValueError(
                        "return_std=True requires a model that returns standard-deviation predictions."
                    )
                std_predictions.append(predicted_std.detach().cpu())

    concatenated_predictions = torch.cat(predictions, dim=0)
    if not return_std:
        return concatenated_predictions
    return concatenated_predictions, torch.cat(std_predictions, dim=0)
