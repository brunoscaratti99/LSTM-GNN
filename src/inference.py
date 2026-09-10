"""Reload a trained graph-temporal run and produce posterior predictions."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Data.temporal_dataset import (  # noqa: E402
    build_station_feature_dataset,
    chronological_split,
    load_station_catalog,
    target_standard_deviation_to_physical_scale,
)
from Evaluation.metrics import (  # noqa: E402
    METRIC_STANDARD_MODIFIED,
    normalize_metric_standard,
    numpy_regression_metrics,
    validate_metric_threshold,
)
from Graph.graph_related_utils import knn_topology  # noqa: E402
from Models.model import GLSTM_v2, GraphTemporalTransformer_v1, NodewiseLSTM  # noqa: E402
from Training.experiment_runner import (  # noqa: E402
    ExperimentRunConfig,
    build_model,
    resolve_catalog_path,
    unpack_model_output,
)


# Editable defaults. Command-line arguments override these values.
RUN_DIR: Path | None = None
MODE = "backtest"  # "backtest" or "forecast".
START_DATE: str | None = None  # Target-date lower bound in backtest mode.
END_DATE: str | None = None  # Target-date upper bound in backtest mode.
INPUT_END_DATE: str | None = None  # Last observed input date in forecast mode.
OUTPUT_DIR: Path | None = None
CATALOG_PATH: Path | None = None
DEVICE = "auto"
BATCH_SIZE: int | None = None
REBUILD_LEGACY_SCALERS = True
SHOW_PROGRESS = True
PROGRESS_TIME_CHUNK_DAYS = 30

DEFAULT_CATALOG_PATH = ROOT / "Datasets" / "dados_inmet" / "Catalogo*.csv"
INFERENCE_STATE_FILENAME = "inference_state.json"


_LEGACY_CONFIG_DEFAULTS = {
    "max_stations": None,
    "normalize_features": False,
    "feature_scaler": "standard",
    "normalize_target": False,
    "target_scaler": "standard",
    "lock_topology": True,
    "learn_self_att": False,
    "station_similarity": "ones",
    "station_similarity_sigma_km": 100.0,
    "dropout": 0.2,
    "epochs": 0,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "adaptative_lr_metric": "loss",
    "weight_decay": 0.0,
    "patience": 0,
    "adj_lr_factor": 1.0,
    "max_grad_norm": 1.0,
    "loss": "mse",
    "loss_quantiles": [0.9],
    "loss_quantile_weights": "auto",
    "loss_quantile_max_weight": 20.0,
    "random_seed": 42,
    "plot_station_name": None,
    "use_daily_cache": True,
    "warm_up": 0,
    "learn_std": False,
}


@dataclass(frozen=True)
class RestoredScaler:
    """Small NumPy implementation of a persisted Standard/MinMax scaler."""

    kind: str
    scale: np.ndarray
    offset: np.ndarray

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | None) -> RestoredScaler | None:
        if payload is None:
            return None
        kind = str(payload.get("kind", "")).lower()
        scale = np.asarray(payload.get("scale", []), dtype=np.float64).reshape(-1)
        offset_name = "mean" if kind == "standard" else "min"
        offset = np.asarray(payload.get(offset_name, []), dtype=np.float64).reshape(-1)
        if kind not in {"standard", "minmax"}:
            raise ValueError(f"Unsupported persisted scaler kind={kind!r}.")
        if (
            scale.size == 0
            or offset.size != scale.size
            or not np.all(np.isfinite(scale))
            or not np.all(np.isfinite(offset))
        ):
            raise ValueError("Persisted scaler has invalid scale/offset arrays.")
        if np.any(scale == 0):
            raise ValueError("Persisted scaler contains a zero scale.")
        return cls(kind=kind, scale=scale, offset=offset)

    def _validate(self, values) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.scale.size:
            raise ValueError(
                f"Scaler expects {self.scale.size} feature(s) on the last axis; got {array.shape}."
            )
        return array

    def transform(self, values) -> np.ndarray:
        array = self._validate(values)
        if self.kind == "standard":
            transformed = (array - self.offset) / self.scale
        else:
            transformed = array * self.scale + self.offset
        return np.asarray(transformed, dtype=np.float32)

    def inverse_transform(self, values) -> np.ndarray:
        array = self._validate(values)
        if self.kind == "standard":
            restored = array * self.scale + self.offset
        else:
            restored = (array - self.offset) / self.scale
        return np.asarray(restored, dtype=np.float64)


@dataclass(frozen=True)
class LoadedExperiment:
    """A checkpoint reconstructed with its node/feature and preprocessing contract."""

    run_dir: Path
    config: ExperimentRunConfig
    config_payload: dict[str, object]
    dataset_contract: dict[str, object]
    inference_state: dict[str, object] | None
    station_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    stations: dict[str, list[float]]
    edge_index: torch.Tensor
    model: torch.nn.Module
    device: torch.device
    feature_scaler: object | None
    target_scaler: object | None
    preprocessing_source: str


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required run artifact not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _config_from_artifacts(
    config_payload: Mapping[str, object],
    dataset_contract: Mapping[str, object],
    run_summary: Mapping[str, object] | None,
) -> ExperimentRunConfig:
    values = {field.name: config_payload[field.name] for field in fields(ExperimentRunConfig) if field.name in config_payload}
    run_summary = run_summary or {}
    aliases = {
        "forecast_horizon": "horizon",
        "window_size": "train_period",
        "learning_rate": "lr",
    }
    for destination, source in aliases.items():
        if destination not in values and source in run_summary:
            values[destination] = run_summary[source]

    scaled = dataset_contract.get("scaled", {})
    if isinstance(scaled, Mapping):
        values.setdefault("normalize_features", bool(scaled.get("features", False)))
        values.setdefault("normalize_target", bool(scaled.get("target", False)))
    for name, default in _LEGACY_CONFIG_DEFAULTS.items():
        values.setdefault(name, default)

    missing = []
    for field in fields(ExperimentRunConfig):
        if field.name in values:
            continue
        if field.default is not MISSING:
            values[field.name] = field.default
        else:
            missing.append(field.name)
    if missing:
        raise ValueError(
            "config.json is too old or incomplete to reconstruct this run. Missing fields: "
            + ", ".join(missing)
        )
    return ExperimentRunConfig(**values)


def _contract_names(dataset_contract: Mapping[str, object], key: str) -> tuple[str, ...]:
    raw_values = dataset_contract.get(key)
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"dataset_contract.json must contain a non-empty {key!r} list.")
    names = tuple(str(value) for value in raw_values)
    if len(set(names)) != len(names):
        raise ValueError(f"dataset_contract.json contains duplicate {key}.")
    return names


def _resolve_device(device: str | torch.device) -> torch.device:
    requested = str(device)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable.")
    return resolved


def _torch_load_weights(checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument.
        payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, Mapping) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state_dict mapping.")
    state_dict = payload
    if all(str(key).startswith("module.") for key in state_dict):
        stripped = OrderedDict((str(key)[7:], value) for key, value in state_dict.items())
        metadata = getattr(state_dict, "_metadata", None)
        if isinstance(metadata, Mapping):
            stripped._metadata = {
                (str(key)[7:] if str(key).startswith("module.") else str(key)): value
                for key, value in metadata.items()
            }
        state_dict = stripped
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise ValueError(f"Checkpoint {checkpoint_path} contains non-tensor state entries.")
    for key, value in state_dict.items():
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor {key!r} contains NaN or infinite values.")
    return state_dict


def _edge_index_from_checkpoint(state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
    for key in ("cell_0.edge_mask", "adj.edge_mask"):
        edge_mask = state_dict.get(key)
        if edge_mask is None:
            continue
        if edge_mask.ndim != 2 or edge_mask.shape[0] != edge_mask.shape[1]:
            raise ValueError(f"Checkpoint topology buffer {key} is not square.")
        return (edge_mask > 0).nonzero(as_tuple=False).T.contiguous().long()
    return None


def _checkpoint_requires_per_layer_adjacency(
    state_dict: Mapping[str, torch.Tensor],
) -> bool:
    """Return whether a legacy GLSTM checkpoint stores distinct layer graphs."""
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "cells" or parts[2] != "a_logits":
            continue
        reference = state_dict.get("cell_0.a_logits")
        if reference is None or reference.shape != value.shape or not torch.equal(reference, value):
            return True
    return False


def _validate_checkpoint_layer_adjacency_buffers(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Reject immutable per-layer graph state that disagrees with cell_0."""
    buffer_names = {
        "I",
        "edge_mask",
        "topology_mask",
        "edge_weight_prior",
        "uses_edge_weight_prior",
        "a_logits_init",
        "A_fixed_raw",
        "A_fixed",
    }
    for key, value in state_dict.items():
        parts = key.split(".")
        if len(parts) != 3 or parts[0] != "cells" or parts[2] not in buffer_names:
            continue
        reference = state_dict.get(f"cell_0.{parts[2]}")
        if reference is None or reference.shape != value.shape or not torch.equal(reference, value):
            raise ValueError(
                "Checkpoint contains inconsistent immutable GLSTM adjacency state: "
                f"{key!r} disagrees with 'cell_0.{parts[2]}'."
            )


def _catalog_path_from_parent_manifest(run_dir: Path) -> Path | None:
    manifest_path = run_dir.parent / "comparative_summary.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        return None
    for record in runs:
        if not isinstance(record, Mapping):
            continue
        record_dir = Path(str(record.get("run_dir", record.get("run_name", "")))).name
        if record_dir != run_dir.name:
            continue
        parameters = record.get("parameters", {})
        if isinstance(parameters, Mapping) and parameters.get("catalog_path"):
            return Path(str(parameters["catalog_path"]))
    return None


def _stations_from_catalog(
    run_dir: Path,
    station_names: tuple[str, ...],
    state: str,
    catalog_path: Path | str | None,
) -> dict[str, list[float]]:
    if catalog_path is not None:
        candidates = [Path(catalog_path)]
    else:
        candidates = [
            candidate
            for candidate in (_catalog_path_from_parent_manifest(run_dir), DEFAULT_CATALOG_PATH)
            if candidate is not None
        ]
    catalog = None
    errors = []
    for candidate in candidates:
        try:
            catalog = load_station_catalog(resolve_catalog_path(candidate), state=state)
            break
        except FileNotFoundError as exc:
            errors.append(str(exc))
    if catalog is None:
        raise FileNotFoundError("Could not resolve a station catalog. " + " | ".join(errors))
    missing = [name for name in station_names if name not in catalog]
    if missing:
        raise ValueError(
            "The station catalog cannot reproduce the trained node order. Missing: "
            + ", ".join(missing)
        )
    return {
        name: [float(catalog[name][0]), float(catalog[name][1])]
        for name in station_names
    }


def _stations_from_inference_state(
    inference_state: Mapping[str, object],
    station_names: tuple[str, ...],
    feature_names: tuple[str, ...],
) -> dict[str, list[float]]:
    saved_stations = tuple(str(value) for value in inference_state.get("stations", []))
    saved_features = tuple(str(value) for value in inference_state.get("features", []))
    if saved_stations != station_names or saved_features != feature_names:
        raise ValueError("inference_state.json disagrees with dataset_contract.json node/feature order.")
    raw_coordinates = inference_state.get("station_coordinates")
    if not isinstance(raw_coordinates, Mapping):
        raise ValueError("inference_state.json lacks station_coordinates.")
    stations = {}
    for name in station_names:
        coordinates = raw_coordinates.get(name)
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            raise ValueError(f"Invalid persisted coordinates for station {name!r}.")
        stations[name] = [float(coordinates[0]), float(coordinates[1])]
    return stations


def _edge_index_from_inference_state(inference_state: Mapping[str, object]) -> torch.Tensor:
    edge_index = torch.as_tensor(inference_state.get("edge_index"), dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("inference_state.json edge_index must have shape [2, E].")
    return edge_index.contiguous()


def _load_model_compatibly(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allow_legacy_topology_mask: bool,
) -> None:
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint tensors are incompatible with the reconstructed architecture: {exc}") from exc
    disallowed_missing = list(incompatible.missing_keys)
    if allow_legacy_topology_mask:
        disallowed_missing = [
            key for key in disallowed_missing if not key.endswith(".topology_mask")
        ]
    legacy_prior_buffers = {
        "edge_weight_prior",
        "uses_edge_weight_prior",
        "A_fixed_raw",
    }
    checkpoint_has_prior_buffers = any(
        key.split(".")[-1] in legacy_prior_buffers for key in state_dict
    )
    if not checkpoint_has_prior_buffers:
        disallowed_missing = [
            key
            for key in disallowed_missing
            if key.split(".")[-1] not in legacy_prior_buffers
        ]
    if disallowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint keys are incompatible with the reconstructed architecture. "
            f"Missing={disallowed_missing}; unexpected={list(incompatible.unexpected_keys)}"
        )


def _build_reconstructed_model(
    config: ExperimentRunConfig,
    n_stations: int,
    n_features: int,
    edge_index: torch.Tensor,
    inference_state: Mapping[str, object] | None,
    state_dict: Mapping[str, torch.Tensor] | None = None,
) -> torch.nn.Module:
    model_build = inference_state.get("model_build") if inference_state is not None else None
    if model_build is None:
        if config.model_type.lower() == "glstm" and state_dict is not None:
            _validate_checkpoint_layer_adjacency_buffers(state_dict)
        share_glstm_adjacency = not (
            config.model_type.lower() == "glstm"
            and state_dict is not None
            and _checkpoint_requires_per_layer_adjacency(state_dict)
        )
        return build_model(
            config.model_type,
            n_stations=n_stations,
            n_features=n_features,
            edge_index=edge_index,
            config=config,
            share_glstm_adjacency=share_glstm_adjacency,
        )
    if not isinstance(model_build, Mapping) or not isinstance(model_build.get("kwargs"), Mapping):
        raise ValueError("inference_state.json contains an invalid model_build contract.")
    model_class = str(model_build.get("model_class", ""))
    expected_model_class = (
        "NodewiseLSTM"
        if config.empty_graph
        else {
            "glstm": "GLSTM_v2",
            "transformer": "GraphTemporalTransformer_v1",
        }.get(config.model_type.lower())
    )
    if model_class != expected_model_class:
        raise ValueError(
            f"Persisted model class {model_class!r} disagrees with config model_type={config.model_type!r}."
        )
    kwargs = dict(model_build["kwargs"])
    if model_class == "GLSTM_v2":
        # Historical schema-v2 artifacts predate these optional GLSTM features.
        kwargs.setdefault("learn_std", False)
        kwargs.setdefault("learn_self_att", False)
    if int(kwargs.get("N", -1)) != n_stations or int(kwargs.get("in_channels", -1)) != n_features:
        raise ValueError("Persisted model dimensions disagree with dataset_contract.json.")

    if model_class == "NodewiseLSTM":
        common_config_checks = {
            "hidden_size": config.hidden_dim,
            "out_channels": config.forecast_horizon,
            "lstm_layers": config.lstm_layers,
            "dropout": config.dropout,
        }
    else:
        common_config_checks = {
            "hidden_size": config.hidden_dim,
            "out_channels": config.forecast_horizon,
            "learn_adj": config.learn_adj,
            "lock_topology": config.lock_topology,
            "dropout": config.dropout,
        }
    if model_class == "NodewiseLSTM":
        allowed = {
            "N",
            "in_channels",
            "hidden_size",
            "out_channels",
            "lstm_layers",
            "dropout",
        }
        if set(kwargs) != allowed:
            raise ValueError("Persisted NodewiseLSTM constructor options are incomplete or inconsistent.")
        return NodewiseLSTM(**kwargs)
    if model_class == "GLSTM_v2":
        common_config_checks["learn_std"] = bool(getattr(config, "learn_std", False))
        common_config_checks["learn_self_att"] = bool(
            getattr(config, "learn_self_att", False)
        )
    for name, expected in common_config_checks.items():
        if kwargs.get(name) != expected:
            raise ValueError(
                f"Persisted model option {name}={kwargs.get(name)!r} disagrees with config.json value {expected!r}."
            )

    if model_class == "GLSTM_v2":
        allowed = {
            "N",
            "in_channels",
            "hidden_size",
            "out_channels",
            "lstm_layers",
            "aggr",
            "learn_adj",
            "learn_self_att",
            "lock_topology",
            "dropout",
            "cell_clip",
            "learn_std",
        }
        adjacency_scope = model_build.get("adjacency_scope")
        if adjacency_scope is None:
            adjacency_scope = "per_layer"
        if adjacency_scope not in {"shared", "per_layer"}:
            raise ValueError("Persisted GLSTM adjacency_scope must be 'shared' or 'per_layer'.")
        if state_dict is not None:
            _validate_checkpoint_layer_adjacency_buffers(state_dict)
            if (
                adjacency_scope == "shared"
                and _checkpoint_requires_per_layer_adjacency(state_dict)
            ):
                raise ValueError(
                    "Checkpoint declares a shared GLSTM adjacency but stores distinct layer logits."
                )
        if set(kwargs) != allowed or int(kwargs["lstm_layers"]) != config.lstm_layers:
            raise ValueError("Persisted GLSTM constructor options are incomplete or inconsistent.")
        kwargs["share_adjacency"] = adjacency_scope == "shared"
        return GLSTM_v2(edge_index=edge_index, **kwargs)
    if model_class == "GraphTemporalTransformer_v1":
        allowed = {
            "N",
            "in_channels",
            "hidden_size",
            "out_channels",
            "learn_adj",
            "lock_topology",
            "target_dim",
            "nhead",
            "num_day_layers",
            "num_window_layers",
            "num_decoder_layers",
            "dim_feedforward",
            "dropout",
            "max_window",
            "max_days",
            "squeeze_output",
            "four_dim_mode",
            "precip_col",
            "output_activation",
            "use_node_embeddings",
            "use_precip_residual",
        }
        if set(kwargs) != allowed:
            raise ValueError("Persisted Transformer constructor options are incomplete or inconsistent.")
        return GraphTemporalTransformer_v1(edge_index=edge_index, **kwargs)
    raise ValueError(f"Unsupported persisted model class {model_class!r}.")


def _scalers_from_state(
    inference_state: Mapping[str, object] | None,
) -> tuple[object | None, object | None]:
    if inference_state is None:
        return None, None
    feature_payload = inference_state.get("feature_scaler")
    target_payload = inference_state.get("target_scaler")
    if feature_payload is not None and not isinstance(feature_payload, Mapping):
        raise ValueError("Invalid feature_scaler in inference_state.json.")
    if target_payload is not None and not isinstance(target_payload, Mapping):
        raise ValueError("Invalid target_scaler in inference_state.json.")
    return RestoredScaler.from_payload(feature_payload), RestoredScaler.from_payload(target_payload)


def _validate_scaler_contract(
    config: ExperimentRunConfig,
    feature_scaler: object | None,
    target_scaler: object | None,
    n_features: int,
) -> None:
    if not config.normalize_features and feature_scaler is not None:
        raise ValueError("inference_state.json has a feature scaler, but config disables feature normalization.")
    if not config.normalize_target and target_scaler is not None:
        raise ValueError("inference_state.json has a target scaler, but config disables target normalization.")
    if isinstance(feature_scaler, RestoredScaler) and feature_scaler.scale.size != n_features:
        raise ValueError(
            f"Persisted feature scaler has {feature_scaler.scale.size} entries; expected {n_features}."
        )
    if isinstance(feature_scaler, RestoredScaler) and feature_scaler.kind != config.feature_scaler.lower():
        raise ValueError(
            f"Persisted feature scaler kind={feature_scaler.kind!r} disagrees with config "
            f"feature_scaler={config.feature_scaler!r}."
        )
    if isinstance(target_scaler, RestoredScaler) and target_scaler.scale.size != 1:
        raise ValueError("Persisted target scaler must contain exactly one scale entry.")
    if isinstance(target_scaler, RestoredScaler) and target_scaler.kind != config.target_scaler.lower():
        raise ValueError(
            f"Persisted target scaler kind={target_scaler.kind!r} disagrees with config "
            f"target_scaler={config.target_scaler!r}."
        )


def _validate_saved_topology(
    edge_index: torch.Tensor,
    state_dict: Mapping[str, torch.Tensor],
    n_stations: int,
) -> None:
    checkpoint_edges = _edge_index_from_checkpoint(state_dict)
    if checkpoint_edges is None:
        return
    saved_mask = torch.zeros((n_stations, n_stations), dtype=torch.bool)
    checkpoint_mask = torch.zeros_like(saved_mask)
    if edge_index.numel():
        saved_mask[edge_index[0], edge_index[1]] = True
    if checkpoint_edges.numel():
        checkpoint_mask[checkpoint_edges[0], checkpoint_edges[1]] = True
    if not torch.equal(saved_mask, checkpoint_mask):
        raise ValueError("Persisted edge_index disagrees with the base topology stored in the checkpoint.")


def _validate_daily_axis(data, label: str) -> pd.DatetimeIndex:
    times = pd.DatetimeIndex(pd.to_datetime(data.time.values, errors="raise"))
    if times.empty:
        raise ValueError(f"{label} contains no dates.")
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError(f"{label} dates must be unique and sorted.")
    if len(times) > 1 and not np.all(np.diff(times.values) == np.timedelta64(1, "D")):
        raise ValueError(f"{label} must have one consecutive observation per day.")
    return times


def _align_data_to_contract(X, y, loaded: LoadedExperiment):
    station_names = [str(value) for value in X.station.values]
    feature_names = [str(value) for value in X.feature.values]
    target_stations = [str(value) for value in y.station.values]
    if len(station_names) != len(set(station_names)) or len(feature_names) != len(set(feature_names)):
        raise ValueError("Inference data contains duplicate station or feature coordinates.")
    if set(station_names) != set(loaded.station_names):
        missing = sorted(set(loaded.station_names) - set(station_names))
        extra = sorted(set(station_names) - set(loaded.station_names))
        raise ValueError(f"Inference station set differs from training. Missing={missing}; extra={extra}")
    if set(target_stations) != set(loaded.station_names):
        raise ValueError("Inference target station set differs from the trained station contract.")
    if set(feature_names) != set(loaded.feature_names):
        missing = sorted(set(loaded.feature_names) - set(feature_names))
        extra = sorted(set(feature_names) - set(loaded.feature_names))
        raise ValueError(f"Inference feature set differs from training. Missing={missing}; extra={extra}")
    X = X.sel(station=list(loaded.station_names), feature=list(loaded.feature_names))
    y = y.sel(station=list(loaded.station_names))
    _validate_daily_axis(X, "Inference features")
    _validate_daily_axis(y, "Inference targets")
    return X, y


def _build_data(
    loaded: LoadedExperiment,
    start_date: str,
    end_date: str,
    *,
    show_progress: bool,
    progress_time_chunk_days: int,
):
    X, y = build_station_feature_dataset(
        stations=loaded.stations,
        start_date=start_date,
        end_date=end_date,
        include_precipitation=loaded.config.include_precipitation,
        include_temperature=loaded.config.include_temperature,
        include_specific_humidity=loaded.config.include_specific_humidity,
        include_wind=loaded.config.include_wind,
        include_vertical_velocity=loaded.config.include_vertical_velocity,
        show_progress=show_progress,
        progress_time_chunk_days=progress_time_chunk_days,
        prefer_daily_cache=loaded.config.use_daily_cache,
    )
    return _align_data_to_contract(X, y, loaded)


def _validate_original_dataset_contract(X, loaded: LoadedExperiment) -> None:
    """Detect data drift before reproducing an old split or fitted scaler."""
    contract = loaded.dataset_contract
    times = _validate_daily_axis(X, "Original run data")
    expected_start = contract.get("raw_time_start")
    expected_end = contract.get("raw_time_end")
    if expected_start is not None and times[0].date().isoformat() != str(expected_start):
        raise ValueError(
            f"Original dataset now starts at {times[0].date()}, but the run recorded {expected_start}."
        )
    if expected_end is not None and times[-1].date().isoformat() != str(expected_end):
        raise ValueError(
            f"Original dataset now ends at {times[-1].date()}, but the run recorded {expected_end}."
        )
    raw_dims = contract.get("raw_X_dims")
    if isinstance(raw_dims, Mapping) and "time" in raw_dims:
        expected_length = int(raw_dims["time"])
        if len(times) != expected_length:
            raise ValueError(
                f"Original dataset now has {len(times)} dates, but the run recorded {expected_length}."
            )


def _window_membership_counts(n_time: int, window_size: int, horizon: int, target: bool) -> np.ndarray:
    if n_time < window_size + horizon:
        raise ValueError(
            f"Training split has {n_time} dates, shorter than window+horizon={window_size + horizon}."
        )
    differences = np.zeros(n_time + 1, dtype=np.int64)
    for target_index in range(window_size, n_time - horizon + 1):
        start = target_index if target else target_index - window_size
        stop = target_index + horizon if target else target_index
        differences[start] += 1
        differences[stop] -= 1
    return np.cumsum(differences[:-1])


def _fit_weighted_scaler(values: np.ndarray, counts: np.ndarray, scaler_name: str):
    values = np.asarray(values, dtype=np.float32)
    selected = counts > 0
    if not np.any(selected):
        raise ValueError("No training observations participate in a model window.")
    flat = values[selected].reshape(-1, values.shape[-1])
    scaler_name = scaler_name.lower()
    if scaler_name == "standard":
        weights = np.repeat(counts[selected], int(np.prod(values.shape[1:-1], dtype=int)))
        scaler = StandardScaler()
        scaler.fit(flat, sample_weight=weights)
        return scaler
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(flat)
        return scaler
    raise ValueError(f"Unsupported scaler {scaler_name!r}; expected 'standard' or 'minmax'.")


def _rebuild_legacy_scalers(
    loaded: LoadedExperiment,
    *,
    show_progress: bool,
    progress_time_chunk_days: int,
) -> tuple[object | None, object | None, bool]:
    X, y = _build_data(
        loaded,
        loaded.config.start_date,
        loaded.config.end_date,
        show_progress=show_progress,
        progress_time_chunk_days=progress_time_chunk_days,
    )
    _validate_original_dataset_contract(X, loaded)
    splits = chronological_split(
        X,
        y,
        train_ratio=loaded.config.train_ratio,
        val_ratio=loaded.config.val_ratio,
    )
    feature_scaler = None
    target_scaler = None
    if loaded.config.normalize_features:
        counts = _window_membership_counts(
            splits.train_X.sizes["time"],
            loaded.config.window_size,
            loaded.config.forecast_horizon,
            target=False,
        )
        feature_scaler = _fit_weighted_scaler(
            np.asarray(splits.train_X.values, dtype=np.float32),
            counts,
            loaded.config.feature_scaler,
        )
    if loaded.config.normalize_target:
        counts = _window_membership_counts(
            splits.train_y.sizes["time"],
            loaded.config.window_size,
            loaded.config.forecast_horizon,
            target=True,
        )
        target_values = np.asarray(splits.train_y.values, dtype=np.float32)[..., None]
        target_scaler = _fit_weighted_scaler(
            target_values,
            counts,
            loaded.config.target_scaler,
        )
    verified = _verify_legacy_reconstruction(
        loaded,
        splits,
        feature_scaler,
        target_scaler,
    )
    return feature_scaler, target_scaler, verified


def _verify_legacy_reconstruction(
    loaded: LoadedExperiment,
    splits,
    feature_scaler,
    target_scaler,
) -> bool:
    """Compare one reconstructed test prediction with the run's saved CSV."""
    prediction_path = loaded.run_dir / "test_predictions_by_lead_day.csv"
    if not prediction_path.exists():
        warnings.warn(
            "Legacy scaler state was reconstructed but could not be verified because "
            "test_predictions_by_lead_day.csv is absent.",
            stacklevel=2,
        )
        return False

    window_size = loaded.config.window_size
    horizon = loaded.config.forecast_horizon
    n_stations = len(loaded.station_names)
    if splits.test_X.sizes["time"] < window_size + horizon:
        raise ValueError("Original test split is too short to verify legacy preprocessing.")
    raw_window = np.asarray(splits.test_X.isel(time=slice(0, window_size)).values, dtype=np.float32)
    if feature_scaler is not None:
        flat = feature_scaler.transform(raw_window.reshape(-1, raw_window.shape[-1]))
        raw_window = np.asarray(flat, dtype=np.float32).reshape(raw_window.shape)
    if not np.all(np.isfinite(raw_window)):
        raise ValueError("Reconstructed legacy verification input contains NaN or infinite values.")
    inputs = torch.as_tensor(raw_window[None, ...], dtype=torch.float32, device=loaded.device)
    with torch.inference_mode():
        predicted = loaded.model(inputs).detach().cpu().numpy()
    expected_shape = (1, horizon, n_stations)
    if predicted.shape != expected_shape or not np.all(np.isfinite(predicted)):
        raise ValueError(
            f"Legacy verification forward returned invalid shape/values: {predicted.shape}."
        )
    if target_scaler is not None:
        predicted = target_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(predicted.shape)

    n_rows = horizon * n_stations
    saved = pd.read_csv(prediction_path, nrows=n_rows)
    actual_column = "actual_mm" if "actual_mm" in saved.columns else "actual"
    predicted_column = "predicted_mm" if "predicted_mm" in saved.columns else "predicted"
    required = {"sample", "lead_day", "target_time", "station", actual_column, predicted_column}
    if len(saved) != n_rows or not required.issubset(saved.columns):
        raise ValueError("Saved test prediction CSV cannot verify the reconstructed legacy pipeline.")
    if saved["sample"].astype(int).ne(0).any():
        raise ValueError("Saved test prediction CSV is not ordered with sample 0 first.")
    expected_stations = list(loaded.station_names) * horizon
    if saved["station"].astype(str).tolist() != expected_stations:
        raise ValueError("Saved test prediction station order differs from dataset_contract.json.")
    expected_leads = np.repeat(np.arange(1, horizon + 1), n_stations)
    if not np.array_equal(saved["lead_day"].to_numpy(dtype=int), expected_leads):
        raise ValueError("Saved test prediction lead-day order is invalid.")

    actual = np.asarray(splits.test_y.isel(time=slice(window_size, window_size + horizon)).values)
    saved_actual = saved[actual_column].to_numpy(dtype=float).reshape(horizon, n_stations)
    saved_predicted = saved[predicted_column].to_numpy(dtype=float).reshape(horizon, n_stations)
    target_dates = pd.DatetimeIndex(
        splits.test_y.time.values[window_size : window_size + horizon]
    ).strftime("%Y-%m-%d")
    saved_dates = saved.groupby("lead_day", sort=True)["target_time"].first().astype(str).tolist()
    if saved_dates != target_dates.tolist() or not np.allclose(saved_actual, actual, rtol=1e-6, atol=1e-6):
        raise ValueError(
            "Original meteorological values/dates no longer reproduce the run's saved test sample."
        )
    if not np.allclose(saved_predicted, predicted[0], rtol=1e-4, atol=1e-3):
        max_error = float(np.max(np.abs(saved_predicted - predicted[0])))
        raise ValueError(
            "Reconstructed legacy preprocessing/model no longer reproduces the saved prediction "
            f"(maximum absolute difference={max_error:.6g})."
        )
    return True


def load_trained_experiment(
    run_dir: Path | str,
    *,
    catalog_path: Path | str | None = None,
    device: str | torch.device = "auto",
    rebuild_legacy_scalers: bool = True,
    show_progress: bool = False,
    progress_time_chunk_days: int = PROGRESS_TIME_CHUNK_DAYS,
) -> LoadedExperiment:
    """Reconstruct a trained run from its config, contract, checkpoint, and scaler state."""
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    contract_path = run_dir / "dataset_contract.json"
    checkpoint_path = run_dir / "model_state_dict.pt"
    config_payload = _read_json(config_path)
    dataset_contract = _read_json(contract_path)
    run_summary = _read_json(run_dir / "run_summary.json") if (run_dir / "run_summary.json").exists() else None
    config = _config_from_artifacts(config_payload, dataset_contract, run_summary)
    station_names = _contract_names(dataset_contract, "stations")
    feature_names = _contract_names(dataset_contract, "features")

    state_path = run_dir / INFERENCE_STATE_FILENAME
    inference_state = _read_json(state_path) if state_path.exists() else None
    if inference_state is not None:
        schema_version = int(inference_state.get("schema_version", -1))
        if schema_version not in {1, 2}:
            raise ValueError(f"Unsupported inference-state schema in {state_path}.")
        if schema_version >= 2 and "model_build" not in inference_state:
            raise ValueError("Schema-v2 inference_state.json lacks model_build.")
        stations = _stations_from_inference_state(inference_state, station_names, feature_names)
    else:
        stations = _stations_from_catalog(run_dir, station_names, config.state, catalog_path)

    checkpoint = _torch_load_weights(checkpoint_path)
    if inference_state is not None:
        edge_index = _edge_index_from_inference_state(inference_state)
    else:
        edge_index = _edge_index_from_checkpoint(checkpoint)
        if edge_index is None:
            edge_index = (
                torch.empty((2, 0), dtype=torch.long)
                if config.empty_graph
                else knn_topology(stations, k=config.k_neighbors)[0]
            )
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= len(station_names)
    ):
        raise ValueError("Saved edge_index contains a node outside the dataset station contract.")
    if inference_state is not None:
        _validate_saved_topology(edge_index, checkpoint, len(station_names))

    model = _build_reconstructed_model(
        config,
        n_stations=len(station_names),
        n_features=len(feature_names),
        edge_index=edge_index,
        inference_state=inference_state,
        state_dict=checkpoint,
    )
    _load_model_compatibly(
        model,
        checkpoint,
        allow_legacy_topology_mask=inference_state is None,
    )
    resolved_device = _resolve_device(device)
    model = model.to(resolved_device)
    model.eval()

    feature_scaler, target_scaler = _scalers_from_state(inference_state)
    _validate_scaler_contract(config, feature_scaler, target_scaler, len(feature_names))
    preprocessing_source = INFERENCE_STATE_FILENAME if inference_state is not None else "not_saved"
    needs_feature_scaler = config.normalize_features and feature_scaler is None
    needs_target_scaler = config.normalize_target and target_scaler is None
    provisional = LoadedExperiment(
        run_dir=run_dir,
        config=config,
        config_payload=dict(config_payload),
        dataset_contract=dict(dataset_contract),
        inference_state=dict(inference_state) if inference_state is not None else None,
        station_names=station_names,
        feature_names=feature_names,
        stations=stations,
        edge_index=edge_index.detach().cpu(),
        model=model,
        device=resolved_device,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        preprocessing_source=preprocessing_source,
    )
    if needs_feature_scaler or needs_target_scaler:
        if not rebuild_legacy_scalers:
            missing = []
            if needs_feature_scaler:
                missing.append("feature scaler")
            if needs_target_scaler:
                missing.append("target scaler")
            raise RuntimeError(
                f"Legacy run lacks its fitted {' and '.join(missing)}. "
                "Enable rebuild_legacy_scalers to reconstruct them from the original training data."
            )
        feature_scaler, target_scaler, verified = _rebuild_legacy_scalers(
            provisional,
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )
        preprocessing_source = (
            "reconstructed_and_verified_against_saved_test_prediction"
            if verified
            else "reconstructed_from_original_training_data_unverified"
        )
    return LoadedExperiment(
        **{
            **provisional.__dict__,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "preprocessing_source": preprocessing_source,
        }
    )


def _scaled_features(X, feature_scaler) -> np.ndarray:
    values = np.asarray(X.values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Inference features contain NaN or infinite values.")
    if feature_scaler is None:
        return values
    flat = feature_scaler.transform(values.reshape(-1, values.shape[-1]))
    transformed = np.asarray(flat, dtype=np.float32).reshape(values.shape)
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Scaled inference features contain NaN or infinite values.")
    return transformed


def _physical_predictions(predictions: np.ndarray, target_scaler) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    if target_scaler is None:
        restored = predictions.astype(np.float64)
    else:
        flat = target_scaler.inverse_transform(predictions.reshape(-1, 1))
        restored = np.asarray(flat, dtype=np.float64).reshape(predictions.shape)
    if not np.all(np.isfinite(restored)):
        raise ValueError("Physical-scale predictions contain NaN or infinite values.")
    return restored


def _predict_windows(
    loaded: LoadedExperiment,
    feature_values: np.ndarray,
    target_values: np.ndarray | None,
    times: pd.DatetimeIndex,
    target_indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None, pd.DatetimeIndex, np.ndarray]:
    window_size = loaded.config.window_size
    horizon = loaded.config.forecast_horizon
    predicted_batches = []
    actual_batches = []
    input_end_dates = []
    target_date_batches = []
    for batch_start in range(0, len(target_indices), batch_size):
        indices = target_indices[batch_start : batch_start + batch_size]
        windows = np.stack(
            [feature_values[index - window_size : index] for index in indices],
            axis=0,
        )
        inputs = torch.as_tensor(windows, dtype=torch.float32, device=loaded.device)
        with torch.inference_mode():
            predictions = loaded.model(inputs)
        predictions = predictions.detach().cpu().numpy()
        expected_shape = (len(indices), horizon, len(loaded.station_names))
        if predictions.shape != expected_shape:
            raise ValueError(
                f"Model output shape {predictions.shape} differs from expected {expected_shape}."
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Model produced NaN or infinite predictions.")
        predicted_batches.append(predictions)
        if target_values is not None:
            actual_batch = np.stack(
                [target_values[index : index + horizon] for index in indices], axis=0
            )
            if not np.all(np.isfinite(actual_batch)):
                raise ValueError("Backtest targets contain NaN or infinite values.")
            actual_batches.append(actual_batch)
        input_end_dates.extend(times[indices - 1])
        target_date_batches.extend([times[index : index + horizon].values for index in indices])

    predicted = _physical_predictions(np.concatenate(predicted_batches, axis=0), loaded.target_scaler)
    actual = np.concatenate(actual_batches, axis=0).astype(np.float64) if actual_batches else None
    target_times = np.asarray(target_date_batches, dtype="datetime64[ns]")
    return predicted, actual, pd.DatetimeIndex(input_end_dates), target_times


def _candidate_target_indices(
    times: pd.DatetimeIndex,
    window_size: int,
    horizon: int,
    target_start: pd.Timestamp | None,
    target_end: pd.Timestamp | None,
) -> np.ndarray:
    indices = np.arange(window_size, len(times) - horizon + 1, dtype=np.int64)
    if target_start is not None:
        indices = indices[times[indices] >= target_start]
    if target_end is not None:
        indices = indices[times[indices] <= target_end]
    if indices.size == 0:
        raise ValueError(
            "No complete inference windows fit the requested interval. "
            "The interval must include input context plus the complete forecast horizon."
        )
    return indices


def _run_backtest(
    loaded: LoadedExperiment,
    start_date: str | None,
    end_date: str | None,
    batch_size: int,
    *,
    show_progress: bool,
    progress_time_chunk_days: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, dict[str, object]]:
    if (start_date is None) != (end_date is None):
        raise ValueError("Backtest mode requires both start_date and end_date, or neither.")
    if start_date is None:
        X, y = _build_data(
            loaded,
            loaded.config.start_date,
            loaded.config.end_date,
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )
        _validate_original_dataset_contract(X, loaded)
        splits = chronological_split(
            X,
            y,
            train_ratio=loaded.config.train_ratio,
            val_ratio=loaded.config.val_ratio,
        )
        X, y = splits.test_X, splits.test_y
        target_start = None
        target_end = None
        selection = "original_test_split"
    else:
        target_start = pd.Timestamp(start_date).normalize()
        target_end = pd.Timestamp(end_date).normalize()
        if target_start > target_end:
            raise ValueError("start_date must not be after end_date.")
        context_start = target_start - pd.Timedelta(days=loaded.config.window_size)
        data_end = target_end + pd.Timedelta(days=loaded.config.forecast_horizon - 1)
        X, y = _build_data(
            loaded,
            context_start.date().isoformat(),
            data_end.date().isoformat(),
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )
        times = _validate_daily_axis(X, "Requested backtest data")
        if times[0] != context_start or times[-1] != data_end:
            raise ValueError(
                "The available datasets do not fully cover the requested backtest context/target interval. "
                f"Expected {context_start.date()}..{data_end.date()}, got {times[0].date()}..{times[-1].date()}."
            )
        selection = "requested_target_interval"

    times = _validate_daily_axis(X, "Backtest data")
    indices = _candidate_target_indices(
        times,
        loaded.config.window_size,
        loaded.config.forecast_horizon,
        target_start,
        target_end,
    )
    predictions, actual, input_end_times, target_times = _predict_windows(
        loaded,
        _scaled_features(X, loaded.feature_scaler),
        np.asarray(y.values, dtype=np.float32),
        times,
        indices,
        batch_size,
    )
    assert actual is not None
    return predictions, actual, input_end_times, target_times, {
        "selection": selection,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
    }


def _run_forecast(
    loaded: LoadedExperiment,
    input_end_date: str | None,
    *,
    show_progress: bool,
    progress_time_chunk_days: int,
) -> tuple[np.ndarray, None, pd.DatetimeIndex, np.ndarray, dict[str, object]]:
    if input_end_date is None:
        raise ValueError("Forecast mode requires input_end_date.")
    input_end = pd.Timestamp(input_end_date).normalize()
    context_start = input_end - pd.Timedelta(days=loaded.config.window_size - 1)
    X, _y = _build_data(
        loaded,
        context_start.date().isoformat(),
        input_end.date().isoformat(),
        show_progress=show_progress,
        progress_time_chunk_days=progress_time_chunk_days,
    )
    times = _validate_daily_axis(X, "Forecast input")
    if len(times) != loaded.config.window_size or times[-1] != input_end:
        raise ValueError(
            f"Forecast requires exactly {loaded.config.window_size} consecutive observations "
            f"ending on {input_end.date().isoformat()}; got {len(times)} ending on {times[-1].date()}."
        )
    values = _scaled_features(X, loaded.feature_scaler)[None, ...]
    inputs = torch.as_tensor(values, dtype=torch.float32, device=loaded.device)
    with torch.inference_mode():
        predictions = loaded.model(inputs).detach().cpu().numpy()
    expected_shape = (1, loaded.config.forecast_horizon, len(loaded.station_names))
    if predictions.shape != expected_shape:
        raise ValueError(f"Model output shape {predictions.shape} differs from expected {expected_shape}.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model produced NaN or infinite predictions.")
    predictions = _physical_predictions(predictions, loaded.target_scaler)
    target_times = pd.date_range(
        input_end + pd.Timedelta(days=1),
        periods=loaded.config.forecast_horizon,
        freq="D",
    ).values.reshape(1, -1)
    return predictions, None, pd.DatetimeIndex([input_end]), target_times, {
        "selection": "single_future_forecast",
        "input_end_date": input_end.date().isoformat(),
    }


def _prediction_dataframe(
    predictions: np.ndarray,
    actual: np.ndarray | None,
    input_end_times: pd.DatetimeIndex,
    target_times: np.ndarray,
    station_names: tuple[str, ...],
) -> pd.DataFrame:
    n_samples, horizon, n_stations = predictions.shape
    samples = np.repeat(np.arange(n_samples, dtype=np.int64), horizon * n_stations)
    input_dates = np.repeat(input_end_times.strftime("%Y-%m-%d"), horizon * n_stations)
    lead_days = np.tile(np.repeat(np.arange(1, horizon + 1), n_stations), n_samples)
    target_dates = np.repeat(
        pd.DatetimeIndex(target_times.reshape(-1)).strftime("%Y-%m-%d"),
        n_stations,
    )
    stations = np.tile(np.asarray(station_names, dtype=object), n_samples * horizon)
    predicted = predictions.reshape(-1)
    actual_flat = actual.reshape(-1) if actual is not None else np.full(predicted.shape, np.nan)
    residual = actual_flat - predicted
    return pd.DataFrame(
        {
            "sample": samples,
            "input_end_time": input_dates,
            "lead_day": lead_days,
            "target_time": target_dates,
            "station": stations,
            "actual": actual_flat,
            "predicted": predicted,
            "residual": residual,
            "actual_mm": actual_flat,
            "predicted_mm": predicted,
            "residual_mm": residual,
            "absolute_error_mm": np.abs(residual),
            "squared_error_mm2": residual**2,
        }
    )


def _metrics_by_lead(
    predictions: np.ndarray,
    actual: np.ndarray | None,
    *,
    metric_standard=None,
    metric_threshold=0.0,
) -> pd.DataFrame:
    metric_standard = normalize_metric_standard(metric_standard)
    metric_threshold = validate_metric_threshold(metric_threshold)
    if actual is None:
        return pd.DataFrame(
            columns=[
                "lead_day", "n", "n_metric_targets", "metric_standard",
                "metric_threshold_mm", "MSE", "RMSE", "MAE", "R2", "bias",
            ]
        )
    rows = []
    for lead_index in range(predictions.shape[1]):
        predicted = predictions[:, lead_index].reshape(-1)
        observed = actual[:, lead_index].reshape(-1)
        metrics = numpy_regression_metrics(
            observed,
            predicted,
            metric_standard=metric_standard,
            metric_threshold=metric_threshold,
        )
        rows.append(
            {
                "lead_day": lead_index + 1,
                "n": int(np.count_nonzero(np.isfinite(predicted) & np.isfinite(observed))),
                "n_metric_targets": metrics["count"],
                "metric_standard": metric_standard,
                "metric_threshold_mm": (
                    metric_threshold
                    if metric_standard == METRIC_STANDARD_MODIFIED
                    else np.nan
                ),
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "R2": metrics["r2"],
                "bias": metrics["bias"],
            }
        )
    return pd.DataFrame(rows)


def _create_output_directory(run_dir: Path, mode: str, output_dir: Path | str | None) -> Path:
    if output_dir is not None:
        destination = Path(output_dir).resolve()
        destination.mkdir(parents=True, exist_ok=False)
        return destination
    parent = run_dir / "inference"
    parent.mkdir(parents=True, exist_ok=True)
    base_name = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for suffix in range(1, 10_000):
        candidate = parent / (base_name if suffix == 1 else f"{base_name}_{suffix:02d}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique inference directory below {parent}.")


def run_inference(
    run_dir: Path | str,
    *,
    mode: str = "backtest",
    start_date: str | None = None,
    end_date: str | None = None,
    input_end_date: str | None = None,
    output_dir: Path | str | None = None,
    catalog_path: Path | str | None = None,
    device: str | torch.device = "auto",
    batch_size: int | None = None,
    rebuild_legacy_scalers: bool = True,
    show_progress: bool = True,
    progress_time_chunk_days: int = PROGRESS_TIME_CHUNK_DAYS,
) -> Path:
    """Run a historical backtest or one future forecast from a saved experiment."""
    mode = mode.lower().strip()
    if mode not in {"backtest", "forecast"}:
        raise ValueError("mode must be 'backtest' or 'forecast'.")
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"Output directory already exists: {Path(output_dir).resolve()}")
    loaded = load_trained_experiment(
        run_dir,
        catalog_path=catalog_path,
        device=device,
        rebuild_legacy_scalers=rebuild_legacy_scalers,
        show_progress=show_progress,
        progress_time_chunk_days=progress_time_chunk_days,
    )
    batch_size = loaded.config.batch_size if batch_size is None else int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if mode == "backtest":
        if input_end_date is not None:
            raise ValueError("input_end_date is only valid in forecast mode.")
        predictions, actual, input_end_times, target_times, selection = _run_backtest(
            loaded,
            start_date,
            end_date,
            batch_size,
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )
    else:
        if start_date is not None or end_date is not None:
            raise ValueError("start_date/end_date are only valid in backtest mode.")
        predictions, actual, input_end_times, target_times, selection = _run_forecast(
            loaded,
            input_end_date,
            show_progress=show_progress,
            progress_time_chunk_days=progress_time_chunk_days,
        )

    destination = _create_output_directory(loaded.run_dir, mode, output_dir)
    metric_standard = normalize_metric_standard(loaded.config.metric_standard)
    metric_threshold = validate_metric_threshold(loaded.config.metric_threshold)
    prediction_path = destination / "inference_predictions_by_lead_day.csv"
    metrics_path = destination / "inference_metrics_by_lead_day.csv"
    prediction_frame = _prediction_dataframe(
        predictions,
        actual,
        input_end_times,
        target_times,
        loaded.station_names,
    )
    prediction_frame["metric_eligible"] = (
        prediction_frame["actual_mm"] > metric_threshold
        if metric_standard == METRIC_STANDARD_MODIFIED
        else prediction_frame["actual_mm"].notna()
    )
    prediction_frame.to_csv(prediction_path, index=False)
    metrics = _metrics_by_lead(
        predictions,
        actual,
        metric_standard=metric_standard,
        metric_threshold=metric_threshold,
    )
    metrics.to_csv(metrics_path, index=False)

    request_payload = {
        "source_run": str(loaded.run_dir),
        "mode": mode,
        "start_date": start_date,
        "end_date": end_date,
        "input_end_date": input_end_date,
        "device": str(loaded.device),
        "batch_size": batch_size,
        "rebuild_legacy_scalers": rebuild_legacy_scalers,
            "preprocessing_source": loaded.preprocessing_source,
            "metric_standard": metric_standard,
            "metric_threshold_mm": metric_threshold,
            **selection,
    }
    _write_json(destination / "inference_config.json", request_payload)
    _write_json(
        destination / "inference_contract.json",
        {
            "input_shape": [len(predictions), loaded.config.window_size, len(loaded.station_names), len(loaded.feature_names)],
            "output_shape": list(predictions.shape),
            "stations": list(loaded.station_names),
            "features": list(loaded.feature_names),
            "input_end_time_start": input_end_times[0].date().isoformat(),
            "input_end_time_end": input_end_times[-1].date().isoformat(),
            "target_time_start": pd.Timestamp(target_times.reshape(-1).min()).date().isoformat(),
            "target_time_end": pd.Timestamp(target_times.reshape(-1).max()).date().isoformat(),
        },
    )
    _write_json(
        destination / "inference_summary.json",
        {
            "source_run": str(loaded.run_dir),
            "mode": mode,
            "n_samples": int(predictions.shape[0]),
            "forecast_horizon": int(predictions.shape[1]),
            "n_stations": int(predictions.shape[2]),
            "has_actual_targets": actual is not None,
            "metric_standard": metric_standard,
            "metric_threshold_mm": (
                metric_threshold
                if metric_standard == METRIC_STANDARD_MODIFIED
                else None
            ),
            "preprocessing_source": loaded.preprocessing_source,
            "predictions_csv": prediction_path.name,
            "metrics_csv": metrics_path.name,
        },
    )
    if show_progress:
        print(f"Inference complete: {destination}")
    return destination


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reload a trained GLSTM/graph Transformer run and create posterior predictions."
    )
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Experiment run directory.")
    parser.add_argument("--mode", choices=("backtest", "forecast"), default=MODE)
    parser.add_argument("--start-date", default=START_DATE, help="First D+1 target date for backtest mode.")
    parser.add_argument("--end-date", default=END_DATE, help="Last D+1 target date for backtest mode.")
    parser.add_argument(
        "--input-end-date",
        default=INPUT_END_DATE,
        help="Last observed input date for one D+1..D+H forecast.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--catalog-path", type=Path, default=CATALOG_PATH)
    parser.add_argument("--device", default=DEVICE, help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.set_defaults(
        rebuild_legacy_scalers=REBUILD_LEGACY_SCALERS,
        show_progress=SHOW_PROGRESS,
    )
    parser.add_argument(
        "--rebuild-legacy-scalers",
        dest="rebuild_legacy_scalers",
        action="store_true",
        help="Refit missing legacy scaler state from the original training data (default).",
    )
    parser.add_argument(
        "--no-rebuild-legacy-scalers",
        dest="rebuild_legacy_scalers",
        action="store_false",
        help="Fail instead of refitting missing legacy scaler state.",
    )
    parser.add_argument("--show-progress", dest="show_progress", action="store_true")
    parser.add_argument("--quiet", dest="show_progress", action="store_false")
    parser.add_argument(
        "--progress-time-chunk-days",
        type=int,
        default=PROGRESS_TIME_CHUNK_DAYS,
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.run_dir is None:
        parser.error("Set RUN_DIR in inference.py or pass --run-dir.")
    return run_inference(
        args.run_dir,
        mode=args.mode,
        start_date=args.start_date,
        end_date=args.end_date,
        input_end_date=args.input_end_date,
        output_dir=args.output_dir,
        catalog_path=args.catalog_path,
        device=args.device,
        batch_size=args.batch_size,
        rebuild_legacy_scalers=args.rebuild_legacy_scalers,
        show_progress=args.show_progress,
        progress_time_chunk_days=args.progress_time_chunk_days,
    )


if __name__ == "__main__":
    main()
