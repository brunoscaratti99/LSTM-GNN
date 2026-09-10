"""Configurable GLSTM experiment runner with dated xarray data handling."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from collections.abc import Mapping
import json
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from Data.prepare_data import create_batchs
from Data.temporal_dataset import (
    build_station_feature_dataset,
    chronological_split,
    create_windowed_splits,
    load_station_catalog,
    scale_windowed_splits,
    to_torch_window_splits,
)
from Evaluation.experiment_outputs import (
    save_dataset_contract,
    save_graph_plot,
    save_inference_state,
    save_prediction_outputs,
    save_topology_heatmap,
    save_weighted_graph_plot,
)
from Evaluation.comparative_outputs import save_comparative_outputs
from Evaluation.metrics import (
    metric_threshold_in_target_scale,
    normalize_metric_standard,
    validate_metric_threshold,
)
from Graph.graph_related_utils import (
    knn_topology,
    normalize_station_similarity,
    station_similarity_edge_weights,
)
from Training.Training_Routines import (
    eval_with_loader_stable,
    resolve_adaptative_lr_metric,
    train_stable,
)
from Training.experiment_runner import (
    ExperimentRunConfig,
    build_model,
    collect_model_predictions,
    print_status,
    resolve_loss_function,
    resolve_catalog_path,
    run_directory,
    select_stations,
)


# Data and station selection
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
STATE = "RS"
CATALOG_PATH = ROOT / "Datasets" / "dados_inmet" / "Catalogo*.csv"
MAX_STATIONS = None  # Set an integer for quick smoke runs.

# Meteorological feature switches
INCLUDE_PRECIPITATION = True
INCLUDE_TEMPERATURE = True
INCLUDE_SPECIFIC_HUMIDITY = True
INCLUDE_WIND = False
INCLUDE_VERTICAL_VELOCITY = True    

# Windowing and split
WINDOW_SIZE = [15]
FORECAST_HORIZON = 5
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

# Scaling. Scalers are fit on train windows only.
NORMALIZE_FEATURES = True
FEATURE_SCALER = "standard"  # "standard" or "minmax"
NORMALIZE_TARGET = False
TARGET_SCALER = "standard"  # Used only when NORMALIZE_TARGET=True.

# Regression metric policy. With "modified", RMSE, MAE, and R2 only use
# targets whose physical precipitation is strictly above METRIC_THRESHOLD.
METRIC_STANDARD = None  # None (legacy/default) or "modified".
METRIC_THRESHOLD = 15.00 # Millimetres; ignored when METRIC_STANDARD=None.

# Graph and model
MODEL_TYPE = "glstm"  # "glstm" or "transformer"
# True selects a distinct LSTM and output head for every station: no edges,
# graph aggregation, or parameter sharing across nodes. False uses the GLSTM.
EMPTY_GRAPH = False
K_NEIGHBORS = 5
HIDDEN_DIM = [128]
LSTM_LAYERS = [2]
LEARN_ADJ = True  # If True, the adjacency matrix is learnable. Otherwise, it is fixed.
LEARN_SELF_ATT = False  # If True, GLSTM also calibrates each diagonal/self-loop weight.
LOCK_TOPOLOGY = True  # True: only initial edges; False: new edges may be learned.
# Initial graph prior for the KNN edges. Choices: "gaussian", "ones",
# "inverse_distance", or "climatology_correlation". Gaussian distances and
# sigma are in kilometres; climatology uses only the chronological train split.
STATION_SIMILARITY = "gaussian"
STATION_SIMILARITY_SIGMA_KM = [150]
DROPOUT = [0.0]
# Adds a GLSTM auxiliary head that predicts the population SD across stations
# for every forecast lead day. False preserves the original single-output model.
LEARN_STD = not EMPTY_GRAPH

# Training
EPOCHS = 400
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
# Validation metric monitored by ReduceLROnPlateau. Available values:
# "loss", "mse", "rmse", "mae", "mape", "r2", or "r2_batch_mean".
# In modified mode, scheduler and early stopping share this filtered metric;
# requests for unfiltered "loss"/"mape" automatically use modified RMSE.
ADAPTATIVE_LR_METRIC = "loss"
WEIGHT_DECAY = 0
PATIENCE = 150
WARM_UP = 5  # Minimum completed epochs before early stopping may interrupt training.
ADJ_LR_FACTOR = 1
MAX_GRAD_NORM = 1
LOSS = "quantile_mse"  # "mse", "mae", "huber", or "quantile_mse".
LOSS_QUANTILES = [0.5,0.99]  # Used only when LOSS="quantile_mse".
LOSS_QUANTILE_WEIGHTS = "auto"  # "auto" or one weight per quantile bin.
LOSS_QUANTILE_MAX_WEIGHT = 50.00
NUM_WORKERS = 0
SHUFFLE_TRAIN = False
RANDOM_SEED = 42
DEBUG_CHECKS = False
# Comparative grid. Every experiment/runtime setting can be a scalar or a list.
# Example: HIDDEN_DIM = [128, 256]; comparative_parameter = "hidden_dim".
# Multiple lists form a Cartesian product; see run_comparative_experiments().
COMPARATIVE_RUN: bool = False
COMPARATIVE_PARAMETER: str = "STATION_SIMILARITY_SIGMA_KM"

# Output
OUTPUT_ROOT = ROOT / "Experiments" / "run_experiment" / "07_09_2026"
SWEEP_NAME = None
SHOW_CONSOLE_INFO = True
PROGRESS_TIME_CHUNK_DAYS = 30
USE_DAILY_CACHE = True
# Use names in the style of Datasets/dados_inmet/*.csv, e.g. "PASSO FUNDO".
PLOT_STATION_NAME = "PORTO ALEGRE JARDIM BOTANICO"


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


_CONFIG_PARAMETER_NAMES = tuple(field.name for field in fields(ExperimentRunConfig))
_RUNTIME_PARAMETER_NAMES = (
    "catalog_path",
    "num_workers",
    "shuffle_train",
    "debug_checks",
    "progress_time_chunk_days",
    "show_console_info",
)
_STRUCTURED_LIST_PARAMETERS = frozenset({"loss_quantiles", "loss_quantile_weights"})
_COMPARATIVE_PARAMETER_ALIASES = {
    name: name for name in (*_CONFIG_PARAMETER_NAMES, *_RUNTIME_PARAMETER_NAMES)
}
_COMPARATIVE_PARAMETER_ALIASES.update(
    {
        "k": "k_neighbors",
        "neighbors": "k_neighbors",
        "lr": "learning_rate",
        "lr_metric": "adaptative_lr_metric",
        "wd": "weight_decay",
        "window": "window_size",
        "horizon": "forecast_horizon",
    }
)


def _default_run_parameter_values() -> dict[str, object]:
    """Return the source-level settings that can participate in a comparison grid."""
    return {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "state": STATE,
        "max_stations": MAX_STATIONS,
        "include_precipitation": INCLUDE_PRECIPITATION,
        "include_temperature": INCLUDE_TEMPERATURE,
        "include_specific_humidity": INCLUDE_SPECIFIC_HUMIDITY,
        "include_wind": INCLUDE_WIND,
        "include_vertical_velocity": INCLUDE_VERTICAL_VELOCITY,
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "normalize_features": NORMALIZE_FEATURES,
        "feature_scaler": FEATURE_SCALER,
        "normalize_target": NORMALIZE_TARGET,
        "target_scaler": TARGET_SCALER,
        "metric_standard": METRIC_STANDARD,
        "metric_threshold": METRIC_THRESHOLD,
        "model_type": MODEL_TYPE,
        "empty_graph": EMPTY_GRAPH,
        "k_neighbors": K_NEIGHBORS,
        "hidden_dim": HIDDEN_DIM,
        "lstm_layers": LSTM_LAYERS,
        "learn_adj": LEARN_ADJ,
        "learn_self_att": LEARN_SELF_ATT,
        "lock_topology": LOCK_TOPOLOGY,
        "station_similarity": STATION_SIMILARITY,
        "station_similarity_sigma_km": STATION_SIMILARITY_SIGMA_KM,
        "dropout": DROPOUT,
        "learn_std": LEARN_STD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "adaptative_lr_metric": ADAPTATIVE_LR_METRIC,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "adj_lr_factor": ADJ_LR_FACTOR,
        "max_grad_norm": MAX_GRAD_NORM,
        "loss": LOSS,
        "loss_quantiles": LOSS_QUANTILES,
        "loss_quantile_weights": LOSS_QUANTILE_WEIGHTS,
        "loss_quantile_max_weight": LOSS_QUANTILE_MAX_WEIGHT,
        "random_seed": RANDOM_SEED,
        "plot_station_name": PLOT_STATION_NAME,
        "use_daily_cache": USE_DAILY_CACHE,
        "warm_up": WARM_UP,
        "catalog_path": CATALOG_PATH,
        "num_workers": NUM_WORKERS,
        "shuffle_train": SHUFFLE_TRAIN,
        "debug_checks": DEBUG_CHECKS,
        "progress_time_chunk_days": PROGRESS_TIME_CHUNK_DAYS,
        "show_console_info": SHOW_CONSOLE_INFO,
    }


def _normalize_comparative_parameter(parameter: str) -> str:
    """Resolve a configuration field or a convenient comparison alias."""
    if not isinstance(parameter, str) or not parameter.strip():
        raise ValueError("comparative_parameter must be a non-empty string.")

    normalized = re.sub(r"[\s-]+", "_", parameter.strip().lower())
    try:
        return _COMPARATIVE_PARAMETER_ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_COMPARATIVE_PARAMETER_ALIASES))
        raise ValueError(
            f"Unknown comparative_parameter={parameter!r}. Available parameters: {available}."
        ) from exc


def _merge_run_parameter_values(overrides: Mapping[str, object] | None = None) -> dict[str, object]:
    """Merge optional scalar/list overrides with the source-level settings."""
    parameter_values = _default_run_parameter_values()
    if overrides is None:
        return parameter_values

    for name, value in overrides.items():
        parameter_values[_normalize_comparative_parameter(name)] = value
    return parameter_values


def _parameter_options(parameter: str, value: object) -> list[object]:
    """Return candidate values while keeping vector-valued loss settings atomic."""
    if parameter in _STRUCTURED_LIST_PARAMETERS:
        if isinstance(value, (list, tuple)) and value and all(
            isinstance(item, (list, tuple, np.ndarray, torch.Tensor)) for item in value
        ):
            options = list(value)
            if parameter == "loss_quantile_weights":
                return ["auto" if list(option) == ["auto"] else option for option in options]
            return options
        return [value]

    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{parameter} must contain at least one candidate value.")
        return list(value)
    return [value]


def _value_identity(value: object) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _single_run_parameter_values(parameter_values: Mapping[str, object]) -> dict[str, object]:
    """Materialize one scalar run, rejecting a grid outside comparative mode."""
    resolved = {}
    for parameter, value in parameter_values.items():
        options = _parameter_options(parameter, value)
        if len(options) != 1:
            raise ValueError(
                f"{parameter} has {len(options)} candidate values. "
                "Set comparative_run=True to execute a comparative grid."
            )
        resolved[parameter] = options[0]
    return resolved


def _build_comparative_grid(
    parameter_values: Mapping[str, object],
    parameter: str,
) -> tuple[str, list[dict[str, object]], tuple[str, ...], dict[str, list[object]]]:
    """Materialize the Cartesian product of every list-valued run setting."""
    resolved_parameter = _normalize_comparative_parameter(parameter)
    option_values = {
        name: _parameter_options(name, value)
        for name, value in parameter_values.items()
    }
    for name, options in option_values.items():
        if len({_value_identity(option) for option in options}) != len(options):
            raise ValueError(f"{name} contains duplicate candidate values.")

    pivot_options = option_values[resolved_parameter]
    if len(pivot_options) < 2:
        raise ValueError(
            f"comparative_parameter={resolved_parameter!r} needs at least two distinct values."
        )

    parameter_names = tuple(parameter_values)
    varied_parameters = tuple(
        name for name in parameter_names if len(option_values[name]) > 1
    )
    configurations = [
        dict(zip(parameter_names, values))
        for values in product(*(option_values[name] for name in parameter_names))
    ]
    return resolved_parameter, configurations, varied_parameters, option_values


def _config_from_parameter_values(parameter_values: Mapping[str, object]) -> ExperimentRunConfig:
    return ExperimentRunConfig(
        **{name: parameter_values[name] for name in _CONFIG_PARAMETER_NAMES}
    )


def _runtime_from_parameter_values(parameter_values: Mapping[str, object]) -> dict[str, object]:
    return {name: parameter_values[name] for name in _RUNTIME_PARAMETER_NAMES}


def _resolve_runtime_argument(
    parameter: str,
    explicit_value: object | None,
    source_value: object,
) -> object:
    value = source_value if explicit_value is None else explicit_value
    options = _parameter_options(parameter, value)
    if len(options) != 1:
        raise ValueError(
            f"{parameter} has {len(options)} candidate values. "
            "Use run_comparative_experiments() for list-valued settings."
        )
    return options[0]


def _safe_value_token(value: object) -> str:
    raw = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip('"'))
    return token.strip(".-_")[:72] or "value"


def _create_unique_directory(output_root: Path, name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in range(1, 10_000):
        candidate = output_root / (name if suffix == 1 else f"{name}_{suffix:02d}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique comparison directory below {output_root}.")


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def _markdown_parameter_value(value: object) -> str:
    """Format one selected value as a safe inline-code Markdown cell."""
    safe_value = _json_safe(value)
    if isinstance(safe_value, str):
        rendered = safe_value
    elif safe_value is None:
        rendered = "None"
    else:
        rendered = json.dumps(safe_value, ensure_ascii=False, sort_keys=True)
    rendered = (
        rendered.replace("\r", r"\r")
        .replace("\n", r"\n")
        .replace("|", r"\|")
        .replace("`", r"\`")
    )
    return f"`{rendered}`"


def _write_parameters_markdown(
    output_dir: Path,
    parameter_values: Mapping[str, object],
    *,
    comparative_run: bool,
    comparative_parameter: str,
    output_root: Path,
    sweep_name: str | None,
) -> Path:
    """Write every source-selectable experiment setting as a Markdown table."""
    selected_values = dict(parameter_values)
    selected_values.update(
        {
            "comparative_run": comparative_run,
            "comparative_parameter": comparative_parameter,
            "output_root": output_root,
            "sweep_name": sweep_name,
        }
    )

    lines = [
        "# Experiment parameters",
        "",
        "Values selected for this output. In comparative parents, lists contain the candidate values.",
        "",
        "| Parameter | Selected value |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{name.upper()}` | {_markdown_parameter_value(value)} |"
        for name, value in selected_values.items()
    )

    output_path = Path(output_dir) / "parameters.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    config: ExperimentRunConfig | None = None,
    output_root: Path | str | None = None,
    sweep_name: str | None = None,
    show_console_info: bool | None = None,
    *,
    catalog_path: Path | str | None = None,
    num_workers: int | None = None,
    shuffle_train: bool | None = None,
    debug_checks: bool | None = None,
    progress_time_chunk_days: int | None = None,
    comparison_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Run one scalar dated GLSTM/Transformer experiment and return its folder."""
    source_parameters = _default_run_parameter_values()
    if config is None:
        config_parameters = _single_run_parameter_values(
            {name: source_parameters[name] for name in _CONFIG_PARAMETER_NAMES}
        )
    else:
        config_parameters = _single_run_parameter_values(
            {name: getattr(config, name) for name in _CONFIG_PARAMETER_NAMES}
        )
    config = _config_from_parameter_values(config_parameters)
    resolve_adaptative_lr_metric(config.adaptative_lr_metric)
    config = replace(
        config,
        metric_standard=normalize_metric_standard(config.metric_standard),
        metric_threshold=validate_metric_threshold(config.metric_threshold),
        station_similarity=normalize_station_similarity(config.station_similarity),
    )
    if config.learn_std and config.model_type.lower().strip() != "glstm":
        raise ValueError("LEARN_STD=True is only supported when MODEL_TYPE='glstm'.")
    if config.learn_self_att and config.model_type.lower().strip() != "glstm":
        raise ValueError("LEARN_SELF_ATT=True is only supported when MODEL_TYPE='glstm'.")
    if config.learn_self_att and not config.learn_adj:
        raise ValueError("LEARN_SELF_ATT=True requires LEARN_ADJ=True.")
    if config.empty_graph and config.model_type.lower().strip() != "glstm":
        raise ValueError("EMPTY_GRAPH=True is only supported when MODEL_TYPE='glstm'.")
    if config.empty_graph and config.learn_std:
        raise ValueError("EMPTY_GRAPH=True requires LEARN_STD=False.")
    if config.empty_graph and config.learn_self_att:
        raise ValueError("EMPTY_GRAPH=True requires LEARN_SELF_ATT=False.")
    if (
        config.station_similarity == "gaussian"
        and (not np.isfinite(config.station_similarity_sigma_km)
             or config.station_similarity_sigma_km <= 0.0)
    ):
        raise ValueError("STATION_SIMILARITY_SIGMA_KM must be finite and greater than zero.")

    runtime_arguments = {
        "catalog_path": catalog_path,
        "num_workers": num_workers,
        "shuffle_train": shuffle_train,
        "debug_checks": debug_checks,
        "progress_time_chunk_days": progress_time_chunk_days,
        "show_console_info": show_console_info,
    }
    runtime = {
        name: _resolve_runtime_argument(name, runtime_arguments[name], source_parameters[name])
        for name in _RUNTIME_PARAMETER_NAMES
    }
    show_console_info = runtime["show_console_info"]
    output_root = Path(OUTPUT_ROOT if output_root is None else output_root)
    sweep_name = SWEEP_NAME if sweep_name is None else sweep_name

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    run_dir = run_directory(output_root, sweep_name)
    effective_parameters = dict(source_parameters)
    effective_parameters.update(asdict(config))
    effective_parameters.update(runtime)
    is_comparative_run = bool(
        comparison_metadata is not None
        and comparison_metadata.get("comparative_run", False)
    )
    effective_comparative_parameter = (
        str(comparison_metadata["comparative_parameter"])
        if comparison_metadata is not None and "comparative_parameter" in comparison_metadata
        else COMPARATIVE_PARAMETER
    )
    _write_parameters_markdown(
        run_dir,
        effective_parameters,
        comparative_run=is_comparative_run,
        comparative_parameter=effective_comparative_parameter,
        output_root=output_root,
        sweep_name=sweep_name,
    )

    print_status(
        f"Loading station catalog and dated xarray datasets ({config.start_date} -> {config.end_date})",
        show_console_info,
    )
    stations = select_stations(
        load_station_catalog(resolve_catalog_path(runtime["catalog_path"]), state=config.state),
        config.max_stations,
    )
    X_da, y_da = build_station_feature_dataset(
        stations=stations,
        start_date=config.start_date,
        end_date=config.end_date,
        include_precipitation=config.include_precipitation,
        include_temperature=config.include_temperature,
        include_specific_humidity=config.include_specific_humidity,
        include_wind=config.include_wind,
        include_vertical_velocity=config.include_vertical_velocity,
        show_progress=runtime["show_console_info"],
        progress_time_chunk_days=runtime["progress_time_chunk_days"],
        prefer_daily_cache=config.use_daily_cache,
    )

    print_status("Splitting chronology-first and creating dated windows", show_console_info)
    dated_splits = chronological_split(
        X_da,
        y_da,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    windowed = create_windowed_splits(
        dated_splits,
        window_size=config.window_size,
        horizon=config.forecast_horizon,
    )
    scaled_windowed, scaling_state = scale_windowed_splits(
        windowed,
        normalize_features=config.normalize_features,
        feature_scaler_type=config.feature_scaler,
        normalize_target=config.normalize_target,
        target_scaler_type=config.target_scaler,
    )
    metric_threshold_target_scale = metric_threshold_in_target_scale(
        config.metric_threshold,
        scaling_state.target_scaler,
    )
    tensor_splits = to_torch_window_splits(scaled_windowed)
    criterion, loss_metadata = resolve_loss_function(
        config.loss,
        tensor_splits.y_train,
        quantiles=config.loss_quantiles,
        quantile_weights=config.loss_quantile_weights,
        max_quantile_weight=config.loss_quantile_max_weight,
    )

    print_status(
        "Building independent nodewise LSTM" if config.empty_graph else "Building graph and model",
        show_console_info,
    )
    if config.empty_graph:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        _pos = {
            index: (float(coordinates[1]), float(coordinates[0]))
            for index, coordinates in enumerate(stations.values())
        }
        edge_weight = None
    else:
        edge_index, _pos = knn_topology(stations, k=config.k_neighbors)
        climatology = (
            dated_splits.train_y.values
            if config.station_similarity == "climatology_correlation"
            else None
        )
        edge_weight = station_similarity_edge_weights(
            stations,
            edge_index,
            station_similarity=config.station_similarity,
            gaussian_sigma_km=config.station_similarity_sigma_km,
            climatology=climatology,
        )
    topology_dir = run_dir / "topology"
    topology_dir.mkdir(exist_ok=True)
    save_graph_plot(topology_dir, edge_index, _pos, filename="graph.png")
    model = build_model(
        config.model_type,
        n_stations=tensor_splits.X_train.shape[2],
        n_features=tensor_splits.X_train.shape[3],
        edge_index=edge_index,
        config=config,
        edge_weight=edge_weight,
    )
    save_topology_heatmap(
        topology_dir,
        model,
        list(stations.keys()),
        filename="initial_adjacency.png",
        title=(
            "Independent node-only identity matrix"
            if config.empty_graph
            else "Initial model adjacency matrix"
        ),
    )
    save_weighted_graph_plot(
        topology_dir,
        model,
        _pos,
        filename="initial_graph.png",
        title="Initial station graph ($W_{adj}$)",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = create_batchs(
        tensor_splits.X_train,
        tensor_splits.X_val,
        tensor_splits.X_test,
        tensor_splits.y_train,
        tensor_splits.y_val,
        tensor_splits.y_test,
        batch_size=config.batch_size,
        device=device,
        num_workers=runtime["num_workers"],
        shuffle_train=runtime["shuffle_train"],
    )

    config_payload = asdict(config)
    if comparison_metadata is not None:
        config_payload["comparative"] = dict(comparison_metadata)
    _write_json(run_dir / "config.json", config_payload)
    save_dataset_contract(run_dir, X_da, y_da, windowed, scaled_windowed, config)
    save_inference_state(run_dir, model, edge_index, stations, X_da, scaling_state)

    print_status(f"Training on {device}; output={run_dir}", show_console_info)
    trained_model, history, summary = train_stable(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        window_size=config.window_size,
        horizon=config.forecast_horizon,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.learning_rate,
        adaptative_lr_metric=config.adaptative_lr_metric,
        weight_decay=config.weight_decay,
        patience=config.patience,
        warm_up=config.warm_up,
        criterion=criterion,
        run_dir=run_dir,
        adj_lr_factor=config.adj_lr_factor,
        max_norm=config.max_grad_norm,
        debug_checks=runtime["debug_checks"],
        loss_name=loss_metadata["loss"],
        loss_metadata=loss_metadata,
        metric_standard=config.metric_standard,
        metric_threshold=metric_threshold_target_scale,
        metric_threshold_mm=config.metric_threshold,
        learn_std=config.learn_std,
    )
    if comparison_metadata is not None:
        summary["comparative"] = _json_safe(dict(comparison_metadata))
        _write_json(run_dir / "run_summary.json", summary)

    test_metrics = eval_with_loader_stable(
        trained_model,
        test_loader,
        criterion=criterion,
        use_amp=False,
        amp_device="cuda" if device == "cuda" else "cpu",
        amp_dtype=torch.float16 if device == "cuda" else torch.bfloat16,
        debug_checks=runtime["debug_checks"],
        metric_standard=config.metric_standard,
        metric_threshold=metric_threshold_target_scale,
        metric_threshold_mm=config.metric_threshold,
        learn_std=config.learn_std,
    )
    test_metrics["metric_units"] = (
        "normalized_target" if scaling_state.target_scaler is not None else "mm"
    )
    _write_json(run_dir / "test_metrics.json", test_metrics)

    collected_predictions = collect_model_predictions(
        trained_model,
        test_loader,
        return_std=config.learn_std,
    )
    if config.learn_std:
        y_pred_test, predicted_node_std_test = collected_predictions
    else:
        y_pred_test = collected_predictions
        predicted_node_std_test = None
    save_prediction_outputs(
        run_dir,
        tensor_splits.y_test,
        y_pred_test,
        scaled_windowed.test_y,
        target_scaler=scaling_state.target_scaler,
        plot_station_name=config.plot_station_name,
        graph_model=trained_model,
        edge_index=edge_index,
        metric_standard=config.metric_standard,
        metric_threshold=config.metric_threshold,
        predicted_node_std=predicted_node_std_test,
    )
    if scaling_state.target_scaler is not None:
        print_status("Saved prediction artifacts were inverse-transformed to precipitation scale.", show_console_info)
    save_topology_heatmap(
        topology_dir,
        trained_model,
        list(stations.keys()),
        filename="final_adjacency.png",
        title=(
            "Independent node-only identity matrix"
            if config.empty_graph
            else "Best-validation model adjacency matrix"
        ),
    )
    save_weighted_graph_plot(
        topology_dir,
        trained_model,
        _pos,
        filename="weighted_graph.png",
    )

    print_status(f"Done: {run_dir}", show_console_info)
    return run_dir


def run_comparative_experiments(
    parameter_values: Mapping[str, object] | None = None,
    output_root: Path | str | None = None,
    sweep_name: str | None = None,
    parameter: str | None = None,
) -> Path:
    """Run the Cartesian product of list-valued settings and save a comparison manifest."""
    parameter_values = _merge_run_parameter_values(parameter_values)
    resolved_parameter, configurations, varied_parameters, option_values = _build_comparative_grid(
        parameter_values,
        COMPARATIVE_PARAMETER if parameter is None else parameter,
    )

    output_root = Path(OUTPUT_ROOT if output_root is None else output_root)
    default_name = f"comparative_{resolved_parameter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    parent_name = SWEEP_NAME if sweep_name is None else sweep_name
    sweep_dir = _create_unique_directory(output_root, parent_name or default_name)
    sweep_show_console_info = any(bool(value) for value in option_values["show_console_info"])
    _write_parameters_markdown(
        sweep_dir,
        parameter_values,
        comparative_run=True,
        comparative_parameter=resolved_parameter,
        output_root=output_root,
        sweep_name=parent_name,
    )

    manifest = {
        "comparative_run": True,
        "comparative_parameter": resolved_parameter,
        "varied_parameters": list(varied_parameters),
        "parameter_options": option_values,
        "total_runs": len(configurations),
        "runs": [],
    }
    for index, run_parameters in enumerate(configurations, start=1):
        suffix = "__".join(
            f"{name}={_safe_value_token(run_parameters[name])}"
            for name in varied_parameters
        )
        run_name = f"run_{index:03d}" + (f"__{suffix[:120]}" if suffix else "")
        metadata = {
            "comparative_run": True,
            "comparative_parameter": resolved_parameter,
            "comparative_value": run_parameters[resolved_parameter],
            "varied_parameters": {
                name: run_parameters[name] for name in varied_parameters
            },
            "run_index": index,
            "total_runs": len(configurations),
        }
        manifest["runs"].append(
            {
                "run_index": index,
                "run_name": run_name,
                "status": "pending",
                "parameters": run_parameters,
                "comparative": metadata,
            }
        )

    manifest_path = sweep_dir / "comparative_summary.json"
    _write_json(manifest_path, manifest)
    print_status(
        f"Starting comparative grid for {resolved_parameter}: {len(configurations)} run(s); output={sweep_dir}",
        sweep_show_console_info,
    )

    for record, run_parameters in zip(manifest["runs"], configurations):
        record["status"] = "running"
        _write_json(manifest_path, manifest)
        runtime = _runtime_from_parameter_values(run_parameters)
        try:
            run_dir = run_experiment(
                config=_config_from_parameter_values(run_parameters),
                output_root=sweep_dir,
                sweep_name=record["run_name"],
                show_console_info=runtime["show_console_info"],
                catalog_path=runtime["catalog_path"],
                num_workers=runtime["num_workers"],
                shuffle_train=runtime["shuffle_train"],
                debug_checks=runtime["debug_checks"],
                progress_time_chunk_days=runtime["progress_time_chunk_days"],
                comparison_metadata=record["comparative"],
            )
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(manifest_path, manifest)
            raise

        record["status"] = "completed"
        record["run_dir"] = str(run_dir.relative_to(sweep_dir))
        record["run_summary"] = _read_json(run_dir / "run_summary.json")
        record["test_metrics"] = _read_json(run_dir / "test_metrics.json")
        _write_json(manifest_path, manifest)

    comparative_analysis = save_comparative_outputs(sweep_dir, manifest)
    manifest["comparative_analysis"] = comparative_analysis
    _write_json(manifest_path, manifest)
    print_status(
        "Comparative report saved: "
        f"{sweep_dir / comparative_analysis['report_compare_tex']}",
        sweep_show_console_info,
    )
    print_status(f"Comparative grid complete: {sweep_dir}", sweep_show_console_info)
    return sweep_dir


if __name__ == "__main__":
    if COMPARATIVE_RUN:
        run_comparative_experiments()
    else:
        run_experiment()
