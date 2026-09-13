# Training Package Documentation

`src/Training` owns experiment configuration, model/loss resolution, training loops, evaluation loops, checkpointing, metric history, and prediction collection.

## Package-Level Conventions

- Training batches are `(xb, yb)` from a Torch `DataLoader`.
- Inputs are `[batch, window, station, feature]`.
- Targets are `[batch, horizon, station]`.
- The stable training path records loss, MSE, MAE, MAPE, R2, per-step R2, epoch time, checkpoints, and a JSON summary.
- Mixed precision is enabled only when configured and supported by the selected device.

## Libraries Used

- `torch` and `torch.nn` for model training, losses, optimizers, AMP, gradient clipping, and device movement.
- `tqdm` for epoch-level progress updates during maintained training runs.
- `copy`, `time`, `json`, and `pathlib` for best-state tracking, timing, summaries, and output files.
- `Evaluation.metrics` for R2, MAPE, weighted losses, and streaming R2.
- `Evaluation.comparison_plots.save_error_plots` for training-curve images.
- `Data.preprocessing.assert_finite` for optional debug checks.
- `Models.model` for GLSTM and Transformer construction.

## `experiment_runner.py`

Shared helper layer used by `src/run_experiment.py`.

Classes:

- `ExperimentRunConfig`: frozen dataclass capturing date range, feature switches, split/window settings, scaler policy, model settings (including `empty_graph`, `learn_adj`, `lock_topology`, and the GLSTM-only `learn_self_att` and `learn_std` options), training settings (including optional `warm_up`), regression `metric_standard`/`metric_threshold`, the physical-scale rain/no-rain `confusion_matrix_threshold`, loss settings, plot station, and daily-cache policy. `empty_graph=True` selects a separate LSTM and output head per station, without shared node parameters or graph edges; it requires `learn_std=False` and `learn_self_att=False`. The dataclass default for `learn_self_att` is `False` so legacy configurations keep the fixed identity diagonal.
- `QuantileMSELoss`: `nn.Module` implementing weighted MSE using train-only target quantile thresholds. It stores thresholds and bin weights as buffers.

Functions:

- `print_status(message, enabled=True)`: conditional console logging.
- `select_stations(stations, max_stations)`: optionally truncates a station dictionary for smoke runs. Raises if `max_stations < 1`.
- `resolve_catalog_path(catalog_path)`: resolves an exact catalog path or glob fallback such as `Catalogo*.csv`.
- `run_directory(output_root, sweep_name)`: creates a timestamped or named run directory.
- `build_model(model_type, n_stations, n_features, edge_index, config, *, edge_weight=None, share_glstm_adjacency=True)`: builds `NodewiseLSTM` when `config.empty_graph=True`, otherwise `GLSTM_v2` or `GraphTemporalTransformer_v1`. The nodewise baseline has one unshared LSTM/head pair per station and ignores graph/topology settings. `edge_weight` supplies an optional geographic/climatological graph prior for graph models. Maintained GLSTM runs use one adjacency shared by all recurrent layers; the optional false value is reserved for exact reconstruction of historical checkpoints. `learn_self_att=True` is passed only to GLSTM and learns a distinct positive diagonal weight per station as `exp(diag(a_logits))`: each diagonal logit starts at `0`, so its initial effective raw weight is `exp(0) = 1`. `False` keeps the historical identity diagonal. The true setting requires `learn_adj=True`; both `learn_self_att=True` and `learn_std=True` are rejected for Transformer runs.
- `QuantileMSELoss.__init__(thresholds, weights, normalize_weights=True, eps=1e-8)`: stores quantile thresholds and weights.
- `QuantileMSELoss.forward(y_pred, y_true)`: bucketizes targets, applies bin weights, normalizes weights by batch mean, and returns weighted MSE.
- `_as_float_tuple(values, name)`: validates scalar/list/tuple/tensor numeric inputs and returns a tuple of floats.
- `_quantile_weights_from_train(y_train, thresholds, quantile_weights, max_weight)`: computes automatic inverse-frequency bin weights or validates manual weights.
- `resolve_loss_function(...)`: resolves `"mse"`, `"mae"`, `"huber"`, or `"quantile_mse"` into an `nn.Module` plus metadata for the run summary.
- `unpack_model_output(output)`: validates either the historical forecast tensor or the optional `(forecast, predicted_node_std)` tuple.
- `collect_model_predictions(model, data_loader, return_std=False)`: runs model inference over a loader and concatenates CPU predictions. Its default keeps the historical tensor return; `return_std=True` returns both concatenated tensors.

## Comparative runs in `run_experiment.py`

The source runner can execute a Cartesian comparison grid without passing lists to
the model or data pipeline. Configure `comparative_run = True` and set
`comparative_parameter` to the string naming the main comparison axis, such as
`"hidden_dim"`, `"HIDDEN_DIM"`, or the aliases `"lr"`, `"wd"`, `"k"`,
`"window"`, and `"horizon"`.

The source-level selector is `LEARN_SELF_ATT` (materialized as
`learn_self_att` in `ExperimentRunConfig`). It is `True` in the maintained
runner, so GLSTM runs calibrate the self-attention diagonal; set it to `False`
to retain unit self-loops. The true setting requires `LEARN_ADJ=True`. As with
other scalar run settings, either Boolean can also participate in a comparative
grid.

- Every data, model, training, and runner setting represented by
  `_default_run_parameter_values()` accepts either one scalar value or a list of
  candidates. Multiple candidate lists form a Cartesian product.
- `LOSS_QUANTILES` and manual `LOSS_QUANTILE_WEIGHTS` are already vector-valued
  settings for one run. Keep a flat list for one run; use nested lists to compare
  alternatives, for example `LOSS_QUANTILES = [[0.75], [0.9]]`.
- The selected `comparative_parameter` must have at least two distinct values.
  With `comparative_run = False`, list-valued candidates are rejected instead of
  being silently passed to a scalar training call.
- A comparison gets a unique parent directory, one uniquely named child directory
  per scalar configuration, and `comparative_summary.json` containing all
  materialized parameters, run paths, training summaries, and test metrics.
  `OUTPUT_ROOT` and `SWEEP_NAME` select that parent directory and are not grid
  dimensions.
- Every scalar run writes `parameters.md` with a two-column table covering all
  source-selectable settings and their effective values. Comparative parents also
  write the table, preserving candidate lists, while each child records its
  materialized scalar choices.
- `CONFUSION_MATRIX_THRESHOLD` is an independent physical precipitation cutoff
  in millimetres. The runner records precision, recall, accuracy, and AUC in
  `logs/test_metrics.json` and physical-scale metrics in
  `logs/test_metrics_physical_scale.json`; it writes `confusion_matrix.png` and
  `logs/test_confusion_matrix.json` with rows for real classes and columns for
  predictions. In each case, `Chove` means precipitation strictly greater than
  the configured cutoff.
- After every comparative grid finishes, the runner writes
  `comparative_analysis/report_compare.tex`. Its figures compare each run's
  train/validation Loss, RMSE, MAE, and R2 histories; selected-station
  predictions and side-by-side scatters are rebuilt from common test dates for
  every shared forecast lead day. The selected station is resolved from
  `PLOT_STATION_NAME` while tolerating punctuation and accent differences.

## `Training_Routines.py`

Training and evaluation loops.

Functions:

- `eval_with_loader_stable(..., metric_standard=None, metric_threshold=0.0, metric_threshold_mm=None)`: evaluates a model on a loader. It accumulates loss and MAPE on all targets; MSE/RMSE, MAE, batch/global/per-step R2, and eligible-target counts follow the configured metric policy. `metric_threshold` is in the target tensor scale and `metric_threshold_mm` is traceability metadata.
- `resolve_adaptative_lr_metric(metric)`: validates the scalar validation metric monitored by `ReduceLROnPlateau` and selects `mode="min"` for error metrics or `mode="max"` for R2 metrics.
- `adaptative_lr_metric_value(val_metrics, metric)`: obtains the selected validation value, deriving RMSE from MSE when requested.
- `resolve_training_metric_monitors(metric_standard, adaptative_lr_metric)`: preserves the legacy scheduler/validation-loss early stopping under `None`; under `"modified"`, both scheduler and checkpoint/early stopping use a filtered regression metric. `loss` or `mape` requests fall back to modified RMSE because those two quantities are intentionally unfiltered.
- `train_stable(..., warm_up=0, adaptative_lr_metric="loss", metric_standard=None, metric_threshold=0.0, metric_threshold_mm=None)`: maintained training loop. It sets up optimizer parameter groups, `ReduceLROnPlateau`, AMP scaler, gradient clipping, early stopping, checkpoint writing, history tracking, and final summary JSON. For models exposing `adjacency_anchor_loss()`, `a_logits` are excluded from ordinary Adam weight decay and receive `weight_decay * adjacency_anchor_loss()` instead, anchoring them to the initial graph prior. `adaptative_lr_metric` accepts `loss`, `mse`, `rmse`, `mae`, `mape`, `r2`, or `r2_batch_mean`. `warm_up` is the number of completed epochs whose validation metrics are recorded for diagnostics but ignored by the scheduler and early-stopping patience. A validation epoch with no eligible modified targets does not update patience, scheduler, or checkpoint state.
- With `learn_std=True`, stable training computes the population standard
  deviation of each `[station]` target slice (`correction=0`) and adds its MSE
  to the configured forecast loss with unit weight. Forecast metrics still use
  only `[B, horizon, station]`; forecast and auxiliary losses are recorded
  separately while `train_loss`/`val_loss` contain their sum. With false, the
  original single-output loss path is unchanged.
- `eval_with_loader(model, loader, criterion, use_amp, amp_device, amp_dtype)`: older evaluation loop retained for compatibility.
- `train_batched_only(model, train_loader, val_loader, train_period, hidden_dim, epochs, lr, weight_decay, patience, criterion=None, run_dir=None)`: legacy training routine for older notebook experiments.

## Training Logic

The maintained `train_stable(...)` flow is:

1. Move the model and criterion to the selected device.
2. Build optimizer parameter groups, allowing a different learning-rate factor for the GLSTM's single shared adjacency parameter (or the Transformer's adjacency parameters).
3. Train one epoch over shuffled training batches.
4. Optionally run finite-value assertions when `debug_checks=True`.
5. Use AMP only when enabled and supported.
6. Backpropagate, unscale if needed, clip gradients, and update parameters.
7. Accumulate train metrics on device.
8. Evaluate validation data with `eval_with_loader_stable(...)`.
9. Update the epoch progress bar with `train_loss`, `val_loss`, `train_mse`, `val_mse`, `train_r2`, and `val_r2`.
10. Step the learning-rate scheduler on the effective validation monitor; minimize error metrics and maximize R2 metrics automatically. Modified mode guarantees that this is one of the threshold-filtered metrics.
11. Record validation metrics throughout the run, but only after the configured warm-up and with a finite effective metric update the scheduler/checkpoint baseline and early-stopping patience; stop when that patience is exceeded.
12. Write history curves to `train_history/` and machine-readable artifacts to
   `logs/`.

## Inputs and Outputs

- Inputs: model, train/validation loaders, window size, horizon, hidden size, training hyperparameters, criterion, run directory, and debug/loss metadata.
- Outputs: trained model, metric-history dictionary, and run-summary dictionary.
- Files: `logs/model_state_dict.pt`, `logs/hist.pt`,
  `logs/run_summary.json`, and `train_history/{mse,mae,r2}_curve.png` when
  `run_dir` is provided.

## Reloading trained runs with `inference.py`

`src/inference.py` reconstructs a trained GLSTM or graph Transformer from the
run directory. `load_trained_experiment(...)` reads the artifacts from
`logs/` (`config.json`, `dataset_contract.json`, `model_state_dict.pt`, and,
for new runs, `inference_state.json`). The model is built on CPU, its checkpoint is checked for
compatible keys/shapes, and only then moved to the requested device.

- `backtest` mode predicts a requested historical interval. `start_date` and
  `end_date` bound the D+1 target/origin dates; later lead dates extend through
  `end_date + horizon - 1`. If no dates are supplied, it reproduces the original
  test split. The long-form CSV keeps the existing
  actual/predicted/residual schema and adds `input_end_time`.
- `forecast` mode consumes exactly the final `window_size` consecutive daily
  observations ending at `input_end_date` and writes one D+1..D+H forecast;
  actual/residual columns remain empty because future targets are not required.
- New training runs save schema-v2 `inference_state.json`, including fitted
  scalers, station coordinates/order, feature order, base topology, and the
  effective non-tensor model-construction options (such as Transformer heads
  and layers or GLSTM cell clipping). GLSTM runs also record
  `model_build.adjacency_scope="shared"` and persist `learn_self_att` in
  `model_build.kwargs`, ensuring inference reconstructs the same fixed or
  calibrated diagonal. Older artifacts that omit `learn_self_att` default to
  `False`; older schema-v2 artifacts without the optional adjacency-scope field
  are reconstructed with their original per-layer graphs. Runs created
  before this artifact existed remain supported: the loader reconstructs their
  missing scalers from the original training period and recorded split/window
  settings, validates dataset dimensions/dates, and checks a reconstructed
  prediction against sample zero in the run's saved test-prediction CSV.

Examples:

```powershell
python src/inference.py --run-dir <run-directory> --mode backtest

python src/inference.py --run-dir <run-directory> --mode backtest `
  --start-date 2025-01-01 --end-date 2025-06-30

python src/inference.py --run-dir <run-directory> --mode forecast `
  --input-end-date 2025-12-31
```

Every invocation creates a unique folder below `<run>/inference/` unless
`--output-dir` is provided. Its `logs/` subdirectory contains
`inference_config.json`, `inference_contract.json`, `inference_summary.json`,
`inference_predictions_by_lead_day.csv`, and
`inference_metrics_by_lead_day.csv`.
