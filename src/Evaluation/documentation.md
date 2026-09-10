# Evaluation Package Documentation

`src/Evaluation` owns metrics, plotting style, experiment-output writers, prediction helpers, and comparison plots. It is used both by the source-level runner, the root-level `create_comparative_report.py` helper, and by older notebooks.

## Package-Level Conventions

- Prediction tensors generally use `[sample, lead_day, station]`.
- Dated prediction plots expect `test_y` xarray targets with a `target_time` coordinate shaped `[sample, lead_day]`.
- When a target scaler is present, saved prediction artifacts are inverse-transformed before CSV/plot generation.
- Plotting code uses Seaborn plotting APIs with Matplotlib only as the render
  backend, under the shared presentation style in `plot_style.py`.

## Libraries Used

- `torch` and `torch.nn` for metrics/loss helpers and prediction.
- `numpy` and `pandas` for metric aggregation and CSV tables.
- `matplotlib`, `matplotlib.dates`, and `seaborn` for figures.
- `xarray` for legacy date-aware plotting helpers.
- `json`, `pathlib`, `re`, and `unicodedata` for run metadata and robust station-name handling.

## `metrics.py`

Metric and loss helpers.

Functions:

- `safe_r2(y_true, y_pred, eps=1e-8)`: computes R2 with protection against zero variance. Inputs are Torch tensors; output is a scalar tensor.
- `safe_mape(y_true, y_pred, eps=1e-3)`: computes MAPE while clamping small denominators.
- `normalize_metric_standard(metric_standard)`: validates `None` or the
  threshold-filtered `"modified"` policy.
- `validate_metric_threshold(metric_threshold)`: validates the non-negative,
  finite precipitation cutoff in millimetres.
- `metric_threshold_in_target_scale(metric_threshold, target_scaler=None)`:
  maps the physical cutoff into the model target scale so normalized training
  tensors are selected by the same millimetre threshold.
- `numpy_regression_metrics(...)`: computes MSE, RMSE, MAE, R2, bias, and the
  eligible-target count. In modified mode it uses only finite pairs for which
  `y_true > metric_threshold`.
- `combined_loss(y_pred, y_true, alpha=0.5)`: blends MSE and MAE-style behavior using a weighting parameter.
- `weighted_mse_loss(...)`: increases squared-error weight for extreme precipitation values. Inputs are prediction/target tensors and quantile/weight settings.
- `_init_r2_tracker(horizon, device=None)`: creates global and per-step accumulators for streaming R2.
- `_update_r2_tracker(tracker, y_true, y_pred, metric_mask=None)`: updates
  streaming global/per-step R2 sums on the tensor device, optionally using a
  target-selection mask.
- `_finalize_r2_tracker(tracker, eps=1e-8)`: converts streaming sums into global and per-step R2 values.
- `make_weighted_mse_loss_standardized(...)`: builds a closure for weighted MSE when targets are already standardized by a train-only scaler.

For modified metrics, let (I_\tau=\{i:y_i>\tau\}),
(n_\tau=|I_\tau|), and
(\bar y_\tau=n_\tau^{-1}\sum_{i\in I_\tau}y_i). Then:

- (\mathrm{RMSE}_\tau=\sqrt{n_\tau^{-1}\sum_{i\in I_\tau}(\hat y_i-y_i)^2});
- (\mathrm{MAE}_\tau=n_\tau^{-1}\sum_{i\in I_\tau}|\hat y_i-y_i|);
- (R^2_\tau=1-\frac{\sum_{i\in I_\tau}(y_i-\hat y_i)^2}{\sum_{i\in I_\tau}(y_i-\bar y_\tau)^2}).

The selection is strict and target-defined. Metrics are `NaN` when no target is
eligible; R2 is also `NaN` when eligible targets have zero variance.

## `plot_style.py`

Shared plot theme and save helpers.

Functions:

- `apply_seaborn_theme()`: applies a publication-oriented Seaborn theme.
- `style_axis(ax, grid_axis="both")`: standardizes grid, spines, and axis appearance.
- `style_time_axis(ax)`: clears the redundant lower date-axis label and uses a
  y-only grid so date series remain readable when imported into Beamer.
- `save_figure(fig, output_path, dpi=190, tight=True)`: creates parent folders and writes a figure.
- `prediction_palette()`: returns a shared color palette for actual/predicted/train/validation/error series.

## `experiment_outputs.py`

Output writers for the maintained `src/run_experiment.py` workflow.

Functions:

- `_as_numpy(values)`: converts Torch tensors or array-like values to NumPy.
- `_model_adjacency_matrix(model, normalized=False)`: extracts a model adjacency matrix through `current_adjacency(...)` when available. For GLSTM this includes the effective diagonal: fixed identity entries with `learn_self_att=False`, or positive station-specific weights with `True`.
- `_heatmap_ticks(n_items)`: computes readable heatmap tick locations.
- `save_topology_heatmap(run_dir, model, station_names, filename="final_adjacency.png", title=...)`: saves an adjacency heatmap from the model's current state, including the GLSTM's calibrated diagonal when `learn_self_att=True`. The runner writes `topology/initial_adjacency.png` before training and `topology/final_adjacency.png` after restoring the best validation checkpoint.
- `save_graph_plot(run_dir, edge_index, pos, filename="graph.png", boundary_geojson=None)`: saves the initial station graph topology over the IBGE boundary of Rio Grande do Sul. `boundary_geojson` supports deterministic/offline callers; otherwise the boundary loader used by `RS_state_map.py` is reused and cached in memory. The runner writes it as `topology/graph.png`.
- `save_weighted_graph_plot(run_dir, model, pos, filename="weighted_graph.png", title=..., boundary_geojson=None)`: saves raw `W_adj` over the same RS map. Self-loops are omitted from this geographic plot even when their GLSTM weights are learned; inspect the topology heatmap for those diagonal values. Edge color represents `|W_adj[i,j]|`, and larger weights use thicker lines. The runner writes `topology/initial_graph.png` before training and `topology/weighted_graph.png` after restoring the best validation checkpoint.
- `_inverse_target_scale(values, target_scaler)`: inverse-transforms prediction arrays when a target scaler exists.
- `target_standard_deviation_to_physical_scale(values, target_scaler=None)`: maps learned target-scale standard deviations to millimetres using only the scaler's multiplicative scale, never its offset.
- `_prepare_prediction_arrays(y_true, y_pred, target_scaler=None)`: validates prediction shapes and returns physical-scale arrays.
- `_target_time_matrix(test_y, n_samples, n_leads)`: extracts and validates `[sample, lead_day]` target dates from xarray.
- `_station_names(test_y, n_stations)`: reads station names from xarray coordinates or creates fallback labels.
- `_safe_regression_metrics(actual, predicted, metric_standard=None, metric_threshold=0.0)`:
  computes physical MSE, RMSE, MAE, R2, bias, and eligible-target count while
  ignoring non-finite pairs and applying the configured strict threshold.
- `_axis_limits(actual, predicted)`: creates padded shared limits for true-vs-predicted plots.
- `_cross_node_mean(values)`: returns the finite daily mean across stations for each sample and forecast lead.
- `_station_mean(values)`: averages a prediction array across stations.
- `_normalize_station_name(name)`: removes accents and punctuation for robust station matching.
- `_resolve_plot_station(requested_station, station_names)`: resolves the station requested in the runner config.
- `_station_series(values, station_idx)`: extracts `[sample, lead_day]` data for one station.
- `_prediction_dataframe(actual, predicted, target_times, station_names)`: expands predictions into a long-form CSV table.
- `save_dataset_contract(run_dir, raw_X, raw_y, windowed, scaled_windowed, config)`: writes dimensions, features, station names, date range, scaler flags, and tensor-boundary notes.
- `_serialize_scaler_state(scaler)`: converts a fitted StandardScaler or MinMaxScaler into numeric JSON state.
- `_model_build_state(model)`: records effective non-tensor constructor options, including GLSTM `cell_clip`, `adjacency_scope`, `learn_self_att`, and `learn_std` plus Transformer heads/layer counts, which are not recoverable from tensor shapes alone. Persisting `learn_self_att` distinguishes a learned positive diagonal from the legacy fixed identity diagonal during reconstruction.
- `save_inference_state(run_dir, model, edge_index, stations, raw_X, scaling_state)`: writes `inference_state.json` with the exact model-construction options, including the GLSTM diagonal-learning choice, node/feature order, station coordinates, base `edge_index`, and fitted feature/target scaler state required to reload a run portably.
- `_save_prediction_overview(...)`: writes overview time-series and all-point scatter plots.
- `_save_prediction_timeseries_splits(...)`: writes station time-series plots split into several chronological chunks for each lead day.
- `_save_absolute_error_boxplots(...)`: writes `forecast_horizon_diagnostics/15_absolute_prediction_error_boxplots.png`, with one Seaborn boxplot of `|actual_mm - predicted_mm|` for each lead day and one final all-leads box. It uses every finite station/sample prediction directly; with the default five-day horizon this produces six boxes.
- `_save_forecast_lead_day_diagnostics(...)`: writes per-lead CSVs, metrics, error curves, scatter plots, and time-series diagnostics. All generated date series retain formatted date ticks but omit the lower `Target date` label.
- `save_oversmoothing_diagnostics(run_dir, actual, predicted, target_times, model=..., edge_index=...)`: writes all-node spatial-collapse diagnostics to `oversmoothing_diagnostics/`. It saves cross-node standard deviation over time in item 1, daily all-station precipitation means in item 2, predicted/actual dispersion ratio by lead day in item 3, predicted-versus-actual cross-node spread scatters in item 4, and graph Dirichlet energy over time using the final model adjacency in item 6. Items 1 and 2 use matching solid Real/Prediction curves of width 2.0; item 1 is a 30-day spread mean and item 2 is the daily station mean. Time-series subplots do not include a `Target date` x-axis label, and lead titles use `Lead Day i`. The raw time-aligned measurements and lead-day summary are also saved as CSV files.
- `save_prediction_outputs(..., metric_standard=None, metric_threshold=0.0, predicted_node_std=None)`:
  public orchestrator that writes all prediction CSVs and plots for a run. It
  adds `metric_eligible` to prediction rows, writes
  `test_metrics_physical_scale.json`, applies the policy to lead-day metrics,
  and includes `oversmoothing_diagnostics/` when supplied the trained graph
  model and/or base `edge_index`. When the auxiliary GLSTM output is supplied,
  it also writes `test_node_standard_deviation_predictions_by_lead_day.csv`
  with one physical-scale real/predicted spread pair per sample and lead day.
  This learned output remains distinct from the standard deviation derived
  directly from the station forecast matrix in oversmoothing diagnostics.

## `rs_animation_maps.py`

Rio Grande do Sul precipitation-map frames for LaTeX/Beamer animations.

Functions:

- `load_rs_state_geojson(url=..., cache_path=None)`: loads the RS boundary from the IBGE GeoJSON endpoint used by `Seminar_GLSTM/RS_state_map.py`, optionally with a local cache.
- `save_rs_precipitation_animation_frames(run_dir, start_date=..., end_date=..., lead_day=..., ...)`: filters the trained run's dated prediction CSV by an inclusive `target_time` interval and writes matching `Predict/frame_XXX.png`, `Real/frame_XXX.png`, and `Residual/frame_XXX.png` files for one fixed lead day, in chronological order. The legacy `sample=...` mode remains available to animate the complete D+1..D+H horizon of one forecast origin. All frames use circular station nodes over a pure-white RS background and no graph edges. Predict/Real share a white-to-blue precipitation scale, drawn only in the `Real` frame; `Residual` is `predicted_mm - actual_mm` on a shared symmetric scale, with positive values red, values near zero white, and negative values blue. The output also includes `animation_manifest.json` and a ready-to-copy Beamer example with one predicted/real frame and a separate residual-only frame.
- `salvar_frames_animacao_rs(...)`: Portuguese alias for the main export function.

`Seminar_GLSTM/RS_state_map.py::salvar_frames_animacao_run(...)` exposes the period mode with required `start_date`, `end_date`, and `lead_day` keywords. `src/export_rs_animation_maps.py` exposes both modes as a command-line tool. A real-valued test/backtest CSV is required because the exporter intentionally creates both `Predict` and `Real`; forecast-only CSVs with missing targets are rejected.

## `comparative_outputs.py`

Sweep-level artifacts for `run_comparative_experiments(...)`. The module reads each
completed run's `hist.pt` and `test_predictions_by_lead_day.csv`; it never uses
an individual run PNG as the source of a comparison.

Functions:

- `filter_complete_run_records(sweep_dir, manifest)`: validates sweep records before report generation. A run is report-ready when its run folder exists and contains `hist.pt`; the manifest status is not used as the inclusion criterion.
- `save_comparative_outputs(sweep_dir, manifest)`: creates the
  `comparative_analysis/` directory after a completed grid, recalculates
  selected-station metrics on the common target dates when prediction CSVs are present, skips runs without `hist.pt`,
  and returns artifact metadata for `comparative_summary.json`.

Root helper:

- `python create_comparative_report.py <comparative_folder>`: regenerates
  `comparative_analysis/report_compare.tex` and its supporting comparison
  figures from an existing `comparative_*` folder without rerunning training.
  A path to one of the sweep's `run_*` folders, its `hist.pt`, the sweep's
  `comparative_summary.json`, or `comparative_analysis/` is also accepted and
  resolved back to the containing sweep.
- `python create_comparative_report.py --search-root Experiments/run_experiment`
  with no positional path creates one aggregate report under
  `Experiments/run_experiment/comparative_analysis/`, combining every completed
  run found in descendant `comparative_*` folders. The aggregate labels include
  the source sweep and run name, and incomplete runs are still skipped.
  It still reads `comparative_summary.json` as the sweep manifest, but uses
  `hist.pt` inside each run folder as the run-inclusion indicator.

Generated files:

- `03_training_history_comparison.png`: a 2×2 panel of Loss, RMSE, MAE, and R2.
  Colors identify configurations; solid lines are training and dashed lines are
  validation curves.
- `04_comparative_parameter_metrics.png`: overview RMSE, MAE, and R2 against
  the configured comparative parameter, with a star marking the best value for
  each metric.
- `01_test_timeseries_comparison_lead_day_XX.png`: one actual line and all
  configuration predictions for the selected station on common dates.
- `02_test_scatter_comparison_lead_day_XX.png`: side-by-side selected-station
  scatters with shared axes, identity line, RMSE, MAE, and R2.
- `station_metrics_common_dates.csv`: per-run/per-lead selected-station metrics.
- `report_compare.tex`: LaTeX source that embeds the generated figures, an
  overview table across all lead days, and one metrics table per lead day.
  Best RMSE, MAE, and R2 values are bolded per table. The report states the
  metric formulas, strict threshold scope, and eligible-target counts. It is
  not compiled automatically.

## `comparison_plots.py`

Legacy and modernized plotting helpers for experiment comparison and notebook analysis.

Functions:

- `scatter_true_pred(y_true, y_pred, T_max=1, title="scatter plot")`: scatter plot for true/predicted arrays.
- `plot_heatmap_nn(...)`: heatmap for adjacency or weight matrices.
- `_parse_lr_token(token)`: parses learning-rate tokens from folder names.
- `_create_axes_with_right_legend_space(...)`: creates axes plus a dedicated legend panel.
- `_figure_size_with_right_legend_space(...)`: expands figure width for external legends.
- `_draw_legend_on_right_panel(ax, legend_ax, fontsize=8)`: moves legend entries to a right-side axis.
- `_apply_ylim(ax, ylim)`: applies optional y-axis limits.
- `_collect_unique_legend_entries(axes)`: deduplicates labels across axes.
- `_place_shared_legend_above_axes(...)`: places a shared legend above a multi-axis figure.
- `plot_por_hiperparametro_train_val(...)`: plots train/validation metrics grouped by hyperparameter.
- `infer_run_metadata(hist_path)`: infers experiment metadata from a history file path.
- `collect_runs(root_dir, recursive=True)`: collects run histories and inferred metadata into a DataFrame.
- `_smooth(values, window=1)`: moving-average smoothing helper.
- `_group_label(value, by)`: builds readable group labels for plot legends.
- `plot_convergence_overlay(...)`: overlays convergence curves across groups.
- `analisar_experimentos(...)`: high-level legacy experiment analysis entry point.
- `save_error_plots(...)`: writes train/validation MSE, MAE, and R2 curves.
- `model_weights_hist(model)`: plots model weight distributions.
- `plot_estacao_unica(...)`: plots one station's real vs predicted precipitation series.
- `_flatten_unique_days(y, stride=1)`: removes overlapped days from batched forecasts.
- `plot_precip_pred_vs_true(...)`: plots precipitation predictions without repeated days.
- `prever_futuro_precip_todos_nos(...)`: autoregressive future forecast helper for all stations.
- `plot_por_hiperparametro_train_val_lstm_glstm(...)`: larger-format variant for comparing LSTM/GLSTM groups.

## `predictions.py`

Small inference helper.

Functions:

- `prediction(model, X_input, mean=None, std=None)`: runs the model in evaluation/no-grad mode and optionally applies inverse log1p/z-score scaling through preprocessing utilities.
