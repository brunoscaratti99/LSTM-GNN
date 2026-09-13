# LSTM-GNN Precipitation Forecasting

This repository implements graph-temporal precipitation forecasting over meteorological station networks. It combines LSTM/GLSTM models, graph topology utilities, and an experimental graph-temporal Transformer with a date-aware ERA5/INMET data pipeline.

The current branch focuses on a reproducible source-level workflow:

- meteorological datasets are kept as dated `xarray` objects for as long as possible;
- chronological train/validation/test splitting happens before overlapping windows are created;
- scalers are fit only on training windows;
- tensors are created only at the model boundary;
- experiment outputs include configuration, dataset-contract metadata, metrics, prediction CSVs, graph images, and lead-day diagnostics.

## Project Motivation

Heavy precipitation events can create flooding, landslides, and infrastructure risk. Forecasting them requires models that can learn both:

- spatial dependencies between meteorological stations;
- temporal dependencies across atmospheric variables.

Graph-based models represent stations as nodes and spatial relationships as edges, while recurrent and attention-based temporal models learn from the historical sequence of meteorological features.

## Repository Layout

```text
LSTM-GNN/
|-- src/
|   |-- Data/          # Dataset paths, ERA5 feature extraction, windowing, scaling, and dataset CLIs.
|   |-- Evaluation/    # Metrics, plotting style, experiment outputs, and comparison utilities.
|   |-- Graph/         # Station graph construction, adjacency matrices, KNN/distance topology, and graph plots.
|   |-- Models/        # GLSTM cells/models and graph-temporal Transformer definitions.
|   |-- Training/      # Training loops, model/loss resolution, run summaries, and prediction collection.
|   |-- run_experiment.py
|   |-- run_benchmark_model.py
|   `-- precompute_daily_datasets.py
|-- Datasets/
|   |-- raw/           # Local raw NetCDF datasets; ignored except .gitkeep.
|   |-- processed/     # Local processed Zarr stores; ignored except .gitkeep.
|   |-- nc_files/      # Legacy NetCDF location used by older notebooks.
|   `-- dados_inmet/   # INMET station/catalog CSV inputs.
|-- Notebooks/         # Exploratory notebooks and legacy analyses.
|-- Experiments/       # Generated run outputs; ignored by git.
|-- PLOTS/             # Generated exploratory figures.
|-- AGENT.md           # Maintainer/development guidance for future agents.
`-- readme.md
```

Each source subfolder has its own `documentation.md` file describing the files, functions, input conventions, dependencies, and role in the pipeline.

## Data Contract

The maintained dataset convention is:

- `Datasets/raw/`: raw meteorological NetCDF files (`*.nc`).
- `Datasets/processed/`: processed Zarr stores (`*.zarr`) for faster repeated reads.
- `Datasets/processed/daily/`: daily aggregated Zarr caches generated from the raw ERA5 datasets.
- `Datasets/nc_files/`: legacy NetCDF location kept for older notebooks.
- `Datasets/dados_inmet/`: station-level INMET CSV files and the station catalog.

The central loader is `src/Data/dataset_paths.py`. It resolves datasets by explicit path, file name, or token such as `precipitation`, `temp`, `sh`, `wind`, or `vv`.

## Date-Aware Pipeline

The main pipeline lives in `src/Data/temporal_dataset.py`:

1. Load the INMET station catalog.
2. Open raw or processed ERA5 datasets through the central path API.
3. Aggregate hourly/forecast-step data into daily feature datasets when needed.
4. Sample gridded ERA5 variables at station coordinates.
5. Align all feature arrays on common `time` and `station` coordinates.
6. Split the raw dated time axis chronologically.
7. Create windows independently inside train, validation, and test blocks.
8. Fit feature/target scalers on training windows only.
9. Convert `[sample, lag, station, feature]` and `[sample, lead_day, station]` xarray windows to Torch tensors only before training.

This preserves `time`, `sample_start_time`, `input_end_time`, and per-lead `target_time` metadata for diagnostics and plots.

## Main Experiment Runner

The primary entry point is:

```powershell
python src/run_experiment.py
```

Edit the constants at the top of `src/run_experiment.py` to choose:

- date range and station subset;
- meteorological feature switches;
- window size and forecast horizon;
- normalization/scaler policy;
- regression metric policy (`METRIC_STANDARD=None` or `"modified"`) and the
  physical precipitation cutoff `METRIC_THRESHOLD` in millimetres;
- model type (`glstm` or `transformer`);
- graph KNN size and adjacency behavior;
- optional GLSTM diagonal/self-loop calibration (`LEARN_SELF_ATT`);
- optional GLSTM cross-node standard-deviation learning (`LEARN_STD`), or an `EMPTY_GRAPH=True` baseline with an independent LSTM/head per station and no graph edges;
- training loss, learning rate, patience, batch size, and output folder.

The runner writes outputs under `Experiments/run_experiment/<run_name>/`, including:

- `parameters.md`, with all source-selectable settings and their effective values;
- `logs/`, containing JSON, CSV, and PyTorch artifacts: configuration,
  dataset/inference contracts, metrics, prediction tables, `hist.pt`, and the
  model checkpoint;
- `train_history/`, containing `mse_curve.png`, `mae_curve.png`, and
  `r2_curve.png`;
- prediction overview plots;
- per-lead-day diagnostics;
- initial graph and learned topology images.

## Benchmark Models

Run statistical and naive baselines over the same dated data, station selection,
chronological split, window size, and forecast horizon used by `run_experiment.py`:

```powershell
python src/run_benchmark_model.py
```

Edit `BENCHMARK_MODELS` at the top of the file to select one or more of
`auto_arima`, `persistence_station`, `seasonal_persistence`, `station_mean`,
`station_median`, and `zero`. AutoARIMA is fitted independently per station,
selects its order from the training block only, and is evaluated with a rolling
forecast origin. Outputs are written below `Experiments/run_benchmark_model/`
with GLSTM-compatible test metrics and prediction diagnostics.

## Dataset Precomputation

To convert raw NetCDF files to processed Zarr stores:

```powershell
$env:PYTHONPATH = "src"
python -m Data.process_datasets --all
```

To precompute daily caches consumed by `run_experiment.py`:

```powershell
python src/precompute_daily_datasets.py
python src/precompute_daily_datasets.py precipitation temp sh --overwrite
```

Daily caches are stored under `Datasets/processed/daily/*.zarr`.

## Installation

Recommended Python versions are 3.10 to 3.12.

```powershell
git clone https://github.com/brunoscaratti99/LSTM-GNN.git
cd LSTM-GNN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If a custom CUDA build is required, install `torch` and `torch-geometric` from their official instructions first, then run `pip install -r requirements.txt`.

Main libraries:

- PyTorch and PyTorch Geometric;
- NumPy, Pandas, SciPy, and scikit-learn;
- Xarray, NetCDF4, Zarr, and Dask;
- Matplotlib, Seaborn, NetworkX, and tqdm.

## Model Families

The repository currently includes:

- `GLSTM_v1` and `GLSTM_v2`: graph-aware LSTM variants over station nodes.
- `GraphTemporalTransformer_v1`: a graph-temporal Transformer that mixes station adjacency with temporal attention.
- Utility components such as `LearnableAdjacency` and `GraphResidualMixer`.

Inputs generally follow:

- before windowing: `[time, station, feature]`;
- after windowing: `[batch, window, station, feature]`;
- targets: `[batch, horizon, station]`.

## Evaluation

Evaluation utilities report:

- MSE, RMSE, MAE, MAPE, and R2;
- global and per-lead-day metrics;
- prediction-vs-actual plots;
- station-specific time-series plots;
- graph topology plots and adjacency heatmaps;
- experiment convergence overlays.

Prediction outputs are inverse-transformed to precipitation units when a target scaler is used.

Set `METRIC_STANDARD="modified"` to calculate MSE/RMSE, MAE, and R2 only for
observations with `target_mm > METRIC_THRESHOLD`. The strict selection is made
from the observed precipitation, never from the prediction. The same filtered
validation metrics drive scheduler/checkpoint/early-stopping patience; if
`ADAPTATIVE_LR_METRIC` is `loss` or `mape`, modified RMSE is used as the
patience monitor. `METRIC_STANDARD=None` preserves the legacy all-target metric
and early-stopping behavior.

### RS maps for a Beamer animation

To animate a target-date period for one fixed forecast lead:

```powershell
python src/export_rs_animation_maps.py --run-dir "Experiments/run_experiment/<run>" --start-date 2021-01-01 --end-date 2021-01-31 --lead-day 3
```

The date bounds apply inclusively to `target_time`. Use the legacy `--sample 0`
mode to animate D+1 through D+H for one forecast origin, or `--sample -1` for
the last test sample. The command reads the run's existing
`test_predictions_by_lead_day.csv` and `inference_state.json`; it does not train
or rerun the model. Its default output is
`<run>/rs_animation_frames/lead_day_XX__START__END/`, containing:

- `Predict/frame_001.png`, `frame_002.png`, ...;
- `Real/frame_001.png`, `frame_002.png`, ...;
- `Residual/frame_001.png`, `frame_002.png`, ..., showing `predicted - real` in mm (red positive, white near zero, blue negative);
- `animation_manifest.json`, with the shared color limits and frame dates;
- `beamer_animate_example.tex`, ready to copy into the presentation, including a residual-only animation frame.

Predicted and real frames share a fixed white-to-blue scale, where white is
0 mm; its color bar appears only in the `Real` frame on the right. Both maps
have a pure-white RS background and a minimal two-line title: `Predict` or
`Real` above the target date in `YYYY-MM-DD` format. Stations are circular nodes
and graph edges are not rendered. Add
`\usepackage{animate}` to the Beamer preamble; the generated `animateinline`
example advances each predicted/real PNG pair in perfect synchronization.

### Real versus predicted time period

To reload a trained run, predict an inclusive target-date period, and save one
PNG with the real and predicted station series:

```powershell
python plot_timeperiod.py --run-dir "Experiments/run_experiment/<run>" --start-date 2024-05-01 --end-date 2024-06-01 --station "Porto Alegre"
```

The default output is `<run>/timeperiod_YYYYMMDD_YYYYMMDD.png`. It contains one
panel per forecast lead day. Add `--lead-day 3` to plot only D+3, or `--output`
to choose another PNG path. The script runs a backtest from the saved model and
does not modify the run directory with intermediate inference files.

For presentation-ready, separate Porto Alegre (Jardim Botânico) figures, use:

Set `START_DATE`, `END_DATE`, `RUN_PATH`, and `LEAD_DAYS` in the configuration
block at the top of `create_timeseries_plot.py`. Running the file from Python or
an IDE reads the run's saved `test_predictions_by_lead_day.csv` and writes
one true-versus-predicted PNG per requested lead day to
`<run>/plots_presentation/`. The date bounds are inclusive and the default lead
days are D+1 and D+5. These presentation-ready PNGs omit the chart titles and
the redundant lower `Target date` label while retaining formatted date ticks,
the precipitation axis, and the real/predicted legend.

## Verification

When dependency state is uncertain, use a syntax-only check:

```powershell
python -B -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in Path('src').rglob('*.py')]"
```

For a full run, make sure the required ERA5 datasets are available under `Datasets/raw/` or `Datasets/processed/`.

## Development Notes

- Keep new maintained code under `src/` whenever possible.
- Avoid committing generated datasets, `Experiments/`, `PLOTS/`, `__pycache__/`, or notebook execution artifacts.
- Prefer the central dataset API instead of hard-coded paths to `Datasets/nc_files/`.
- Preserve the chronology-first split and train-only fitting policy unless a task explicitly changes the experiment design.

## Citation

```bibtex
@software{lstm_gnn_precipitation,
  author = {Veloso, Bruno},
  title = {LSTM-GNN Precipitation Forecasting},
  year = {2026},
  url = {https://github.com/brunoscaratti99/LSTM-GNN}
}
```

## License

This project is available under the MIT License.
