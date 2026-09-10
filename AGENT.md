# AGENT.md

## Project Context

This repository implements precipitation forecasting with LSTM, GLSTM, and graph-temporal models over meteorological station nodes. The maintained source code is under `src/`, while `Notebooks/`, `Experiments/`, and `PLOTS/` contain exploratory work and generated artifacts.

Read this file before making future changes. The worktree often contains local experiment outputs and notebook edits, so do not revert unrelated files.

## Repository Layout

- `src/Data/`: dataset loading, ERA5 feature extraction, preprocessing, temporal windowing, and train/validation/test split helpers.
- `src/run_experiment.py`: configurable source-level experiment runner in the same spirit as `climate_temporal_cluster`; edit the constants at the top to choose a run.
- `src/Graph/`: station graph construction, distance/KNN topology, adjacency matrix helpers, and graph plotting utilities.
- `src/Models/`: GLSTM and graph-temporal Transformer model definitions.
- `src/Training/`: training/evaluation loops, metric history, checkpoint saving, and run summaries.
- `src/Evaluation/`: metrics, prediction helpers, experiment aggregation, and plotting utilities.
- `Datasets/`: local data root. Raw and processed data are intentionally large and should stay out of git.
- `Experiments/` and `PLOTS/`: generated outputs from training/evaluation runs.

Each maintained `src/` subpackage has a `documentation.md` file. Update the relevant package documentation when adding files, changing function contracts, or changing the expected tensor/xarray shapes.

## Dataset Contract

Use these canonical folders for meteorological datasets:

- `Datasets/raw/`: raw meteorological feature datasets in NetCDF format (`*.nc`).
- `Datasets/processed/`: processed datasets in Zarr format (`*.zarr`) for faster repeated reads.
- `Datasets/nc_files/`: legacy NetCDF location. New code should not write here; it may contain NTFS hardlinks to `Datasets/raw` only to keep old notebooks from breaking during migration.
- `Datasets/dados_inmet/` and `Datasets/Dados INMET/`: existing CSV station/catalog inputs. Keep them unchanged unless the task explicitly asks to normalize the station-data layout.

The central dataset API is `src/Data/dataset_paths.py`:

- `ensure_dataset_directories()` creates `Datasets/raw` and `Datasets/processed`.
- `open_meteorological_dataset(...)` prefers a matching `.zarr` cache in `Datasets/processed` and falls back to `.nc` in `Datasets/raw`, then to legacy `Datasets/nc_files`.
- `convert_raw_nc_to_zarr(...)` converts one raw NetCDF file into a processed Zarr store.
- `convert_all_raw_nc_to_zarr(...)` converts all top-level raw NetCDF files.
- `python -m Data.process_datasets --all` materializes the Zarr cache from `Datasets/raw` when `PYTHONPATH=src` or the command is run with `src` on `sys.path`.

`src/Data/feature_extraction.py::smart_load_dataset(...)` now uses the central API. Future notebooks should call `smart_load_dataset(dataset_path, "precipitation")` or `open_meteorological_dataset("precipitation")` instead of hard-coding `xr.open_dataset("../Datasets/nc_files/...")`.

## Data Pipeline Notes

- ERA5 raw datasets are selected by token in file name, for example `precipitation`, `temp`, `sh`, `wind`, or `vv`.
- Daily feature extraction lives in `feature_extraction.py`, including precipitation, temperature/dew point, specific humidity, wind UV, and vertical velocity helpers.
- Dated station-level extraction and chronology-first windowing live in `temporal_dataset.py`. This layer keeps `xarray.DataArray` objects with `time`, `station`, `feature`, `sample_start_time`, `input_end_time`, and per-lead-day `target_time` coordinates until the final model boundary.
- Sliding-window generation and temporal split logic live in `prepare_data.py`.
- `temporal_train_val_test_split(...)` splits the raw time axis first and creates windows afterward, avoiding target leakage across train/validation/test intervals.
- New experiment code should prefer `Data.temporal_dataset.chronological_split(...)` plus `create_windowed_splits(...)` so dates remain attached before conversion to Torch.
- Model input tensors generally follow `[time, nodes, features]` before windowing and `[batch, window, nodes, features]` after windowing.
- Targets generally follow `[batch, horizon, nodes]`.
- Convert to `torch.Tensor` only at the model boundary via `to_torch_window_splits(...)`.

## Development Guidance

- Keep source changes in `src/` when possible; avoid editing notebooks unless the user asks for notebook deliverables.
- Do not commit or generate large raw/processed datasets into git. `.gitignore` ignores `*.nc` and `*.zarr/`, while `.gitkeep` files preserve the canonical folders.
- Preserve compatibility with existing experiment outputs and run summaries unless the task specifically changes the output contract.
- Prefer adding small, reusable helpers in `src/Data/` rather than copying path logic into notebooks.
- When touching training behavior, verify shape contracts for `[B, T, N, F]` inputs and `[B, H, N]` targets.

## Quick Verification

Use a syntax-only compile check when dependency state is uncertain:

```powershell
python -B -c "from pathlib import Path; [compile(p.read_text(encoding='utf-8'), str(p), 'exec') for p in Path('src').rglob('*.py')]"
```

If `zarr` is not installed, NetCDF fallback reading still works, but `open_zarr` and `to_zarr` require installing the dependency from `requirements.txt`.
