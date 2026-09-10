import argparse
import importlib.util

from Data.dataset_paths import convert_all_raw_nc_to_zarr, convert_raw_nc_to_zarr


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert raw NetCDF meteorological datasets to processed Zarr stores."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Raw NetCDF path, file name, or token such as precipitation, temp, sh, wind, or vv.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert every top-level .nc file found in Datasets/raw.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .zarr stores.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also search legacy Datasets/nc_files for raw NetCDF files.",
    )
    parser.add_argument(
        "--chunks-time",
        type=int,
        default=None,
        help="Optional xarray chunk size for the time dimension.",
    )
    return parser.parse_args(argv)


def _require_zarr():
    if importlib.util.find_spec("zarr") is None:
        raise SystemExit(
            "zarr is not installed. Install project dependencies with "
            "`pip install -r requirements.txt` before building processed stores."
        )


def main(argv=None):
    args = _parse_args(argv)
    if not args.all and not args.datasets:
        raise SystemExit("Pass one or more datasets, or use --all.")

    _require_zarr()
    chunks = {"time": args.chunks_time} if args.chunks_time is not None else None

    if args.all:
        outputs = convert_all_raw_nc_to_zarr(
            overwrite=args.overwrite,
            include_legacy=args.include_legacy,
            chunks=chunks,
        )
    else:
        outputs = [
            convert_raw_nc_to_zarr(
                dataset,
                overwrite=args.overwrite,
                include_legacy=args.include_legacy,
                chunks=chunks,
            )
            for dataset in args.datasets
        ]

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
