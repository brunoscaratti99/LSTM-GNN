"""Regenerate ``report_compare.tex`` from an existing comparative run folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Evaluation.comparative_outputs import (  # noqa: E402
    ANALYSIS_DIRECTORY,
    REPORT_FILENAME,
    filter_complete_run_records,
    save_comparative_outputs,
)


MANIFEST_FILENAME = "comparative_summary.json"
DEFAULT_SEARCH_ROOT = (
    ROOT
    / "Experiments"
    / "run_experiment"
)


def _read_json(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: object) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)


def _resolve_sweep_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    start = resolved.parent if resolved.is_file() else resolved
    for candidate in (start, *start.parents):
        if (candidate / MANIFEST_FILENAME).is_file():
            return candidate
    return resolved


def _candidate_sweep_dirs(search_root: Path) -> list[Path]:
    search_root = search_root.expanduser().resolve()
    if not search_root.exists():
        return []

    candidates: set[Path] = set()
    containing_sweep = _resolve_sweep_dir(search_root)
    if (containing_sweep / MANIFEST_FILENAME).is_file():
        candidates.add(containing_sweep)

    scan_root = search_root if search_root.is_dir() else search_root.parent
    candidates.update(
        manifest_path.parent.resolve()
        for manifest_path in scan_root.rglob(MANIFEST_FILENAME)
        if manifest_path.parent.name.startswith("comparative_")
    )
    return sorted(
        candidates,
        key=lambda path: (path / MANIFEST_FILENAME).stat().st_mtime,
        reverse=True,
    )


def _latest_sweep_dir(search_root: Path) -> Path:
    candidates = _candidate_sweep_dirs(search_root)
    if not candidates:
        raise FileNotFoundError(f"No comparative_* folder with {MANIFEST_FILENAME} found under {search_root}.")
    for candidate in candidates:
        try:
            manifest = _read_json(candidate / MANIFEST_FILENAME)
            complete_runs, _skipped_runs = filter_complete_run_records(candidate, manifest)
        except Exception:  # noqa: BLE001 - keep looking for a usable completed sweep.
            continue
        if complete_runs:
            return candidate.resolve()
    raise FileNotFoundError(
        f"No comparative_* folder with at least one run containing hist.pt found under {search_root}."
    )


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _aggregate_manifest(search_root: Path) -> tuple[Path, dict[str, object]] | None:
    """Build a report manifest spanning all completed comparative runs below a root."""
    aggregate_root = search_root.expanduser().resolve()
    candidates = _candidate_sweep_dirs(aggregate_root)
    if not candidates:
        return None

    aggregate_runs: list[dict[str, object]] = []
    skipped_runs: list[dict[str, object]] = []
    for sweep_dir in candidates:
        try:
            manifest = _read_json(sweep_dir / MANIFEST_FILENAME)
            complete_runs, skipped = filter_complete_run_records(sweep_dir, manifest)
        except Exception as exc:  # noqa: BLE001 - tolerate damaged old sweeps.
            skipped_runs.append({"run_name": sweep_dir.name, "reasons": [str(exc)]})
            continue

        skipped_runs.extend(
            {
                "run_name": f"{sweep_dir.name}/{item['run_name']}",
                "reasons": item["reasons"],
            }
            for item in skipped
        )
        for record in complete_runs:
            run_dir = Path(record.get("run_dir") or record.get("run_name", ""))
            source_run_dir = run_dir if run_dir.is_absolute() else sweep_dir / run_dir
            original_label = (
                record.get("run_name")
                or record.get("label")
                or source_run_dir.name
            )
            configuration_label = f"{sweep_dir.name}/{original_label}"
            aggregate_record = dict(record)
            aggregate_record["run_dir"] = _relative_to_root(source_run_dir, aggregate_root)
            aggregate_record["run_name"] = configuration_label
            aggregate_record["run_index"] = len(aggregate_runs) + 1
            aggregate_record["comparative"] = {
                "comparative_parameter": "configuration",
                "comparative_run": True,
                "comparative_value": configuration_label,
                "run_index": len(aggregate_runs) + 1,
                "total_runs": None,
                "source_sweep": sweep_dir.name,
                "source_run_name": original_label,
                "varied_parameters": {"configuration": configuration_label},
            }
            aggregate_runs.append(aggregate_record)

    if not aggregate_runs:
        skipped_text = "; ".join(
            f"{item['run_name']} ({', '.join(item['reasons'])})" for item in skipped_runs
        )
        raise RuntimeError(
            "No runs with hist.pt were found for aggregate report generation. "
            f"Skipped runs: {skipped_text or 'none'}."
        )

    return aggregate_root, {
        "comparative_parameter": "configuration",
        "comparative_run": True,
        "aggregate_report": True,
        "source_root": str(aggregate_root),
        "source_sweeps": [str(path) for path in candidates],
        "runs": aggregate_runs,
    }


def _report_manifest_for_path(path: Path) -> tuple[Path, dict[str, object], bool]:
    sweep_dir = _resolve_sweep_dir(path)
    manifest_path = sweep_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        return sweep_dir, _read_json(manifest_path), False

    aggregate = _aggregate_manifest(path)
    if aggregate is not None:
        aggregate_root, manifest = aggregate
        return aggregate_root, manifest, True

    raise FileNotFoundError(f"Could not find {MANIFEST_FILENAME} in {sweep_dir}.")


def create_comparative_report(sweep_dir: Path, *, update_manifest: bool = False) -> dict[str, object]:
    """Create comparative figures and LaTeX report, using hist.pt as the run indicator."""
    sweep_dir, manifest, aggregate_report = _report_manifest_for_path(sweep_dir)
    complete_runs, skipped_runs = filter_complete_run_records(sweep_dir, manifest)
    if not complete_runs:
        skipped_text = "; ".join(
            f"{item['run_name']} ({', '.join(item['reasons'])})" for item in skipped_runs
        )
        raise RuntimeError(
            "No runs with hist.pt were found for report generation. "
            f"Skipped runs: {skipped_text or 'none'}."
        )

    analysis = save_comparative_outputs(sweep_dir, manifest)
    if update_manifest and not aggregate_report:
        manifest_path = sweep_dir / MANIFEST_FILENAME
        manifest["comparative_analysis"] = analysis
        _write_json(manifest_path, manifest)
    if aggregate_report:
        analysis["aggregate_report"] = True
        analysis["source_sweeps"] = manifest["source_sweeps"]
    return analysis


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate comparative_analysis/report_compare.tex from an existing "
            "comparative_* folder, or aggregate all completed comparative runs "
            "below a search root."
        )
    )
    parser.add_argument(
        "sweep_dir",
        nargs="?",
        type=Path,
        help=(
            "Path to a comparative_* folder, its comparative_summary.json, or its "
            "comparative_analysis folder. A parent folder with multiple comparative_* "
            "children creates one aggregate report with all completed runs. When "
            "omitted, --search-root is used."
        ),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=DEFAULT_SEARCH_ROOT,
        help="Root used when sweep_dir is omitted.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write the regenerated comparative_analysis metadata back to comparative_summary.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sweep_dir = args.search_root if args.sweep_dir is None else args.sweep_dir
    analysis = create_comparative_report(sweep_dir, update_manifest=args.update_manifest)

    resolved_sweep_dir, _manifest, aggregate_report = _report_manifest_for_path(sweep_dir)
    report_path = resolved_sweep_dir / str(analysis["report_compare_tex"])
    skipped = analysis.get("skipped_incomplete_runs", [])

    print(f"{'Aggregate root' if aggregate_report else 'Sweep folder'}: {resolved_sweep_dir}")
    print(f"Compared runs with hist.pt: {analysis['completed_runs']}")
    if skipped:
        print("Skipped incomplete runs:")
        for item in skipped:
            print(f"  - {item['run_name']}: {', '.join(item['reasons'])}")
    print(f"Report written: {report_path}")
    print(f"Expected filename: {REPORT_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
