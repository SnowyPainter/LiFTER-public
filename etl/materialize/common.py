from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    return parser


def raw_dir(raw_root: Path, domain: str, dataset: str) -> Path:
    return raw_root / domain / dataset


def processed_dir(processed_root: Path, domain: str, dataset: str) -> Path:
    path = processed_root / domain / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_files(path: Path, suffixes: tuple[str, ...]) -> Iterable[Path]:
    if not path.exists():
        return []
    return (
        file
        for file in sorted(path.rglob("*"))
        if file.is_file()
        and not file.name.startswith(".")
        and file.name != "manifest.json"
        and any(file.name.lower().endswith(suffix) for suffix in suffixes)
    )


def read_table(path: Path) -> pd.DataFrame:
    if path.name.lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return _read_table_handle(handle, path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return _read_table_handle(handle, path)


def _read_table_handle(handle, path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if path.suffix.lower() == ".jsonl" or name.endswith(".jsonl.gz"):
        return pd.read_json(handle, lines=True)
    if path.suffix.lower() == ".json" or name.endswith(".json.gz"):
        return pd.read_json(handle)
    if path.suffix.lower() in {".tsv", ".txt"} or name.endswith((".tsv.gz", ".txt.gz")):
        frame = pd.read_csv(handle, sep=None, engine="python")
        if frame.shape[1] > 1:
            return frame
        handle.seek(0)
        return pd.read_csv(handle, sep=r"\s+|\t|,", engine="python", header=None)
    return pd.read_csv(handle)


def first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def write_metadata(out_dir: Path, *, domain: str, dataset: str, source_files: list[Path], rows: int, schema: dict) -> None:
    metadata = {
        "domain": domain,
        "dataset": dataset,
        "rows": int(rows),
        "source_files": [str(path) for path in source_files],
        "schema": schema,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def chronological_split(frame: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    frame = frame.sort_values(time_col).reset_index(drop=True)
    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    split = pd.Series("test", index=frame.index)
    split.iloc[:train_end] = "train"
    split.iloc[train_end:val_end] = "val"
    frame["split"] = split
    return frame


def require_files(files: list[Path], raw: Path) -> None:
    if not files:
        raise SystemExit(f"No usable raw files found in {raw}")
