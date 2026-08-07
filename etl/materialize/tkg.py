from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import common_parser, iter_files, processed_dir, raw_dir, require_files, write_metadata


DOMAIN = "tkg"


def normalize_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if frame.shape[1] < 3:
        raise ValueError("TKG table needs at least head, relation, tail columns")
    cols = list(frame.columns)
    head, relation, tail = cols[:3]
    timestamp = cols[3] if frame.shape[1] >= 4 else None
    out = pd.DataFrame(
        {
            "head": frame[head].astype(str),
            "relation": frame[relation].astype(str),
            "tail": frame[tail].astype(str),
            "timestamp": frame[timestamp].astype(str) if timestamp else split,
            "split": split,
        }
    )
    return out


def split_name(path: Path) -> str:
    name = path.name.lower()
    if "valid" in name or "val" in name:
        return "val"
    if "test" in name:
        return "test"
    return "train"


def is_fact_file(path: Path) -> bool:
    name = path.name.lower()
    return name in {"train.txt", "valid.txt", "val.txt", "test.txt", "train.tsv", "valid.tsv", "val.tsv", "test.tsv", "train.csv", "valid.csv", "val.csv", "test.csv"}


def materialize(dataset: str, raw_root: Path, processed_root: Path) -> None:
    raw = raw_dir(raw_root, DOMAIN, dataset)
    files = [file for file in iter_files(raw, (".txt", ".tsv", ".csv")) if is_fact_file(file)]
    require_files(files, raw)
    frames = []
    used = []
    for file in files:
        try:
            frame = pd.read_csv(file, sep=r"\s+|\t|,", engine="python", header=None)
            frames.append(normalize_frame(frame, split_name(file)))
            used.append(file)
        except Exception as exc:
            print(f"skip {file}: {exc}")
    require_files(used, raw)
    facts = pd.concat(frames, ignore_index=True)
    out = processed_dir(processed_root, DOMAIN, dataset)
    facts.to_csv(out / "facts.csv", index=False)
    for split, split_frame in facts.groupby("split"):
        split_frame.to_csv(out / f"{split}.csv", index=False)
    write_metadata(
        out,
        domain=DOMAIN,
        dataset=dataset,
        source_files=used,
        rows=len(facts),
        schema={"task": "temporal_knowledge_graph_forecasting", "primary": "facts.csv"},
    )
    print(f"wrote {out / 'facts.csv'} ({len(facts)} rows)")


def main() -> None:
    args = common_parser("Materialize TKG forecasting datasets.").parse_args()
    materialize(args.dataset, args.raw_root, args.processed_root)


if __name__ == "__main__":
    main()
