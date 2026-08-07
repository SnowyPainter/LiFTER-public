from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    chronological_split,
    common_parser,
    first_existing_column,
    iter_files,
    processed_dir,
    raw_dir,
    read_table,
    require_files,
    write_metadata,
)


DOMAIN = "mtpp"


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if {"time_since_start", "type_event"}.issubset(set(frame.columns)):
        rows = []
        for row_idx, row in frame.iterrows():
            seq_id = row.get("seq_idx", row_idx)
            times = row["time_since_start"]
            marks = row["type_event"]
            if not isinstance(times, list) or not isinstance(marks, list):
                continue
            for event_idx, (timestamp, mark) in enumerate(zip(times, marks, strict=False)):
                rows.append(
                    {
                        "sequence_id": str(seq_id),
                        "timestamp": timestamp,
                        "mark": str(mark),
                        "event_index": event_idx,
                    }
                )
        return pd.DataFrame(rows).dropna(subset=["timestamp"])

    seq = first_existing_column(frame, ("sequence_id", "seq_id", "user", "user_id", "case_id"))
    ts = first_existing_column(frame, ("timestamp", "time", "ts", "t"))
    mark = first_existing_column(frame, ("mark", "event_type", "type", "label", "item_id"))
    if ts is None:
        if frame.shape[1] < 1:
            raise ValueError("MTPP table needs at least a timestamp column")
        ts = frame.columns[0]
    out = pd.DataFrame(
        {
            "sequence_id": frame[seq].astype(str) if seq else "sequence_0",
            "timestamp": pd.to_numeric(frame[ts], errors="coerce"),
            "mark": frame[mark].astype(str) if mark else "event",
        }
    )
    return out.dropna(subset=["timestamp"])


def materialize(dataset: str, raw_root: Path, processed_root: Path) -> None:
    raw = raw_dir(raw_root, DOMAIN, dataset)
    files = list(iter_files(raw, (".csv", ".txt", ".tsv", ".json", ".jsonl", ".csv.gz", ".txt.gz", ".jsonl.gz")))
    require_files(files, raw)
    frames = []
    used = []
    for file in files:
        try:
            frames.append(normalize_frame(read_table(file)))
            used.append(file)
        except Exception as exc:
            print(f"skip {file}: {exc}")
    require_files(used, raw)
    events = chronological_split(pd.concat(frames, ignore_index=True))
    out = processed_dir(processed_root, DOMAIN, dataset)
    events.to_csv(out / "events.csv", index=False)
    write_metadata(
        out,
        domain=DOMAIN,
        dataset=dataset,
        source_files=used,
        rows=len(events),
        schema={"task": "marked_temporal_point_process", "primary": "events.csv"},
    )
    print(f"wrote {out / 'events.csv'} ({len(events)} rows)")


def main() -> None:
    args = common_parser("Materialize MTPP datasets.").parse_args()
    materialize(args.dataset, args.raw_root, args.processed_root)


if __name__ == "__main__":
    main()
