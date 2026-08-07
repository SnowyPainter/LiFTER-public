from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import chronological_split, common_parser, first_existing_column, iter_files, processed_dir, raw_dir, read_table, require_files, write_metadata


DOMAIN = "stpp"


def normalize_frame(frame: pd.DataFrame, *, source_name: str | None = None) -> pd.DataFrame:
    if frame.shape[1] == 5 and first_existing_column(frame, ("timestamp", "time", "ts")) is None:
        frame = frame.copy()
        frame.columns = ["user", "timestamp", "lat", "lon", "location_id"]

    user = first_existing_column(frame, ("user", "user_id", "sequence_id"))
    ts = first_existing_column(frame, ("timestamp", "time", "ts", "check_in_time", "datetime"))
    # EarthquakeNPP catalogs explicitly recommend projected x/y coordinates for
    # spatial models; mobility datasets generally provide latitude/longitude.
    lat = first_existing_column(frame, ("x", "lat", "latitude"))
    lon = first_existing_column(frame, ("y", "lon", "lng", "longitude"))
    mark = first_existing_column(frame, ("mark", "event_type", "location_id", "place_id", "type"))
    if ts is None or lat is None or lon is None:
        if frame.shape[1] >= 5:
            user, ts, lat, lon, mark = frame.columns[:5]
        elif frame.shape[1] >= 3:
            ts, lat, lon = frame.columns[:3]
        else:
            raise ValueError("STPP table needs timestamp, latitude, longitude columns")
    numeric_time = pd.to_numeric(frame[ts], errors="coerce")
    parsed_dt = pd.to_datetime(frame[ts], errors="coerce", utc=True)
    parsed_time = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid_dt = parsed_dt.notna()
    parsed_time.loc[valid_dt] = parsed_dt.loc[valid_dt].astype("int64") / 1.0e9
    timestamp = numeric_time.where(numeric_time.notna(), parsed_time)
    if user:
        sequence_id = frame[user].astype(str)
    elif source_name:
        sequence_id = source_name
    else:
        sequence_id = "sequence_0"

    mark_values = frame[mark].astype(str) if mark else "event"

    out = pd.DataFrame(
        {
            "sequence_id": sequence_id,
            "timestamp": timestamp,
            "lat": pd.to_numeric(frame[lat], errors="coerce"),
            "lon": pd.to_numeric(frame[lon], errors="coerce"),
            "mark": mark_values,
        }
    )
    return out.dropna(subset=["timestamp", "lat", "lon"])


def materialize(dataset: str, raw_root: Path, processed_root: Path) -> None:
    raw = raw_dir(raw_root, DOMAIN, dataset)
    files = list(iter_files(raw, (".csv", ".txt", ".tsv", ".json", ".jsonl", ".gz")))
    require_files(files, raw)
    frames = []
    used = []
    for file in files:
        try:
            source_name = file.parent.name if file.parent != raw else file.stem
            frames.append(normalize_frame(read_table(file), source_name=source_name))
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
        schema={"task": "spatio_temporal_point_process", "primary": "events.csv"},
    )
    print(f"wrote {out / 'events.csv'} ({len(events)} rows)")


def main() -> None:
    args = common_parser("Materialize STPP datasets.").parse_args()
    materialize(args.dataset, args.raw_root, args.processed_root)


if __name__ == "__main__":
    main()
