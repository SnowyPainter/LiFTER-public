from __future__ import annotations

from pathlib import Path
from typing import Any

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


DOMAIN = "ctdg"
JODIE_HEADER = "comma_separated_list_of_features"
CHUNK_SIZE = 100_000


def read_ctdg_table(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    if "comma_separated_list_of_features" not in frame.columns:
        return frame
    # JODIE-style CSVs have a short logical header followed by a variable-width
    # comma-separated feature vector.  pandas' default parser treats the first
    # field as an index when data rows are wider than the header, shifting
    # timestamp/label columns and silently corrupting the event stream.
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        for line in handle:
            if line.strip():
                rows.append(line.rstrip("\n\r").split(","))
    if not rows or min(len(row) for row in rows) < 4:
        return frame
    width = max(len(row) for row in rows)
    columns = ["user_id", "item_id", "timestamp", "state_label"] + [f"feature_{idx}" for idx in range(max(0, width - 4))]
    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, columns=columns)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    src = first_existing_column(frame, ("src", "source", "source_id", "u", "user", "user_id", "userid", "from"))
    dst = first_existing_column(frame, ("dst", "destination", "target", "target_id", "targetid", "i", "item", "item_id", "to"))
    ts = first_existing_column(frame, ("timestamp", "time", "ts", "t"))
    label = first_existing_column(frame, ("label", "state_label", "y"))
    if src is None or dst is None or ts is None:
        if frame.shape[1] < 3:
            raise ValueError("CTDG table needs at least source, destination, timestamp columns")
        src, dst, ts = frame.columns[:3]
    src_values = "src:" + frame[src].astype(str)
    dst_values = "dst:" + frame[dst].astype(str)
    node_index = pd.Index(pd.concat([src_values, dst_values], ignore_index=True).unique())
    node_map = pd.Series(range(len(node_index)), index=node_index)
    out = pd.DataFrame(
        {
            "src": src_values.map(node_map).astype("int64"),
            "dst": dst_values.map(node_map).astype("int64"),
            "timestamp": pd.to_numeric(frame[ts], errors="coerce"),
        }
    )
    out["label"] = pd.to_numeric(frame[label], errors="coerce").fillna(1.0) if label else 1.0
    feature_cols = [
        column
        for column in frame.columns
        if column not in {src, dst, ts, label}
        and str(column).lower() not in {"actionid", "action_id", "edge_id"}
    ]
    for idx, column in enumerate(feature_cols):
        out[f"feat_{idx}"] = pd.to_numeric(frame[column], errors="coerce")
    return out.dropna(subset=["src", "dst", "timestamp"])


def _count_data_rows(path: Path) -> int:
    opener: Any
    if path.name.lower().endswith(".gz"):
        import gzip

        opener = gzip.open
    else:
        opener = Path.open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return max(0, sum(1 for line in handle if line.strip()) - 1)


def _stream_jodie(path: Path, out: Path) -> dict[str, Any]:
    """Convert a large JODIE CSV without holding the full feature table in RAM."""

    total = _count_data_rows(path)
    train_end, val_end = int(total * 0.70), int(total * 0.85)
    node_map: dict[str, int] = {}
    train_nodes: set[int] = set()
    feature_count = 0
    written = 0
    first = True

    reader = pd.read_csv(path, header=None, skiprows=1, chunksize=CHUNK_SIZE, low_memory=False)
    for chunk in reader:
        if chunk.shape[1] < 4:
            raise ValueError(f"invalid JODIE row width in {path}: {chunk.shape[1]}")
        feature_count = max(feature_count, chunk.shape[1] - 4)

        def encode(prefix: str, values: pd.Series) -> pd.Series:
            encoded = []
            for value in values.astype(str):
                key = f"{prefix}:{value}"
                node_id = node_map.get(key)
                if node_id is None:
                    node_id = len(node_map)
                    node_map[key] = node_id
                encoded.append(node_id)
            return pd.Series(encoded, index=values.index, dtype="int64")

        base = pd.DataFrame(
            {
                "src": encode("src", chunk.iloc[:, 0]),
                "dst": encode("dst", chunk.iloc[:, 1]),
                "timestamp": pd.to_numeric(chunk.iloc[:, 2], errors="coerce"),
                "label": pd.to_numeric(chunk.iloc[:, 3], errors="coerce").fillna(1.0),
            }
        )
        features = chunk.iloc[:, 4:].apply(pd.to_numeric, errors="coerce").astype("float32")
        features.columns = [f"feat_{idx}" for idx in range(features.shape[1])]
        normalized = pd.concat([base, features], axis=1)

        positions = pd.Series(range(written, written + len(normalized)), index=normalized.index)
        normalized["split"] = "test"
        normalized.loc[positions < train_end, "split"] = "train"
        normalized.loc[(positions >= train_end) & (positions < val_end), "split"] = "val"
        train_mask = positions < train_end
        if bool(train_mask.any()):
            train_nodes.update(normalized.loc[train_mask, "src"].tolist())
            train_nodes.update(normalized.loc[train_mask, "dst"].tolist())
        normalized["is_inductive"] = (normalized["split"] != "train") & (
            ~normalized["src"].isin(train_nodes) | ~normalized["dst"].isin(train_nodes)
        )
        normalized = normalized.dropna(subset=["timestamp"])
        normalized.to_csv(out, mode="w" if first else "a", header=first, index=False)
        written += len(chunk)
        first = False
        print(f"materialize: {written}/{total} rows", flush=True)

    return {
        "rows": written,
        "num_nodes": len(node_map),
        "num_features": feature_count,
        "split": {"train": train_end, "val": val_end - train_end, "test": total - val_end},
    }


def _read_actmooc(raw: Path) -> tuple[pd.DataFrame, list[Path]]:
    actions = next(iter(raw.rglob("mooc_actions.tsv")), None)
    features = next(iter(raw.rglob("mooc_action_features.tsv")), None)
    labels = next(iter(raw.rglob("mooc_action_labels.tsv")), None)
    if not actions or not features or not labels:
        raise ValueError("actmooc requires mooc_actions.tsv, mooc_action_features.tsv, and mooc_action_labels.tsv")
    action_frame = pd.read_csv(actions, sep=r"\s+")
    feature_frame = pd.read_csv(features, sep=r"\s+")
    label_frame = pd.read_csv(labels, sep=r"\s+")
    key = first_existing_column(action_frame, ("ACTIONID", "action_id"))
    feature_key = first_existing_column(feature_frame, ("ACTIONID", "action_id"))
    if key is None or feature_key is None:
        raise ValueError("actmooc tables are missing ACTIONID")
    merged = action_frame.merge(feature_frame, left_on=key, right_on=feature_key, how="left")
    # The published label file contains exactly one row per action, but 15,116
    # ACTIONID values are duplicated while the row order remains aligned.  A
    # key join therefore inflates the graph.  SNAP's documented one-label-per-
    # action contract makes positional attachment the lossless interpretation.
    if len(label_frame) != len(action_frame):
        raise ValueError("actmooc labels are not row-aligned with actions")
    label = first_existing_column(label_frame, ("LABEL", "label"))
    if label is None:
        raise ValueError("actmooc labels are missing LABEL")
    merged["LABEL"] = label_frame[label].to_numpy()
    return merged, [actions, features, labels]


def _materialize_retailrocket(raw: Path, out: Path) -> tuple[int, list[Path]]:
    """Materialize one untyped Link stream while retaining action labels for audit."""
    source = raw / "events.csv"
    if not source.exists():
        raise ValueError("retailrocket requires events.csv")
    frame = pd.read_csv(source)
    required = {"timestamp", "visitorid", "event", "itemid"}
    if not required.issubset(frame.columns):
        raise ValueError(f"retailrocket events.csv is missing {sorted(required - set(frame))}")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    src_values = "src:" + frame["visitorid"].astype(str)
    dst_values = "dst:" + frame["itemid"].astype(str)
    node_index = pd.Index(pd.concat([src_values, dst_values], ignore_index=True).unique())
    node_map = pd.Series(range(len(node_index)), index=node_index)
    events = pd.DataFrame({
        "src": src_values.map(node_map).astype("int64"),
        "dst": dst_values.map(node_map).astype("int64"),
        "timestamp": pd.to_numeric(frame["timestamp"], errors="raise").astype("float64") / 1000.0,
        "label": 1.0,
        # Kept outside feat_* so the predictor cannot observe it. This column
        # is opened only by the predicate-recovery evaluator after training.
        "action_type": frame["event"].astype(str),
    })
    events = chronological_split(events)
    train_nodes = set(events.loc[events["split"] == "train", "src"]) | set(
        events.loc[events["split"] == "train", "dst"]
    )
    events["is_inductive"] = (events["split"] != "train") & (
        ~events["src"].isin(train_nodes) | ~events["dst"].isin(train_nodes)
    )
    events.to_csv(out / "events.csv", index=False)
    return len(events), [source]


def materialize(dataset: str, raw_root: Path, processed_root: Path) -> None:
    raw = raw_dir(raw_root, DOMAIN, dataset)
    files = list(iter_files(raw, (".csv", ".txt", ".tsv", ".csv.gz", ".txt.gz", ".tsv.gz")))
    require_files(files, raw)
    out = processed_dir(processed_root, DOMAIN, dataset)

    if dataset == "retailrocket":
        rows, used = _materialize_retailrocket(raw, out)
        write_metadata(
            out,
            domain=DOMAIN,
            dataset=dataset,
            source_files=used,
            rows=rows,
            schema={
                "task": "ctdg_link_prediction",
                "primary": "events.csv",
                "columns": ["src", "dst", "timestamp", "label", "action_type", "split", "is_inductive"],
                "observed_relation": "Link",
                "action_type_usage": "evaluation_only; excluded from feat_* model inputs",
            },
        )
        print(f"wrote {out / 'events.csv'} ({rows} rows)")
        return

    jodie_files = []
    for file in files:
        with file.open("rt", encoding="utf-8", errors="replace") as handle:
            if JODIE_HEADER in handle.readline():
                jodie_files.append(file)
    if len(jodie_files) == 1:
        stats = _stream_jodie(jodie_files[0], out / "events.csv")
        write_metadata(
            out,
            domain=DOMAIN,
            dataset=dataset,
            source_files=jodie_files,
            rows=stats["rows"],
            schema={
                "task": "ctdg_link_prediction",
                "primary": "events.csv",
                "columns": ["src", "dst", "timestamp", "label", "feat_*", "split", "is_inductive"],
                **{key: value for key, value in stats.items() if key != "rows"},
            },
        )
        print(f"wrote {out / 'events.csv'} ({stats['rows']} rows)")
        return

    frames = []
    used = []
    if dataset == "actmooc":
        frame, used = _read_actmooc(raw)
        frames.append(normalize_frame(frame))
    else:
        for file in files:
            try:
                frames.append(normalize_frame(read_ctdg_table(file)))
                used.append(file)
            except Exception as exc:
                print(f"skip {file}: {exc}")
    require_files(used, raw)
    events = chronological_split(pd.concat(frames, ignore_index=True))
    train_nodes = set(events.loc[events["split"] == "train", "src"]) | set(events.loc[events["split"] == "train", "dst"])
    events["is_inductive"] = (events["split"] != "train") & (
        ~events["src"].isin(train_nodes) | ~events["dst"].isin(train_nodes)
    )
    events.to_csv(out / "events.csv", index=False)
    write_metadata(
        out,
        domain=DOMAIN,
        dataset=dataset,
        source_files=used,
        rows=len(events),
        schema={
            "task": "ctdg_link_prediction",
            "primary": "events.csv",
            "columns": ["src", "dst", "timestamp", "label", "feat_*", "split", "is_inductive"],
            "num_nodes": int(pd.concat([events["src"], events["dst"]]).nunique()),
            "num_features": len([column for column in events if column.startswith("feat_")]),
        },
    )
    print(f"wrote {out / 'events.csv'} ({len(events)} rows)")


def main() -> None:
    args = common_parser("Materialize CTDG link-prediction datasets.").parse_args()
    materialize(args.dataset, args.raw_root, args.processed_root)


if __name__ == "__main__":
    main()
