from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import common_parser, first_existing_column, iter_files, processed_dir, raw_dir, read_table, require_files, write_metadata


DOMAIN = "rag"


def normalize_frame(frame: pd.DataFrame) -> list[dict]:
    question_col = first_existing_column(frame, ("question", "query", "input"))
    answer_col = first_existing_column(frame, ("answers", "answer", "target", "output"))
    context_col = first_existing_column(frame, ("contexts", "context", "passages", "paragraphs", "documents"))
    support_col = first_existing_column(frame, ("supporting_facts", "supporting", "support_facts", "supporting_passages"))
    if question_col is None:
        raise ValueError("RAG table needs a question/query column")
    examples = []
    for idx, row in frame.iterrows():
        answer = row[answer_col] if answer_col else []
        if not isinstance(answer, list):
            answer = [answer]
        contexts = row[context_col] if context_col else []
        if not isinstance(contexts, list):
            contexts = [contexts] if pd.notna(contexts) else []
        example = {
            "id": str(row.get("id", idx)),
            "question": row[question_col],
            "answers": answer,
            "contexts": contexts,
        }
        if support_col is not None and row.get(support_col) is not None:
            example["supporting_facts"] = row[support_col]
        examples.append(example)
    return examples


def materialize(dataset: str, raw_root: Path, processed_root: Path) -> None:
    raw = raw_dir(raw_root, DOMAIN, dataset)
    files = list(iter_files(raw, (".json", ".jsonl", ".json.gz", ".jsonl.gz", ".csv")))
    require_files(files, raw)
    examples = []
    used = []
    for file in files:
        try:
            examples.extend(normalize_frame(read_table(file)))
            used.append(file)
        except Exception as exc:
            print(f"skip {file}: {exc}")
    require_files(used, raw)
    out = processed_dir(processed_root, DOMAIN, dataset)
    with (out / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    write_metadata(
        out,
        domain=DOMAIN,
        dataset=dataset,
        source_files=used,
        rows=len(examples),
        schema={"task": "rag_long_context_evidence_selection", "primary": "examples.jsonl"},
    )
    print(f"wrote {out / 'examples.jsonl'} ({len(examples)} rows)")


def main() -> None:
    args = common_parser("Materialize RAG evidence-selection datasets.").parse_args()
    materialize(args.dataset, args.raw_root, args.processed_root)


if __name__ == "__main__":
    main()
