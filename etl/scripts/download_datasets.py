#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external"


@dataclass(frozen=True)
class Source:
    kind: str
    url: str
    name: str
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    name: str
    sources: tuple[Source, ...]
    note: str = ""


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "ctdg",
        "wikipedia",
        (Source("url", "https://snap.stanford.edu/jodie/wikipedia.csv", "wikipedia.csv"),),
    ),
    DatasetSpec(
        "ctdg",
        "reddit",
        (Source("url", "https://snap.stanford.edu/jodie/reddit.csv", "reddit.csv"),),
        "Large download (about 2.4 GB). The downloader can resume a partial .part file.",
    ),
    DatasetSpec(
        "ctdg",
        "mooc",
        (Source("url", "https://snap.stanford.edu/jodie/mooc.csv", "mooc.csv"),),
    ),
    DatasetSpec(
        "ctdg",
        "lastfm",
        (Source("url", "https://snap.stanford.edu/jodie/lastfm.csv", "lastfm.csv"),),
    ),
    DatasetSpec("ctdg", "starcraft", (), "Destructive CTDG setting; place raw files manually if not public."),
    DatasetSpec("ctdg", "lanl", (), "LANL often requires separate access; place raw files manually."),
    DatasetSpec(
        "ctdg",
        "actmooc",
        (Source("tar", "https://snap.stanford.edu/data/act-mooc.tar.gz", "act-mooc.tar.gz"),),
        "Official Stanford SNAP MOOC actions, features, and dropout labels.",
    ),
    DatasetSpec(
        "ctdg",
        "retailrocket",
        (
            Source(
                "zip",
                "https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset",
                "retailrocket.zip",
                ("events.csv",),
            ),
        ),
        "RetailRocket user-item events; only events.csv is extracted.",
    ),
    DatasetSpec(
        "mtpp",
        "stackoverflow",
        (
            Source(
                "huggingface",
                "easytpp/stackoverflow",
                "easytpp_stackoverflow",
            ),
        ),
    ),
    DatasetSpec(
        "mtpp",
        "retweets",
        (
            Source(
                "huggingface",
                "easytpp/retweet",
                "easytpp_retweet",
            ),
        ),
    ),
    DatasetSpec(
        "stpp",
        "earthquake",
        (
            Source(
                "git",
                "https://github.com/ss15859/EarthquakeNPP.git",
                "EarthquakeNPP",
                (
                    "Datasets/ComCat/ComCat_catalog.csv",
                    "Datasets/QTM/SanJac_catalog.csv",
                    "Datasets/QTM/SaltonSea_catalog.csv",
                    "Datasets/SCEDC/SCEDC_catalog.csv",
                    "Datasets/WHITE/WHITE_catalog.csv",
                ),
            ),
        ),
    ),
    DatasetSpec(
        "stpp",
        "gowalla",
        (
            Source(
                "url",
                "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
                "loc-gowalla_totalCheckins.txt.gz",
            ),
        ),
    ),
    DatasetSpec(
        "tkg",
        "icews14",
        (Source("git", "https://github.com/INK-USC/RE-Net.git", "RE-Net", ("data/ICEWS14/*",)),),
    ),
    DatasetSpec(
        "tkg",
        "icews18",
        (Source("git", "https://github.com/INK-USC/RE-Net.git", "RE-Net", ("data/ICEWS18/*",)),),
    ),
    DatasetSpec(
        "tkg",
        "gdelt",
        (Source("git", "https://github.com/INK-USC/RE-Net.git", "RE-Net", ("data/GDELT/*",)),),
    ),
    DatasetSpec(
        "rag",
        "natural_questions",
        (
            Source(
                "git",
                "https://github.com/google-research-datasets/natural-questions.git",
                "natural-questions",
                ("**/*.jsonl*", "**/*.json*", "**/*.gz"),
            ),
        ),
        "Natural Questions is large; official download may require the upstream instructions.",
    ),
    DatasetSpec(
        "rag",
        "hotpotqa",
        (
            Source(
                "huggingface",
                "hotpotqa/hotpot_qa",
                "hotpotqa_hotpot_qa",
            ),
        ),
        "Uses HuggingFace datasets when installed; otherwise prints the manual source.",
    ),
)


def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_or_pull(source: Source, external_root: Path, *, dry_run: bool) -> Path:
    target = external_root / source.name
    if target.exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], dry_run=dry_run)
    else:
        run(["git", "clone", "--depth", "1", source.url, str(target)], dry_run=dry_run)
    return target


def _download_with_resume(url: str, target: Path, *, retries: int = 3) -> None:
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "goodctdg-dataset-downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    print("server does not support resume; restarting partial download")
                    partial.unlink()
                    offset = 0
                total_header = response.headers.get("Content-Length")
                total = offset + int(total_header) if total_header else None
                mode = "ab" if offset else "wb"
                downloaded = offset
                last_report = time.monotonic()
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if time.monotonic() - last_report >= 5:
                            suffix = f"/{total / 1e6:.1f} MB" if total else ""
                            print(f"  {downloaded / 1e6:.1f} MB{suffix}", flush=True)
                            last_report = time.monotonic()
            partial.replace(target)
            return
        except Exception:
            if attempt == retries:
                raise
            print(f"download interrupted; retry {attempt}/{retries} from {partial.stat().st_size if partial.exists() else 0} bytes")
            time.sleep(attempt * 2)


def download_url(source: Source, raw_dir: Path, *, dry_run: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source.name
    if target.exists():
        print(f"exists: {target}")
        return 1
    print(f"download: {source.url} -> {target}")
    if not dry_run:
        _download_with_resume(source.url, target)
    return 1


def download_tar(source: Source, raw_dir: Path, *, dry_run: bool) -> int:
    count = download_url(source, raw_dir, dry_run=dry_run)
    archive = raw_dir / source.name
    if dry_run:
        print(f"would extract: {archive} -> {raw_dir}")
        return count
    with tarfile.open(archive, "r:*") as bundle:
        members = [member for member in bundle.getmembers() if member.isfile()]
        root = raw_dir.resolve()
        for member in members:
            target = (raw_dir / member.name).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
        bundle.extractall(raw_dir, members=members, filter="data")
    print(f"extracted: {archive} ({len(members)} files)")
    return count + len(members)


def download_zip(source: Source, raw_dir: Path, *, dry_run: bool) -> int:
    count = download_url(source, raw_dir, dry_run=dry_run)
    archive = raw_dir / source.name
    if dry_run:
        print(f"would extract: {archive} -> {raw_dir}")
        return count
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        root = raw_dir.resolve()
        for member in bundle.infolist():
            if member.is_dir():
                continue
            if source.patterns and Path(member.filename).name not in source.patterns:
                continue
            target = (raw_dir / Path(member.filename).name).resolve()
            if root not in target.parents:
                raise ValueError(f"unsafe archive path: {member.filename}")
            with bundle.open(member) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            extracted += 1
            print(f"extracted: {target}")
    return count + extracted


def copy_matching_files(source_dir: Path, raw_dir: Path, patterns: tuple[str, ...], *, dry_run: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen: set[Path] = set()
    for pattern in patterns or ("**/*",):
        for path in source_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if any(part.startswith(".git") for part in path.parts):
                continue
            rel = path.relative_to(source_dir)
            target = raw_dir / rel
            print(f"copy: {path} -> {target}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            seen.add(path)
            copied += 1
    return copied


def download_huggingface(source: Source, raw_dir: Path, *, dry_run: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / "HUGGINGFACE_DATASET.txt"
    if not dry_run:
        marker.write_text(f"{source.url}\n", encoding="utf-8")
    try:
        import datasets  # type: ignore
    except Exception:
        print(f"HuggingFace datasets is not installed. Manual dataset id: {source.url}")
        return 0
    if dry_run:
        print(f"would load HuggingFace dataset: {source.url}")
        return 1
    if source.url == "hotpotqa/hotpot_qa":
        ds = datasets.load_dataset(source.url, "distractor")
    else:
        ds = datasets.load_dataset(source.url)
    for split, table in ds.items():
        out = raw_dir / f"{split}.jsonl"
        table.to_json(str(out), lines=True, force_ascii=False)
        print(f"wrote: {out}")
    return len(ds)


def write_manifest(spec: DatasetSpec, raw_dir: Path, copied: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "domain": spec.domain,
        "dataset": spec.name,
        "sources": [asdict(source) for source in spec.sources],
        "note": spec.note,
        "copied_files": copied,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def has_payload_files(raw_dir: Path) -> bool:
    ignored = {"manifest.json", "HUGGINGFACE_DATASET.txt"}
    return raw_dir.exists() and any(path.is_file() and path.name not in ignored for path in raw_dir.rglob("*"))


def materialize(domain: str, dataset: str, *, raw_root: Path, processed_root: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        f"etl.materialize.{domain}",
        "--dataset",
        dataset,
        "--raw-root",
        str(raw_root),
        "--processed-root",
        str(processed_root),
    ]
    run(cmd, dry_run=dry_run)


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs = list(DATASETS)
    if args.domain != "all":
        specs = [spec for spec in specs if spec.domain == args.domain]
    if args.dataset != "all":
        wanted = {name.strip().lower() for name in args.dataset.split(",")}
        specs = [spec for spec in specs if spec.name.lower() in wanted]
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="분야별 공개 데이터를 다운로드하고 전처리합니다.")
    parser.add_argument("--domain", default="all", choices=["all", "ctdg", "mtpp", "stpp", "tkg", "rag"])
    parser.add_argument("--dataset", default="all", help="Dataset name, comma list, or all.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--materialize", action="store_true", help="Run etl.materialize after download.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = selected_specs(args)
    if not specs:
        raise SystemExit("No matching datasets.")

    for spec in specs:
        raw_dir = args.raw_root / spec.domain / spec.name
        copied = 0
        print(f"\n== {spec.domain}/{spec.name} ==")
        if spec.note:
            print(f"note: {spec.note}")
        for source in spec.sources:
            if source.kind == "git":
                repo = clone_or_pull(source, args.external_root, dry_run=args.dry_run)
                copied += copy_matching_files(repo, raw_dir, source.patterns, dry_run=args.dry_run)
            elif source.kind == "url":
                copied += download_url(source, raw_dir, dry_run=args.dry_run)
            elif source.kind == "tar":
                copied += download_tar(source, raw_dir, dry_run=args.dry_run)
            elif source.kind == "zip":
                copied += download_zip(source, raw_dir, dry_run=args.dry_run)
            elif source.kind == "huggingface":
                copied += download_huggingface(source, raw_dir, dry_run=args.dry_run)
            else:
                raise ValueError(f"unknown source kind: {source.kind}")
        if not spec.sources:
            raw_dir.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            write_manifest(spec, raw_dir, copied)
        if args.materialize and (copied > 0 or has_payload_files(raw_dir)):
            try:
                materialize(
                    spec.domain,
                    spec.name,
                    raw_root=args.raw_root,
                    processed_root=args.processed_root,
                    dry_run=args.dry_run,
                )
            except subprocess.CalledProcessError as exc:
                if len(specs) == 1:
                    raise
                print(f"skip materialize: {spec.domain}/{spec.name} failed with exit code {exc.returncode}")
        elif args.materialize:
            print(f"skip materialize: no raw files for {spec.domain}/{spec.name}")


if __name__ == "__main__":
    main()
