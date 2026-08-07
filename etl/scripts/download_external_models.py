#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external"


@dataclass(frozen=True)
class ModelRepo:
    domain: str
    models: tuple[str, ...]
    url: str
    name: str
    note: str = ""
    pip: tuple[str, ...] = ()


MODEL_REPOS: tuple[ModelRepo, ...] = (
    ModelRepo(
        "ctdg",
        ("TGAT", "DyGFormer", "GraphMixer"),
        "https://github.com/yule-BUAA/DyGLib.git",
        "DyGLib",
        "Unified dynamic graph library used as reference implementations.",
    ),
    ModelRepo(
        "ctdg",
        ("TGAT",),
        "https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs.git",
        "TGAT",
    ),
    ModelRepo(
        "mtpp",
        ("SAHP", "THP", "AttNHP"),
        "https://github.com/ant-research/EasyTemporalPointProcess.git",
        "EasyTemporalPointProcess",
    ),
    ModelRepo(
        "stpp",
        ("Transformer-STPP",),
        "https://github.com/ss15859/EarthquakeNPP.git",
        "EarthquakeNPP",
        "Reference source for earthquake neural point process experiments.",
    ),
    ModelRepo(
        "stpp",
        ("NSTPP",),
        "https://github.com/facebookresearch/neural_stpp.git",
        "neural_stpp",
        "Official reference for Neural Spatio-Temporal Point Processes.",
        ("torchdiffeq",),
    ),
    ModelRepo(
        "stpp",
        ("DeepSTPP",),
        "https://github.com/rose-stl-lab/deepstpp.git",
        "deepstpp",
        "Official reference for Deep Spatiotemporal Point Process.",
        ("torchdiffeq",),
    ),
    ModelRepo(
        "tkg",
        ("RE-NET",),
        "https://github.com/INK-USC/RE-Net.git",
        "RE-Net",
        "Requires DGL at runtime for the official RGCN aggregator.",
        ("dgl==1.1.3",),
    ),
    ModelRepo(
        "tkg",
        ("xERTE",),
        "https://github.com/TemporalKGTeam/xERTE.git",
        "xERTE",
        "Best-effort public xERTE source; update URL here if using another fork.",
    ),
    ModelRepo(
        "tkg",
        ("CyGNet",),
        "https://github.com/CunchaoZ/CyGNet.git",
        "CyGNet",
        "Best-effort public CyGNet source; update URL here if using another fork.",
    ),
    ModelRepo(
        "rag",
        ("FiD",),
        "https://github.com/facebookresearch/FiD.git",
        "FiD",
        "Requires HuggingFace transformers runtime dependencies.",
        ("transformers",),
    ),
    ModelRepo(
        "rag",
        ("LED",),
        "https://github.com/huggingface/transformers.git",
        "transformers",
        "Reference implementation for LED/LongformerEncoderDecoder.",
        ("transformers",),
    ),
)


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def clone_or_pull(repo: ModelRepo, root: Path, *, dry_run: bool) -> Path:
    target = root / repo.name
    if target.exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], dry_run=dry_run)
    else:
        run(["git", "clone", "--depth", "1", repo.url, str(target)], dry_run=dry_run)
    return target


def selected_repos(args: argparse.Namespace) -> list[ModelRepo]:
    repos = list(MODEL_REPOS)
    if args.domain != "all":
        repos = [repo for repo in repos if repo.domain == args.domain]
    if args.model != "all":
        wanted = {name.strip().lower() for name in args.model.split(",")}
        repos = [
            repo
            for repo in repos
            if repo.name.lower() in wanted or any(model.lower() in wanted for model in repo.models)
        ]
    return repos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="분야별 공개 모델 저장소를 선택해서 받습니다.")
    parser.add_argument("--domain", default="all", choices=["all", "ctdg", "mtpp", "stpp", "tkg", "rag"])
    parser.add_argument("--model", default="all", help="Model/repo name, comma list, or all.")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repos = selected_repos(args)
    if not repos:
        raise SystemExit("No matching model repositories.")
    args.external_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for repo in repos:
        print(f"\n== {repo.domain}: {', '.join(repo.models)} ==")
        if repo.note:
            print(f"note: {repo.note}")
        target = clone_or_pull(repo, args.external_root, dry_run=args.dry_run)
        manifest.append({**asdict(repo), "path": str(target)})
    if not args.dry_run:
        (args.external_root / "model_repos.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        deps = sorted({dep for repo in repos for dep in repo.pip})
        (args.external_root / "model_requirements.txt").write_text("\n".join(deps) + ("\n" if deps else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
