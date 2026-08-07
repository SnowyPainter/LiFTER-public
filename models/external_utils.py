from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"


@contextmanager
def prepend_sys_path(path: str | Path) -> Iterator[None]:
    path_str = str(Path(path).resolve())
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def import_with_path(module_name: str, root: str | Path):
    with prepend_sys_path(root):
        return importlib.import_module(module_name)


def import_from_file(module_name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(module_name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def isolated_top_level_package(package_name: str, package_path: str | Path) -> Iterator[None]:
    """Temporarily bind a top-level package name to an external package path."""

    package_path = Path(package_path).resolve()
    backup = {name: module for name, module in sys.modules.items() if name == package_name or name.startswith(f"{package_name}.")}
    for name in list(backup):
        del sys.modules[name]

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        yield
    finally:
        for name in [name for name in sys.modules if name == package_name or name.startswith(f"{package_name}.")]:
            del sys.modules[name]
        sys.modules.update(backup)


__all__ = ["EXTERNAL_ROOT", "PROJECT_ROOT", "import_from_file", "import_with_path", "isolated_top_level_package", "prepend_sys_path"]
