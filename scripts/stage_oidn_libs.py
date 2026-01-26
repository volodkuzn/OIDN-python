from __future__ import annotations

import argparse
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

SHARED_EXTENSIONS = (".so", ".dylib", ".dll")


@dataclass(frozen=True)
class StageConfig:
    install_root: Path
    package_root: Path
    platform_tag: str
    arch_tag: str
    clean: bool


def _normalize_platform(raw: str) -> str:
    system = raw.lower()
    if system.startswith("darwin") or system == "macos":
        return "macos"
    if system.startswith("win"):
        return "win"
    if system.startswith("linux"):
        return "linux"
    raise ValueError(f"Unsupported platform: {raw}")


def _normalize_arch(raw: str) -> str:
    machine = raw.lower()
    if machine in {"x86_64", "amd64", "x64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    return machine


def _collect_libs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    matches: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if (
            path.suffix in SHARED_EXTENSIONS
            or any(path.name.endswith(ext) for ext in SHARED_EXTENSIONS)
            or ".so." in path.name
        ):
            matches.append(path)
    return matches


def _stage_libs(cfg: StageConfig) -> None:
    lib_dirs = [
        cfg.install_root / "lib",
        cfg.install_root / "lib64",
        cfg.install_root / "bin",
    ]
    libs = []
    for lib_dir in lib_dirs:
        libs.extend(_collect_libs(lib_dir))

    if not libs:
        raise RuntimeError(f"No shared libraries found under {cfg.install_root}")

    dest_dir = cfg.package_root / f"lib.{cfg.platform_tag}.{cfg.arch_tag}"
    if cfg.clean and dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for lib in libs:
        shutil.copy2(lib, dest_dir / lib.name)


def _parse_args() -> StageConfig:
    parser = argparse.ArgumentParser(description="Stage OIDN shared libraries.")
    parser.add_argument(
        "--install-root",
        default="build/oidn/install",
        help="OIDN install root produced by the build script.",
    )
    parser.add_argument(
        "--package-root",
        default="src/oidn",
        help="Package root where lib.* directories live.",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="Override platform tag (linux|macos|win).",
    )
    parser.add_argument(
        "--arch",
        default=None,
        help="Override architecture tag (x64|aarch64).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the destination lib directory before staging.",
    )

    args = parser.parse_args()
    platform_tag = _normalize_platform(args.platform or platform.system())
    arch_tag = _normalize_arch(args.arch or platform.machine())
    return StageConfig(
        install_root=Path(args.install_root).resolve(),
        package_root=Path(args.package_root).resolve(),
        platform_tag=platform_tag,
        arch_tag=arch_tag,
        clean=args.clean,
    )


def main() -> None:
    cfg = _parse_args()
    _stage_libs(cfg)


if __name__ == "__main__":
    main()
