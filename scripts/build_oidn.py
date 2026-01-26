from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BACKENDS = {"cpu", "cuda", "hip", "sycl", "metal"}


@dataclass(frozen=True)
class BuildConfig:
    source_dir: Path
    build_dir: Path
    install_dir: Path
    build_type: str
    backends: set[str]
    cmake_vars: list[str]
    stage: bool
    stage_package_root: Path
    stage_platform: str | None
    stage_arch: str | None


def _normalize_backends(value: str) -> set[str]:
    items = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not items:
        items = {"cpu"}
    items.add("cpu")
    unknown = items - SUPPORTED_BACKENDS
    if unknown:
        raise ValueError(f"Unknown backends: {', '.join(sorted(unknown))}")
    return items


def _check_lfs_weights(source_dir: Path) -> None:
    weights_file = source_dir / "weights" / "rt_hdr.tza"
    if not weights_file.exists():
        return
    with weights_file.open("rb") as handle:
        header = handle.read(200)
    lfs_marker = b"version https://git-lfs.github.com/spec/v1"
    if header.startswith(lfs_marker):
        raise RuntimeError(
            "OIDN weights look like Git LFS pointers. Run `git lfs install` and "
            "`git submodule update --init --recursive` to fetch the real files."
        )


def _cmake_device_vars(backends: Iterable[str]) -> list[str]:
    enabled = set(backends)
    return [
        f"OIDN_DEVICE_CPU={'ON' if 'cpu' in enabled else 'OFF'}",
        f"OIDN_DEVICE_CUDA={'ON' if 'cuda' in enabled else 'OFF'}",
        f"OIDN_DEVICE_HIP={'ON' if 'hip' in enabled else 'OFF'}",
        f"OIDN_DEVICE_SYCL={'ON' if 'sycl' in enabled else 'OFF'}",
        f"OIDN_DEVICE_METAL={'ON' if 'metal' in enabled else 'OFF'}",
        "OIDN_APPS=OFF",
    ]


def _check_metal_toolchain() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Metal backend is only supported on macOS.")
    try:
        result = subprocess.run(
            ["xcrun", "--sdk", "macosx", "--find", "metal"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("xcrun is missing. Install Xcode Command Line Tools.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Metal toolchain not found. Run `xcodebuild -downloadComponent MetalToolchain` " "and try again."
        ) from exc
    if not result.stdout.strip():
        raise RuntimeError(
            "Metal toolchain not found. Run `xcodebuild -downloadComponent MetalToolchain` " "and try again."
        )


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _build_oidn(cfg: BuildConfig) -> None:
    _check_lfs_weights(cfg.source_dir)
    if cfg.source_dir not in cfg.build_dir.parents:
        raise RuntimeError(
            "build_dir must be inside the OIDN source directory because the "
            "upstream build script configures CMake with '..'. Use "
            "`--build-dir oidn_cpp/build` or omit the flag to use the default."
        )
    if "metal" in cfg.backends:
        _check_metal_toolchain()
    cfg.build_dir.mkdir(parents=True, exist_ok=True)
    cfg.install_dir.mkdir(parents=True, exist_ok=True)

    build_script = cfg.source_dir / "scripts" / "build.py"
    if not build_script.exists():
        raise RuntimeError(f"OIDN build script not found at {build_script}")

    cmd = [
        sys.executable,
        str(build_script),
        "install",
        "--build_dir",
        str(cfg.build_dir),
        "--install_dir",
        str(cfg.install_dir),
        "--config",
        cfg.build_type,
    ]

    for var in _cmake_device_vars(cfg.backends):
        cmd.extend(["-D", var])

    for var in cfg.cmake_vars:
        cmd.extend(["-D", var])

    _run(cmd, cwd=cfg.source_dir)

    if cfg.stage:
        stage_script = Path(__file__).resolve().parent / "stage_oidn_libs.py"
        stage_cmd = [
            sys.executable,
            str(stage_script),
            "--install-root",
            str(cfg.install_dir),
            "--package-root",
            str(cfg.stage_package_root),
        ]
        if cfg.stage_platform:
            stage_cmd.extend(["--platform", cfg.stage_platform])
        if cfg.stage_arch:
            stage_cmd.extend(["--arch", cfg.stage_arch])
        _run(stage_cmd, cwd=cfg.source_dir)


def _parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build OIDN from the submodule.")
    parser.add_argument(
        "--source",
        default="oidn_cpp",
        help="Path to the OIDN submodule source directory.",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="Build directory for OIDN (relative to --source when not absolute).",
    )
    parser.add_argument(
        "--install-dir",
        default="install",
        help="Install directory for OIDN build outputs (relative to --source).",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        choices=["Release", "RelWithDebInfo", "Debug"],
        help="CMake build type.",
    )
    parser.add_argument(
        "--backends",
        default="cpu",
        help="Comma-separated list: cpu,cuda,hip,sycl,metal.",
    )
    parser.add_argument(
        "-D",
        dest="cmake_vars",
        action="append",
        default=[],
        help="Additional CMake cache entries to forward (e.g. OIDN_DEVICE_CUDA_API=Driver).",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="Stage built shared libraries into src/oidn/lib.*.",
    )
    parser.add_argument(
        "--stage-package-root",
        default="src/oidn",
        help="Package root to receive staged libraries.",
    )
    parser.add_argument(
        "--stage-platform",
        default=None,
        help="Override platform tag when staging (linux|macos|win).",
    )
    parser.add_argument(
        "--stage-arch",
        default=None,
        help="Override arch tag when staging (x64|aarch64).",
    )

    args = parser.parse_args()
    source_dir = Path(args.source).resolve()
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = source_dir / build_dir
    install_dir = Path(args.install_dir)
    if not install_dir.is_absolute():
        install_dir = source_dir / install_dir
    stage_package_root = Path(args.stage_package_root)
    if not stage_package_root.is_absolute():
        stage_package_root = Path.cwd() / stage_package_root
    return BuildConfig(
        source_dir=source_dir,
        build_dir=build_dir.resolve(),
        install_dir=install_dir.resolve(),
        build_type=args.build_type,
        backends=_normalize_backends(args.backends),
        cmake_vars=list(args.cmake_vars),
        stage=args.stage,
        stage_package_root=stage_package_root.resolve(),
        stage_platform=args.stage_platform,
        stage_arch=args.stage_arch,
    )


def main() -> None:
    cfg = _parse_args()
    _build_oidn(cfg)


if __name__ == "__main__":
    main()
