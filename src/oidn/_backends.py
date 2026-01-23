from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from oidn import _ffi
from oidn.constants import (
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_CUDA,
    DEVICE_TYPE_DEFAULT,
    DEVICE_TYPE_HIP,
    DEVICE_TYPE_METAL,
    DEVICE_TYPE_SYCL,
)


class Backend(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    HIP = "hip"
    SYCL = "sycl"
    METAL = "metal"

    @classmethod
    def parse(cls, value: Backend | str) -> Backend:
        if isinstance(value, Backend):
            return value
        if not isinstance(value, str):
            raise TypeError("backend must be a Backend or str")
        normalized = value.strip().lower()
        aliases = {"default": cls.AUTO, "auto": cls.AUTO}
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported backend: {value}")


@dataclass(frozen=True, slots=True)
class DeviceOptions:
    set_affinity: bool | None = None
    verbose: int | None = None
    num_threads: int | None = None
    extra_params: Mapping[str, int | bool] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendAvailability:
    backend: Backend
    available: bool
    reason: str | None = None


_BACKEND_ORDER = (
    Backend.CPU,
    Backend.CUDA,
    Backend.HIP,
    Backend.SYCL,
    Backend.METAL,
)

_DEVICE_TYPE_MAP = {
    Backend.AUTO: DEVICE_TYPE_DEFAULT,
    Backend.CPU: DEVICE_TYPE_CPU,
    Backend.SYCL: DEVICE_TYPE_SYCL,
    Backend.CUDA: DEVICE_TYPE_CUDA,
    Backend.HIP: DEVICE_TYPE_HIP,
    Backend.METAL: DEVICE_TYPE_METAL,
}

_DEVICE_LIBRARY_TOKEN = {
    Backend.CPU: "device_cpu",
    Backend.CUDA: "device_cuda",
    Backend.HIP: "device_hip",
    Backend.SYCL: "device_sycl",
    Backend.METAL: "device_metal",
}


def device_type_for_backend(backend: Backend | str) -> int:
    return _DEVICE_TYPE_MAP[Backend.parse(backend)]


def available_backends(*, check_runtime: bool = True) -> list[Backend]:
    return [
        backend
        for backend in _BACKEND_ORDER
        if is_backend_available(backend, check_runtime=check_runtime)
    ]


def is_backend_available(backend: Backend | str, *, check_runtime: bool = True) -> bool:
    return backend_availability(backend, check_runtime=check_runtime).available


def backend_availability(
    backend: Backend | str, *, check_runtime: bool = True
) -> BackendAvailability:
    resolved = Backend.parse(backend)
    if resolved is Backend.AUTO:
        for candidate in _BACKEND_ORDER:
            if is_backend_available(candidate, check_runtime=check_runtime):
                return BackendAvailability(resolved, True)
        return BackendAvailability(resolved, False, "No available backends detected.")

    lib_dir = _ffi.library_dir()
    if not _device_library_present(lib_dir, resolved):
        return BackendAvailability(resolved, False, "Device library not found.")

    if check_runtime:
        ok, reason = _check_python_runtime(resolved)
        if not ok:
            return BackendAvailability(resolved, False, reason)
        ok, reason = _probe_device(resolved)
        if not ok:
            return BackendAvailability(resolved, False, reason)

    return BackendAvailability(resolved, True)


def apply_device_options(device_handle: int, options: DeviceOptions) -> None:
    reserved = {"setAffinity", "verbose", "numThreads"}
    extra_keys = set(options.extra_params.keys())
    if reserved & extra_keys:
        raise ValueError("extra_params may not override reserved device options.")

    functions = _ffi.get_functions()

    if options.set_affinity is not None:
        functions.oidnSetDeviceBool(device_handle, b"setAffinity", options.set_affinity)
    if options.verbose is not None:
        functions.oidnSetDeviceInt(device_handle, b"verbose", options.verbose)
    if options.num_threads is not None:
        functions.oidnSetDeviceInt(device_handle, b"numThreads", options.num_threads)

    for name, value in options.extra_params.items():
        try:
            encoded = name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(f"Device option name must be ASCII: {name}") from exc
        if isinstance(value, bool):
            functions.oidnSetDeviceBool(device_handle, encoded, value)
        elif isinstance(value, int):
            functions.oidnSetDeviceInt(device_handle, encoded, value)
        else:
            raise TypeError(f"Unsupported option type for {name}: {type(value).__name__}")


def _device_library_present(lib_dir: Path, backend: Backend) -> bool:
    token = _DEVICE_LIBRARY_TOKEN[backend]
    for path in _iter_shared_libraries(lib_dir):
        if token in path.name.lower():
            return True
    return False


def _iter_shared_libraries(lib_dir: Path) -> list[Path]:
    if not lib_dir.exists():
        return []
    matches: list[Path] = []
    for path in lib_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith((".so", ".dylib", ".dll")) or ".so." in name:
            matches.append(path)
    return matches


def _check_python_runtime(backend: Backend) -> tuple[bool, str | None]:
    if backend is Backend.CPU:
        return True, None
    if backend is Backend.METAL:
        if sys.platform != "darwin":
            return False, "Metal backend requires macOS."
        return True, None
    if backend is Backend.SYCL:
        if importlib.util.find_spec("dpctl") is None:
            return False, "SYCL backend requires dpctl for array support."
        return True, None
    if backend in {Backend.CUDA, Backend.HIP}:
        ok, message, hip = _torch_runtime_status()
        if not ok:
            return False, message
        if backend is Backend.HIP and not hip:
            return False, "HIP backend requires a ROCm-enabled torch build."
        return True, None
    return True, None


def _torch_runtime_status() -> tuple[bool, str | None, bool]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return False, "CUDA/HIP backends require torch to be installed.", False

    is_available = getattr(torch, "cuda").is_available()
    if not is_available:
        return False, "torch.cuda.is_available() is False.", False

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    return True, None, hip_version is not None


def _probe_device(backend: Backend) -> tuple[bool, str | None]:
    functions = _ffi.get_functions()
    device_type = device_type_for_backend(backend)
    handle = _ffi.handle_from_ptr(functions.oidnNewDevice(device_type))
    if handle == 0:
        return False, "Failed to create device handle."
    functions.oidnCommitDevice(handle)
    code, message = _ffi.get_device_error(handle)
    functions.oidnReleaseDevice(handle)
    if code != 0:
        return False, message
    return True, None
