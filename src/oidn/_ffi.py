from __future__ import annotations

import ctypes
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence, cast

DeviceHandle = int
FilterHandle = int
DataPointer = int

NewDeviceFn = Callable[[int], int | None]
CommitDeviceFn = Callable[[int], None]
GetDeviceErrorFn = Callable[[int, ctypes.POINTER(ctypes.c_char_p)], int]
ReleaseDeviceFn = Callable[[int], None]
RetainDeviceFn = Callable[[int], None]
SetDeviceBoolFn = Callable[[int, bytes, bool], None]
SetDeviceIntFn = Callable[[int, bytes, int], None]
GetDeviceBoolFn = Callable[[int, bytes], bool]
GetDeviceIntFn = Callable[[int, bytes], int]

NewFilterFn = Callable[[int, bytes], int | None]
SetSharedFilterImageFn = Callable[[int, bytes, DataPointer, int, int, int, int, int, int], None]
UnsetFilterImageFn = Callable[[int, bytes], None]
SetSharedFilterDataFn = Callable[[int, bytes, DataPointer, int], None]
UpdateFilterDataFn = Callable[[int, bytes], None]
UnsetFilterDataFn = Callable[[int, bytes], None]
GetFilterBoolFn = Callable[[int, bytes], bool]
GetFilterFloatFn = Callable[[int, bytes], float]
GetFilterIntFn = Callable[[int, bytes], int]
SetFilterBoolFn = Callable[[int, bytes, bool], None]
SetFilterFloatFn = Callable[[int, bytes, float], None]
SetFilterIntFn = Callable[[int, bytes, int], None]
CommitFilterFn = Callable[[int], None]
ExecuteFilterFn = Callable[[int], None]
ReleaseFilterFn = Callable[[int], None]
RetainFilterFn = Callable[[int], None]

_ERROR_MESSAGES = {
    0: "No error occurred.",
    1: "An unknown error occurred: ",
    2: "An invalid argument was specified: ",
    3: "The operation is not allowed: ",
    4: "No enough memory to execute the operation: ",
    5: "The hardware (e.g., CPU) is not supported: ",
    6: "The operation was cancelled by the user: ",
}

_SHARED_EXTENSIONS = (".so", ".dylib", ".dll")
_VERSION_RE = re.compile(r"(\\d+\\.\\d+\\.\\d+)")


@dataclass(frozen=True)
class BoundFunctions:
    oidnNewDevice: NewDeviceFn
    oidnCommitDevice: CommitDeviceFn
    oidnGetDeviceError: GetDeviceErrorFn
    oidnReleaseDevice: ReleaseDeviceFn
    oidnRetainDevice: RetainDeviceFn
    oidnSetDeviceBool: SetDeviceBoolFn
    oidnSetDeviceInt: SetDeviceIntFn
    oidnGetDeviceBool: GetDeviceBoolFn
    oidnGetDeviceInt: GetDeviceIntFn

    oidnNewFilter: NewFilterFn
    oidnSetSharedFilterImage: SetSharedFilterImageFn
    oidnUnsetFilterImage: UnsetFilterImageFn
    oidnSetSharedFilterData: SetSharedFilterDataFn
    oidnUpdateFilterData: UpdateFilterDataFn
    oidnUnsetFilterData: UnsetFilterDataFn
    oidnGetFilterBool: GetFilterBoolFn
    oidnGetFilterFloat: GetFilterFloatFn
    oidnGetFilterInt: GetFilterIntFn
    oidnSetFilterBool: SetFilterBoolFn
    oidnSetFilterFloat: SetFilterFloatFn
    oidnSetFilterInt: SetFilterIntFn
    oidnCommitFilter: CommitFilterFn
    oidnExecuteFilter: ExecuteFilterFn
    oidnReleaseFilter: ReleaseFilterFn
    oidnRetainFilter: RetainFilterFn


_FUNCTIONS: BoundFunctions | None = None
_LOADED_LIBRARY_PATH: Path | None = None


class _CFuncLike(Protocol):
    argtypes: list[object]
    restype: object | None

    def __call__(self, *args: object) -> object: ...


def _package_root() -> Path:
    return Path(__file__).resolve().parent


def _platform_tag() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "win"
    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def _arch_tag() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "x64"}:
        return "x64"
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    return machine


def _lib_dir() -> Path:
    return _package_root() / f"lib.{_platform_tag()}.{_arch_tag()}"


def _is_shared_library(path: Path) -> bool:
    if path.suffix in _SHARED_EXTENSIONS:
        return True
    return ".so." in path.name


def _library_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    lowered = name.lower()
    if "tbb" in lowered or "stdc++" in lowered:
        return (0, name)
    if "openimagedenoise_core" in lowered:
        return (1, name)
    if "openimagedenoise_device" in lowered:
        return (2, name)
    if "openimagedenoise" in lowered:
        return (3, name)
    return (4, name)


def _find_shared_libraries(lib_dir: Path) -> list[Path]:
    if not lib_dir.exists():
        raise RuntimeError(f"OIDN library directory not found: {lib_dir}")
    return [path for path in lib_dir.rglob("*") if path.is_file() and _is_shared_library(path)]


def _load_shared_libraries(lib_dir: Path) -> ctypes.CDLL:
    global _LOADED_LIBRARY_PATH

    candidates = _find_shared_libraries(lib_dir)
    if not candidates:
        raise RuntimeError(f"No shared libraries found under {lib_dir}")

    main_handle: ctypes.CDLL | None = None
    for path in sorted(candidates, key=_library_sort_key):
        handle = ctypes.CDLL(str(path))
        lowered = path.name.lower()
        if "openimagedenoise" in lowered and "device" not in lowered and "core" not in lowered:
            main_handle = handle
            _LOADED_LIBRARY_PATH = path

    if main_handle is None:
        raise RuntimeError("Failed to locate the OpenImageDenoise shared library")
    return main_handle


def load_library() -> ctypes.CDLL:
    return _load_shared_libraries(_lib_dir())


def library_dir() -> Path:
    return _lib_dir()


def loaded_library_version() -> str | None:
    if _LOADED_LIBRARY_PATH is None:
        return None
    match = _VERSION_RE.search(_LOADED_LIBRARY_PATH.name)
    if match is None:
        return None
    return match.group(1)


def packaged_library_version() -> str | None:
    lib_dir = _lib_dir()
    if not lib_dir.exists():
        return None
    candidates = _find_shared_libraries(lib_dir)
    if not candidates:
        return None
    main_candidates = [
        path
        for path in candidates
        if "openimagedenoise" in path.name.lower()
        and "device" not in path.name.lower()
        and "core" not in path.name.lower()
    ]
    for path in main_candidates + candidates:
        match = _VERSION_RE.search(path.name)
        if match is not None:
            return match.group(1)
    return None


def _get_func(
    lib: ctypes.CDLL,
    name: str,
    argtypes: Sequence[type[object]],
    restype: type[object] | None,
) -> _CFuncLike:
    func = getattr(lib, name)
    if not callable(func):
        raise TypeError(f"Expected callable ctypes function for {name}")
    if not hasattr(func, "argtypes") or not hasattr(func, "restype"):
        raise TypeError(f"Missing ctypes attributes for {name}")
    func.argtypes = list(argtypes)
    func.restype = restype
    return func


def bind_functions(lib: ctypes.CDLL) -> BoundFunctions:
    return BoundFunctions(
        oidnNewDevice=cast(
            NewDeviceFn, _get_func(lib, "oidnNewDevice", (ctypes.c_int,), ctypes.c_void_p)
        ),
        oidnCommitDevice=cast(
            CommitDeviceFn, _get_func(lib, "oidnCommitDevice", (ctypes.c_void_p,), None)
        ),
        oidnGetDeviceError=cast(
            GetDeviceErrorFn,
            _get_func(
                lib,
                "oidnGetDeviceError",
                (ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)),
                ctypes.c_int,
            ),
        ),
        oidnReleaseDevice=cast(
            ReleaseDeviceFn, _get_func(lib, "oidnReleaseDevice", (ctypes.c_void_p,), None)
        ),
        oidnRetainDevice=cast(
            RetainDeviceFn, _get_func(lib, "oidnRetainDevice", (ctypes.c_void_p,), None)
        ),
        oidnSetDeviceBool=cast(
            SetDeviceBoolFn,
            _get_func(
                lib, "oidnSetDeviceBool", (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool), None
            ),
        ),
        oidnSetDeviceInt=cast(
            SetDeviceIntFn,
            _get_func(
                lib, "oidnSetDeviceInt", (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int), None
            ),
        ),
        oidnGetDeviceBool=cast(
            GetDeviceBoolFn,
            _get_func(lib, "oidnGetDeviceBool", (ctypes.c_void_p, ctypes.c_char_p), ctypes.c_bool),
        ),
        oidnGetDeviceInt=cast(
            GetDeviceIntFn,
            _get_func(lib, "oidnGetDeviceInt", (ctypes.c_void_p, ctypes.c_char_p), ctypes.c_int),
        ),
        oidnNewFilter=cast(
            NewFilterFn,
            _get_func(lib, "oidnNewFilter", (ctypes.c_void_p, ctypes.c_char_p), ctypes.c_void_p),
        ),
        oidnSetSharedFilterImage=cast(
            SetSharedFilterImageFn,
            _get_func(
                lib,
                "oidnSetSharedFilterImage",
                (
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                ),
                None,
            ),
        ),
        oidnUnsetFilterImage=cast(
            UnsetFilterImageFn,
            _get_func(lib, "oidnUnsetFilterImage", (ctypes.c_void_p, ctypes.c_char_p), None),
        ),
        oidnSetSharedFilterData=cast(
            SetSharedFilterDataFn,
            _get_func(
                lib,
                "oidnSetSharedFilterData",
                (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t),
                None,
            ),
        ),
        oidnUpdateFilterData=cast(
            UpdateFilterDataFn,
            _get_func(lib, "oidnUpdateFilterData", (ctypes.c_void_p, ctypes.c_char_p), None),
        ),
        oidnUnsetFilterData=cast(
            UnsetFilterDataFn,
            _get_func(lib, "oidnUnsetFilterData", (ctypes.c_void_p, ctypes.c_char_p), None),
        ),
        oidnGetFilterBool=cast(
            GetFilterBoolFn,
            _get_func(lib, "oidnGetFilterBool", (ctypes.c_void_p, ctypes.c_char_p), ctypes.c_bool),
        ),
        oidnGetFilterFloat=cast(
            GetFilterFloatFn,
            _get_func(
                lib, "oidnGetFilterFloat", (ctypes.c_void_p, ctypes.c_char_p), ctypes.c_float
            ),
        ),
        oidnGetFilterInt=cast(
            GetFilterIntFn,
            _get_func(lib, "oidnGetFilterInt", (ctypes.c_void_p, ctypes.c_char_p), ctypes.c_int),
        ),
        oidnSetFilterBool=cast(
            SetFilterBoolFn,
            _get_func(
                lib, "oidnSetFilterBool", (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool), None
            ),
        ),
        oidnSetFilterFloat=cast(
            SetFilterFloatFn,
            _get_func(
                lib, "oidnSetFilterFloat", (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float), None
            ),
        ),
        oidnSetFilterInt=cast(
            SetFilterIntFn,
            _get_func(
                lib, "oidnSetFilterInt", (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int), None
            ),
        ),
        oidnCommitFilter=cast(
            CommitFilterFn, _get_func(lib, "oidnCommitFilter", (ctypes.c_void_p,), None)
        ),
        oidnExecuteFilter=cast(
            ExecuteFilterFn, _get_func(lib, "oidnExecuteFilter", (ctypes.c_void_p,), None)
        ),
        oidnReleaseFilter=cast(
            ReleaseFilterFn, _get_func(lib, "oidnReleaseFilter", (ctypes.c_void_p,), None)
        ),
        oidnRetainFilter=cast(
            RetainFilterFn, _get_func(lib, "oidnRetainFilter", (ctypes.c_void_p,), None)
        ),
    )


def init(lib: ctypes.CDLL | None = None, *, force: bool = False) -> BoundFunctions:
    global _FUNCTIONS, _LOADED_LIBRARY_PATH

    if _FUNCTIONS is not None and not force:
        return _FUNCTIONS

    if lib is None:
        lib = load_library()
    else:
        _LOADED_LIBRARY_PATH = None

    _FUNCTIONS = bind_functions(lib)
    return _FUNCTIONS


def get_functions() -> BoundFunctions:
    if _FUNCTIONS is None:
        init()
    if _FUNCTIONS is None:
        raise RuntimeError("OIDN bindings were not initialized")
    return _FUNCTIONS


def handle_from_ptr(value: int | None) -> int:
    if value is None:
        return 0
    return int(value)


def get_device_error(device_handle: DeviceHandle) -> tuple[int, str]:
    functions = get_functions()
    message_ptr = ctypes.c_char_p()
    error = functions.oidnGetDeviceError(device_handle, ctypes.pointer(message_ptr))
    message = message_ptr.value.decode() if message_ptr.value else ""
    prefix = _ERROR_MESSAGES.get(error, "An unknown error occurred: ")
    return error, f"{prefix}{message}"


def raise_for_error(device_handle: DeviceHandle) -> None:
    code, message = get_device_error(device_handle)
    if code != 0:
        raise RuntimeError(message)
