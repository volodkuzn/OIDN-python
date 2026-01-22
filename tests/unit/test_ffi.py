from __future__ import annotations

from pathlib import Path
from typing import cast

import ctypes
import pytest

import oidn._ffi as ffi


class FakeFunc:
    def __init__(self, name: str, return_value: int = 0) -> None:
        self.name = name
        self.return_value = return_value
        self.argtypes: list[object] = []
        self.restype: object | None = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        return self.return_value


class FakeGetDeviceError(FakeFunc):
    def __call__(self, device: object, message_ptr: object) -> object:
        self.calls.append((device, message_ptr))
        ptr = ctypes.cast(message_ptr, ctypes.POINTER(ctypes.c_char_p))
        ptr.contents.value = b"bad argument"
        return 2


class FakeLibrary:
    def __init__(self) -> None:
        self._funcs: dict[str, FakeFunc] = {
            "oidnGetDeviceError": FakeGetDeviceError("oidnGetDeviceError"),
        }

    def __getattr__(self, name: str) -> FakeFunc:
        if name not in self._funcs:
            self._funcs[name] = FakeFunc(name)
        return self._funcs[name]


class FakeCDLL:
    def __init__(self, name: str) -> None:
        self.name = name


def test_load_library_uses_package_data(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib.macos.aarch64"
    lib_dir.mkdir(parents=True)
    names = [
        "libtbb.12.dylib",
        "libOpenImageDenoise_core.2.4.1.dylib",
        "libOpenImageDenoise_device_cpu.2.4.1.dylib",
        "libOpenImageDenoise.2.4.1.dylib",
    ]
    for name in names:
        (lib_dir / name).touch()

    loaded: list[str] = []

    def fake_cdll(path: str) -> ctypes.CDLL:
        loaded.append(Path(path).name)
        fake = FakeCDLL(Path(path).name)
        return cast(ctypes.CDLL, fake)

    monkeypatch.setattr(ffi, "_package_root", lambda: tmp_path)
    monkeypatch.setattr(ffi, "_platform_tag", lambda: "macos")
    monkeypatch.setattr(ffi, "_arch_tag", lambda: "aarch64")
    monkeypatch.setattr(ffi.ctypes, "CDLL", fake_cdll)

    handle = ffi.load_library()

    assert loaded == names
    assert isinstance(handle, FakeCDLL)
    assert handle.name == "libOpenImageDenoise.2.4.1.dylib"


def test_get_device_error_formats_message(monkeypatch) -> None:
    fake_lib = cast(ctypes.CDLL, FakeLibrary())
    monkeypatch.setattr(ffi, "_FUNCTIONS", None)
    ffi.init(fake_lib, force=True)

    code, message = ffi.get_device_error(123)

    assert code == 2
    assert message == "An invalid argument was specified: bad argument"


def test_raise_for_error_raises(monkeypatch) -> None:
    fake_lib = cast(ctypes.CDLL, FakeLibrary())
    monkeypatch.setattr(ffi, "_FUNCTIONS", None)
    ffi.init(fake_lib, force=True)

    with pytest.raises(RuntimeError, match="invalid argument"):
        ffi.raise_for_error(123)


def test_handle_from_ptr() -> None:
    assert ffi.handle_from_ptr(None) == 0
    assert ffi.handle_from_ptr(42) == 42


def test_loaded_library_version(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ffi, "_LOADED_LIBRARY_PATH", tmp_path / "libOpenImageDenoise.2.4.1.dylib")
    assert ffi.loaded_library_version() == "2.4.1"
