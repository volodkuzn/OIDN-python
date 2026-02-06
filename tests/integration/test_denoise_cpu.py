from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import oidn
import pytest

if TYPE_CHECKING:
    CCharPPtr = ctypes._Pointer[ctypes.c_char_p]
else:
    CCharPPtr = ctypes.POINTER(ctypes.c_char_p)


@dataclass
class FakeFunctions:
    new_device_calls: list[int]
    new_filter_calls: list[bytes]
    image_calls: list[tuple[bytes, int, int]]
    commit_filter_calls: list[int]
    execute_filter_calls: list[int]

    def __init__(self) -> None:
        self.new_device_calls = []
        self.new_filter_calls = []
        self.image_calls = []
        self.commit_filter_calls = []
        self.execute_filter_calls = []

    def oidnNewDevice(self, device_type: int) -> int:
        self.new_device_calls.append(device_type)
        return 10

    def oidnCommitDevice(self, _handle: int) -> None:
        return None

    def oidnReleaseDevice(self, _handle: int) -> None:
        return None

    def oidnGetDeviceError(self, _handle: int, message_ptr: CCharPPtr) -> int:
        message_ptr.contents.value = b""
        return 0

    def oidnNewFilter(self, _handle: int, filter_type: bytes) -> int:
        self.new_filter_calls.append(filter_type)
        return 20

    def oidnCommitFilter(self, handle: int) -> None:
        self.commit_filter_calls.append(handle)

    def oidnExecuteFilter(self, handle: int) -> None:
        self.execute_filter_calls.append(handle)

    def oidnReleaseFilter(self, _handle: int) -> None:
        return None

    def oidnSetSharedFilterImage(
        self,
        _handle: int,
        name: bytes,
        _data_ptr: int,
        _format: int,
        width: int,
        height: int,
        _byte_offset: int,
        _byte_pixel_stride: int,
        _byte_row_stride: int,
    ) -> None:
        self.image_calls.append((name, width, height))


def test_cpu_denoise_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(oidn._ffi, "get_functions", lambda: fake)

    def fake_availability(_backend: object, *, check_runtime: bool = True) -> oidn._backends.BackendAvailability:
        return oidn._backends.BackendAvailability(oidn.Backend.CPU, True)

    monkeypatch.setattr(oidn._backends, "backend_availability", fake_availability)

    color = np.zeros((2, 2, 3), dtype=np.float32)
    albedo = np.zeros((2, 2, 3), dtype=np.float32)
    normal = np.zeros((2, 2, 3), dtype=np.float32)

    output = oidn.denoise(color, albedo=albedo, normal=normal, backend="cpu")

    assert output.width == 2
    assert fake.new_device_calls
    assert fake.new_filter_calls == [b"RT"]
    assert fake.commit_filter_calls == [20]
    assert fake.execute_filter_calls == [20]
    assert [name for name, _w, _h in fake.image_calls] == [
        b"color",
        b"albedo",
        b"normal",
        b"output",
    ]
