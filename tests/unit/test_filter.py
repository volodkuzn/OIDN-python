from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import oidn


class DummyDevice:
    def __init__(self, backend: oidn.Backend) -> None:
        self.backend = backend
        self.device_handle = 7

    def raise_if_error(self) -> None:
        return None

    def release(self) -> None:
        return None


class FakeFunctions:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def oidnSetSharedFilterImage(self, *args: object) -> None:
        self.calls.append(args)


class FakeCudaArray:
    def __init__(self) -> None:
        self.__cuda_array_interface__ = {
            "shape": (1, 1, 3),
            "strides": (12, 12, 4),
            "typestr": np.dtype(np.float32).str,
            "data": (321, False),
            "version": 3,
        }


def _patch_filter(monkeypatch: pytest.MonkeyPatch, fake_functions: FakeFunctions) -> None:
    monkeypatch.setattr(oidn._ffi, "get_functions", lambda: fake_functions)
    monkeypatch.setattr(oidn, "NewFilter", lambda device_handle, type: 11)
    monkeypatch.setattr(oidn, "CommitFilter", lambda handle: None)
    monkeypatch.setattr(oidn, "ReleaseFilter", lambda handle: None)
    monkeypatch.setattr(oidn, "ExecuteFilter", lambda handle: None)


def test_filter_set_images_calls_ffi(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.from_array(device, np.zeros((2, 2, 3), dtype=np.float32))

    filter_obj = oidn.Filter(device, "RT")
    filter_obj.set_images(color=buffer, output=buffer)

    assert len(fake_functions.calls) == 2
    assert fake_functions.calls[0][1] == b"color"
    assert fake_functions.calls[1][1] == b"output"


def test_filter_rejects_invalid_name(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.from_array(device, np.zeros((1, 1, 3), dtype=np.float32))
    filter_obj = oidn.Filter(device, "RT")

    with pytest.raises(ValueError, match="Unsupported image name"):
        filter_obj.set_image("depth", buffer)


def test_filter_rejects_aux_for_non_rt(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.from_array(device, np.zeros((1, 1, 3), dtype=np.float32))
    filter_obj = oidn.Filter(device, "RTLightmap")

    with pytest.raises(RuntimeError, match="only supported for RT"):
        filter_obj.set_image("albedo", buffer)


def test_filter_rejects_backend_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    gpu_device = cast(oidn.Device, DummyDevice(oidn.Backend.CUDA))
    buffer = oidn.Buffer.from_array(gpu_device, FakeCudaArray())
    filter_obj = oidn.Filter(device, "RT")

    with pytest.raises(RuntimeError, match="backend does not match"):
        filter_obj.set_image("color", buffer)


def test_filter_rejects_channel_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.from_array(device, np.zeros((1, 1, 3), dtype=np.float32))
    buffer.channel_order = oidn.ChannelOrder.CHW
    filter_obj = oidn.Filter(device, "RT")

    with pytest.raises(RuntimeError, match="HWC"):
        filter_obj.set_image("color", buffer)


def test_filter_rejects_channel_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.from_array(device, np.zeros((1, 1), dtype=np.float32))
    filter_obj = oidn.Filter(device, "RT")

    with pytest.raises(RuntimeError, match="3 channels"):
        filter_obj.set_image("color", buffer)


def test_filter_rejects_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.from_array(device, np.zeros((2, 2, 3), dtype=np.float32))
    other = oidn.Buffer.from_array(device, np.zeros((3, 2, 3), dtype=np.float32))
    filter_obj = oidn.Filter(device, "RT")
    filter_obj.set_image("color", buffer)

    with pytest.raises(RuntimeError, match="same dimensions"):
        filter_obj.set_image("output", other)


def test_filter_execute_invalid_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_functions = FakeFunctions()
    _patch_filter(monkeypatch, fake_functions)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    filter_obj = oidn.Filter(device, "RT")
    filter_obj.release()

    with pytest.raises(RuntimeError, match="Invalid filter handle"):
        filter_obj.execute()
