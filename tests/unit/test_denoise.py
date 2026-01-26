from __future__ import annotations

from typing import cast

import numpy as np
import oidn


class DummyDevice:
    def __init__(self, backend: oidn.Backend) -> None:
        self.backend = backend
        self.device_handle = 5
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeFilter:
    def __init__(self, device: oidn.Device, filter_type: str) -> None:
        self.device = device
        self.filter_type = filter_type
        self.images: dict[str, oidn.Buffer] = {}
        self.executed = False
        self.released = False

    def __enter__(self) -> FakeFilter:
        return self

    def __exit__(self, _1, _2, _3) -> None:
        self.release()

    def set_images(
        self,
        *,
        color: oidn.Buffer,
        output: oidn.Buffer,
        albedo: oidn.Buffer | None = None,
        normal: oidn.Buffer | None = None,
    ) -> None:
        self.images["color"] = color
        self.images["output"] = output
        if albedo is not None:
            self.images["albedo"] = albedo
        if normal is not None:
            self.images["normal"] = normal

    def execute(self) -> None:
        self.executed = True

    def release(self) -> None:
        self.released = True


def test_denoise_with_existing_device(monkeypatch) -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    fake_filter = FakeFilter(device, "RT")
    monkeypatch.setattr(oidn, "Filter", lambda device, filter_type: fake_filter)

    color = np.zeros((2, 2, 3), dtype=np.float32)
    output = oidn.denoise(color, device=device)

    assert output.width == 2
    assert fake_filter.images["color"].width == 2
    assert fake_filter.images["output"] is output
    assert fake_filter.executed is True
    assert device.released is False


def test_denoise_creates_device(monkeypatch) -> None:
    created: list[DummyDevice] = []

    def fake_device(*, backend=None, options=None):
        device = DummyDevice(oidn.Backend.parse(backend or "cpu"))
        created.append(device)
        return cast(oidn.Device, device)

    monkeypatch.setattr(oidn, "Device", fake_device)
    monkeypatch.setattr(oidn, "Filter", lambda device, filter_type: FakeFilter(device, filter_type))

    color = np.zeros((1, 1, 3), dtype=np.float32)
    oidn.denoise(color, backend="cpu")

    assert created[0].released is True


def test_denoise_with_aux_images(monkeypatch) -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    fake_filter = FakeFilter(device, "RT")
    monkeypatch.setattr(oidn, "Filter", lambda device, filter_type: fake_filter)

    color = np.zeros((1, 1, 3), dtype=np.float32)
    albedo = np.zeros((1, 1, 3), dtype=np.float32)
    normal = np.zeros((1, 1, 3), dtype=np.float32)
    output = oidn.Buffer.from_array(device, np.zeros((1, 1, 3), dtype=np.float32))

    result = oidn.denoise(color, albedo=albedo, normal=normal, output=output, device=device)

    assert result is output
    assert "albedo" in fake_filter.images
    assert "normal" in fake_filter.images
