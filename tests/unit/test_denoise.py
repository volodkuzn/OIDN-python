from __future__ import annotations

from types import TracebackType
from typing import cast

import numpy as np
import oidn
import pytest


class DummyDevice:
    def __init__(self, backend: oidn.Backend) -> None:
        self.backend = backend
        self.device_handle = 5
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeFilter:
    def __init__(
        self,
        device: oidn.Device,
        filter_type: str,
        *,
        hdr: bool = False,
        inputScale: float | None = None,
        cleanAux: bool = False,
        directional: bool = False,
        quality: oidn.FilterQuality | str | None = None,
    ) -> None:
        self.device = device
        self.filter_type = filter_type
        self.hdr = hdr
        self.input_scale = inputScale
        self.clean_aux = cleanAux
        self.directional = directional
        self.quality = quality
        self.images: dict[str, oidn.Buffer] = {}
        self.executed = False
        self.released = False

    def __enter__(self) -> FakeFilter:
        return self

    def __exit__(
        self,
        _1: type[BaseException] | None,
        _2: BaseException | None,
        _3: TracebackType | None,
    ) -> None:
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


def test_denoise_with_existing_device(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_device = DummyDevice(oidn.Backend.CPU)
    device = cast(oidn.Device, dummy_device)
    fake_filter = FakeFilter(device, "RT")
    monkeypatch.setattr(oidn, "Filter", lambda device, filter_type, **_kwargs: fake_filter)

    color = np.zeros((2, 2, 3), dtype=np.float32)
    output = oidn.denoise(color, device=device)

    assert output.width == 2
    assert fake_filter.images["color"].width == 2
    assert fake_filter.images["output"] is output
    assert fake_filter.executed is True
    assert dummy_device.released is False


def test_denoise_creates_device(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[DummyDevice] = []

    def fake_device(
        *,
        backend: oidn.Backend | str | None = None,
        options: oidn.DeviceOptions | None = None,
    ) -> oidn.Device:
        device = DummyDevice(oidn.Backend.parse(backend or "cpu"))
        created.append(device)
        return cast(oidn.Device, device)

    monkeypatch.setattr(oidn, "Device", fake_device)
    monkeypatch.setattr(oidn, "Filter", lambda device, filter_type, **_kwargs: FakeFilter(device, filter_type))

    color = np.zeros((1, 1, 3), dtype=np.float32)
    oidn.denoise(color, backend="cpu")

    assert created[0].released is True


def test_denoise_with_aux_images(monkeypatch: pytest.MonkeyPatch) -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    fake_filter = FakeFilter(device, "RT")
    monkeypatch.setattr(oidn, "Filter", lambda device, filter_type, **_kwargs: fake_filter)

    color = np.zeros((1, 1, 3), dtype=np.float32)
    albedo = np.zeros((1, 1, 3), dtype=np.float32)
    normal = np.zeros((1, 1, 3), dtype=np.float32)
    output = oidn.Buffer.from_array(device, np.zeros((1, 1, 3), dtype=np.float32))

    result = oidn.denoise(color, albedo=albedo, normal=normal, output=output, device=device)

    assert result is output
    assert "albedo" in fake_filter.images
    assert "normal" in fake_filter.images


def test_denoise_forwards_filter_options(monkeypatch: pytest.MonkeyPatch) -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    created: list[FakeFilter] = []

    def fake_filter_ctor(
        device: oidn.Device,
        filter_type: str,
        *,
        hdr: bool = False,
        inputScale: float | None = None,
        cleanAux: bool = False,
        directional: bool = False,
        quality: oidn.FilterQuality | str | None = None,
    ) -> FakeFilter:
        filter_obj = FakeFilter(
            device,
            filter_type,
            hdr=hdr,
            inputScale=inputScale,
            cleanAux=cleanAux,
            directional=directional,
            quality=quality,
        )
        created.append(filter_obj)
        return filter_obj

    monkeypatch.setattr(oidn, "Filter", fake_filter_ctor)

    color = np.zeros((1, 1, 3), dtype=np.float32)
    oidn.denoise(
        color,
        device=device,
        hdr=True,
        inputScale=4.0,
        cleanAux=False,
        quality=oidn.FilterQuality.BALANCED,
    )

    forwarded = created[0]
    assert forwarded.hdr is True
    assert forwarded.input_scale == 4.0
    assert forwarded.clean_aux is False
    assert forwarded.quality is oidn.FilterQuality.BALANCED


def test_denoise_forwards_directional_for_rtlightmap(monkeypatch: pytest.MonkeyPatch) -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    created: list[FakeFilter] = []

    def fake_filter_ctor(
        device: oidn.Device,
        filter_type: str,
        *,
        hdr: bool = False,
        inputScale: float | None = None,
        cleanAux: bool = False,
        directional: bool = False,
        quality: oidn.FilterQuality | str | None = None,
    ) -> FakeFilter:
        filter_obj = FakeFilter(
            device,
            filter_type,
            hdr=hdr,
            inputScale=inputScale,
            cleanAux=cleanAux,
            directional=directional,
            quality=quality,
        )
        created.append(filter_obj)
        return filter_obj

    monkeypatch.setattr(oidn, "Filter", fake_filter_ctor)

    color = np.zeros((1, 1, 3), dtype=np.float32)
    oidn.denoise(
        color,
        device=device,
        filter_type="RTLightmap",
        directional=True,
    )

    forwarded = created[0]
    assert forwarded.filter_type == "RTLightmap"
    assert forwarded.directional is True
