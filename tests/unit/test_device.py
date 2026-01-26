from __future__ import annotations

import oidn
import oidn._backends as backends
import pytest


def _patch_device(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: bool = True,
    reason: str | None = None,
    new_device_return: int = 11,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_backend_availability(
        backend: backends.Backend | str, *, check_runtime: bool = True
    ) -> backends.BackendAvailability:
        resolved = backends.Backend.parse(backend)
        calls["backend"] = resolved
        calls["check_runtime"] = check_runtime
        return backends.BackendAvailability(resolved, available, reason)

    def fake_new_device(device_type: int) -> int:
        calls["device_type"] = device_type
        return new_device_return

    def fake_commit(handle: int) -> None:
        calls["commit"] = handle

    def fake_apply(handle: int, options: backends.DeviceOptions) -> None:
        calls["apply"] = (handle, options)

    monkeypatch.setattr(backends, "backend_availability", fake_backend_availability)
    monkeypatch.setattr(backends, "apply_device_options", fake_apply)
    monkeypatch.setattr(oidn, "NewDevice", fake_new_device)
    monkeypatch.setattr(oidn, "CommitDevice", fake_commit)
    monkeypatch.setattr(oidn, "ReleaseDevice", lambda handle: None)
    return calls


def test_device_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_device(monkeypatch)

    device = oidn.Device()

    assert calls["backend"] is backends.Backend.CPU
    assert calls["device_type"] == backends.device_type_for_backend(backends.Backend.CPU)
    assert calls["commit"] == device.device_handle
    assert device.backend is backends.Backend.CPU
    assert device.type == backends.device_type_for_backend(backends.Backend.CPU)


def test_device_accepts_device_type_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_device(monkeypatch)

    device = oidn.Device(device_type="cuda")

    assert calls["backend"] is backends.Backend.CUDA
    assert calls["device_type"] == backends.device_type_for_backend(backends.Backend.CUDA)
    assert device.backend is backends.Backend.CUDA


def test_device_rejects_backend_and_device_type(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_device(monkeypatch)

    with pytest.raises(ValueError, match="Specify backend or device_type"):
        oidn.Device("cpu", device_type="cuda")


def test_device_raises_when_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_device(monkeypatch, available=False, reason="Backend missing")

    with pytest.raises(RuntimeError, match="Backend missing"):
        oidn.Device("cuda")


def test_device_raises_when_new_device_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_device(monkeypatch, new_device_return=0)

    with pytest.raises(RuntimeError, match="Failed to create device"):
        oidn.Device("cpu")


def test_device_applies_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_device(monkeypatch)
    options = backends.DeviceOptions(set_affinity=True)

    device = oidn.Device("cpu", options=options)

    assert calls["apply"] == (device.device_handle, options)
