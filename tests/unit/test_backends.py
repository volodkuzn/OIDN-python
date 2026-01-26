from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import cast

import oidn._backends as backends
import oidn._ffi as ffi
import pytest
from oidn.constants import (
    DEVICE_TYPE_CPU,
    DEVICE_TYPE_CUDA,
    DEVICE_TYPE_DEFAULT,
    DEVICE_TYPE_HIP,
    DEVICE_TYPE_METAL,
    DEVICE_TYPE_SYCL,
)


class FakeFunctions:
    def __init__(self) -> None:
        self.bool_calls: list[tuple[int, bytes, bool]] = []
        self.int_calls: list[tuple[int, bytes, int]] = []

    def oidnSetDeviceBool(self, handle: int, name: bytes, value: bool) -> None:
        self.bool_calls.append((handle, name, value))

    def oidnSetDeviceInt(self, handle: int, name: bytes, value: int) -> None:
        self.int_calls.append((handle, name, value))

    def oidnNewDevice(self, device_type: int) -> int:
        return 42

    def oidnCommitDevice(self, handle: int) -> None:
        return None

    def oidnReleaseDevice(self, handle: int) -> None:
        return None


def test_backend_parse_accepts_values() -> None:
    assert backends.Backend.parse(backends.Backend.CPU) is backends.Backend.CPU
    assert backends.Backend.parse("cpu") is backends.Backend.CPU
    assert backends.Backend.parse("AUTO") is backends.Backend.AUTO
    assert backends.Backend.parse("default") is backends.Backend.AUTO


def test_backend_parse_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        backends.Backend.parse("vulkan")
    with pytest.raises(TypeError, match="backend must be a Backend or str"):
        backends.Backend.parse(cast(backends.Backend, 3))


def test_device_type_for_backend() -> None:
    assert backends.device_type_for_backend(backends.Backend.AUTO) == DEVICE_TYPE_DEFAULT
    assert backends.device_type_for_backend(backends.Backend.CPU) == DEVICE_TYPE_CPU
    assert backends.device_type_for_backend(backends.Backend.SYCL) == DEVICE_TYPE_SYCL
    assert backends.device_type_for_backend(backends.Backend.CUDA) == DEVICE_TYPE_CUDA
    assert backends.device_type_for_backend(backends.Backend.HIP) == DEVICE_TYPE_HIP
    assert backends.device_type_for_backend(backends.Backend.METAL) == DEVICE_TYPE_METAL


def test_backend_library_detection(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib.macos.aarch64"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libOpenImageDenoise_device_cpu.2.4.1.dylib").touch()

    monkeypatch.setattr(ffi, "library_dir", lambda: lib_dir)
    assert backends.is_backend_available(backends.Backend.CPU, check_runtime=False) is True
    assert backends.is_backend_available(backends.Backend.CUDA, check_runtime=False) is False


def test_available_backends_respects_order(tmp_path: Path, monkeypatch) -> None:
    lib_dir = tmp_path / "lib.linux.x64"
    lib_dir.mkdir(parents=True)
    (lib_dir / "libOpenImageDenoise_device_cpu.so.2.4.1").touch()
    (lib_dir / "libOpenImageDenoise_device_cuda.so.2.4.1").touch()

    monkeypatch.setattr(ffi, "library_dir", lambda: lib_dir)
    backends_list = backends.available_backends(check_runtime=False)

    assert backends_list == [backends.Backend.CPU, backends.Backend.CUDA]


def test_backend_auto_availability(monkeypatch) -> None:
    def fake_available(backend: backends.Backend | str, *, check_runtime: bool = True) -> bool:
        return backends.Backend.parse(backend) is backends.Backend.CPU

    monkeypatch.setattr(backends, "is_backend_available", fake_available)

    availability = backends.backend_availability(backends.Backend.AUTO)

    assert availability.available is True


def test_apply_device_options_sets_values(monkeypatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(ffi, "get_functions", lambda: fake)

    options = backends.DeviceOptions(
        set_affinity=True,
        verbose=2,
        num_threads=4,
        extra_params={"customFlag": True, "threads": 8},
    )

    backends.apply_device_options(5, options)

    assert fake.bool_calls == [(5, b"setAffinity", True), (5, b"customFlag", True)]
    assert fake.int_calls == [(5, b"verbose", 2), (5, b"numThreads", 4), (5, b"threads", 8)]


def test_apply_device_options_rejects_reserved(monkeypatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(ffi, "get_functions", lambda: fake)

    options = backends.DeviceOptions(extra_params={"verbose": 2})
    with pytest.raises(ValueError, match="extra_params may not override"):
        backends.apply_device_options(1, options)


def test_apply_device_options_rejects_non_ascii(monkeypatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(ffi, "get_functions", lambda: fake)

    options = backends.DeviceOptions(extra_params={"t\u00e9st": True})
    with pytest.raises(ValueError, match="Device option name must be ASCII"):
        backends.apply_device_options(1, options)


def test_apply_device_options_rejects_type(monkeypatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(ffi, "get_functions", lambda: fake)

    options = backends.DeviceOptions(extra_params={"invalid": 1.5})
    with pytest.raises(TypeError, match="Unsupported option type"):
        backends.apply_device_options(1, options)


def test_check_python_runtime_cpu() -> None:
    ok, reason = backends._check_python_runtime(backends.Backend.CPU)
    assert ok is True
    assert reason is None


def test_check_python_runtime_sycl(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    ok, reason = backends._check_python_runtime(backends.Backend.SYCL)
    assert ok is False
    assert reason == "SYCL backend requires dpctl for array support."


def test_check_python_runtime_metal(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    ok, reason = backends._check_python_runtime(backends.Backend.METAL)
    assert ok is False
    assert reason == "Metal backend requires macOS."


def test_check_python_runtime_cuda(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_torch_runtime_status", lambda: (False, "missing", False))
    ok, reason = backends._check_python_runtime(backends.Backend.CUDA)
    assert ok is False
    assert reason == "missing"


def test_check_python_runtime_hip(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_torch_runtime_status", lambda: (True, None, False))
    ok, reason = backends._check_python_runtime(backends.Backend.HIP)
    assert ok is False
    assert reason == "HIP backend requires a ROCm-enabled torch build."
    monkeypatch.setattr(backends, "_torch_runtime_status", lambda: (True, None, True))
    ok, reason = backends._check_python_runtime(backends.Backend.HIP)
    assert ok is True
    assert reason is None


def test_torch_runtime_status(monkeypatch) -> None:
    def fail_import(_name: str):
        raise ModuleNotFoundError

    monkeypatch.setattr(importlib, "import_module", fail_import)
    ok, message, hip = backends._torch_runtime_status()
    assert ok is False
    assert message == "CUDA/HIP backends require torch to be installed."
    assert hip is False

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()
        version = type("Version", (), {"hip": None})

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorch())
    ok, message, hip = backends._torch_runtime_status()
    assert ok is False
    assert message == "torch.cuda.is_available() is False."
    assert hip is False

    class FakeCudaOk:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorchHip:
        cuda = FakeCudaOk()
        version = type("Version", (), {"hip": "1.0"})

    monkeypatch.setattr(importlib, "import_module", lambda _name: FakeTorchHip())
    ok, message, hip = backends._torch_runtime_status()
    assert ok is True
    assert message is None
    assert hip is True


def test_probe_device_success(monkeypatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(ffi, "get_functions", lambda: fake)
    monkeypatch.setattr(ffi, "get_device_error", lambda handle: (0, ""))

    ok, reason = backends._probe_device(backends.Backend.CPU)

    assert ok is True
    assert reason is None


def test_probe_device_failure(monkeypatch) -> None:
    fake = FakeFunctions()
    monkeypatch.setattr(ffi, "get_functions", lambda: fake)
    monkeypatch.setattr(ffi, "get_device_error", lambda handle: (5, "unsupported"))

    ok, reason = backends._probe_device(backends.Backend.CPU)

    assert ok is False
    assert reason == "unsupported"
