from __future__ import annotations

import builtins
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import numpy as np
import oidn
import pytest
from numpy.typing import DTypeLike, NDArray


class DummyDevice:
    def __init__(self, backend: oidn.Backend) -> None:
        self.backend = backend


class FakeTorchTensor:
    def __init__(self, interface: dict[str, object], dtype: object, is_cuda: bool) -> None:
        self.__cuda_array_interface__ = interface
        self.shape = interface["shape"]
        self.dtype = dtype
        self.is_cuda = is_cuda

    def detach(self) -> FakeTorchTensor:
        return self

    def cpu(self) -> FakeTorchTensor:
        return self

    def numpy(self) -> NDArray[np.float32]:
        shape = cast(tuple[int, ...], self.__cuda_array_interface__["shape"])
        return np.zeros(shape, dtype=np.float32)

    def float(self) -> FakeTorchTensor:
        return self

    def __truediv__(self, _value: builtins.float) -> FakeTorchTensor:
        return self


class FakeTorchModule:
    def __init__(self, *, cuda_available: bool, hip_available: bool) -> None:
        self.float16 = object()
        self.float32 = object()
        self.uint8 = object()
        self.int16 = object()
        self.short = self.int16
        self.Tensor = FakeTorchTensor
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.version = SimpleNamespace(hip="1.0" if hip_available else None)

    def zeros(
        self,
        shape: Sequence[int],
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> FakeTorchTensor:
        interface = make_interface(shape, np.float32)
        return FakeTorchTensor(interface, dtype, is_cuda=True)

    def tensor(
        self,
        data: object,
        *,
        device: object | None = None,
        dtype: object | None = None,
    ) -> FakeTorchTensor:
        interface = make_interface(getattr(data, "shape", (1,)), np.float32)
        return FakeTorchTensor(interface, dtype, is_cuda=True)


class FakeCupyModule:
    def zeros(self, shape: Sequence[int], dtype: DTypeLike | None = None) -> FakeCudaArray:
        return FakeCudaArray(make_interface(shape, dtype or np.float32))

    def asarray(self, array: NDArray[np.generic], dtype: DTypeLike | None = None) -> FakeCudaArray:
        return FakeCudaArray(make_interface(array.shape, dtype or array.dtype))


class FakeDpctlTensorModule:
    def zeros(self, shape: Sequence[int], dtype: DTypeLike | None = None) -> FakeSyclArray:
        return FakeSyclArray(make_interface(shape, dtype or np.float32))

    def asarray(self, array: NDArray[np.generic], dtype: DTypeLike | None = None) -> FakeSyclArray:
        return FakeSyclArray(make_interface(array.shape, dtype or array.dtype))


class FakeCudaArray:
    def __init__(self, interface: dict[str, object]) -> None:
        self.__cuda_array_interface__ = interface


class FakeHipArray:
    def __init__(self, interface: dict[str, object]) -> None:
        self.__hip_array_interface__ = interface


class FakeSyclArray:
    def __init__(self, interface: dict[str, object]) -> None:
        self.__sycl_usm_array_interface__ = interface


def make_interface(
    shape: Sequence[int],
    dtype: DTypeLike,
    *,
    ptr: int = 123,
    strides: Sequence[int] | None = None,
) -> dict[str, object]:
    np_dtype = np.dtype(dtype)
    if strides is None:
        strides = oidn._expected_strides(tuple(shape), np_dtype.itemsize)
    return {
        "shape": tuple(shape),
        "strides": tuple(strides),
        "typestr": np_dtype.str,
        "data": (ptr, False),
        "version": 3,
    }


def test_channel_order_parse() -> None:
    assert oidn.ChannelOrder.parse("hwc") is oidn.ChannelOrder.HWC
    assert oidn.ChannelOrder.parse("CHW") is oidn.ChannelOrder.CHW
    with pytest.raises(TypeError):
        oidn.ChannelOrder.parse(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        oidn.ChannelOrder.parse("bad")


def test_load_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(oidn.importlib, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="missing"):
        oidn._load_module("missing", reason="missing")


def test_load_torch_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)

    def fake_import(name: str) -> object:
        calls.append(name)
        return fake_torch

    monkeypatch.setattr(oidn.importlib, "import_module", fake_import)
    oidn._torch_module = None
    assert oidn._load_torch() is fake_torch
    assert oidn._load_torch() is fake_torch
    assert calls == ["torch"]


def test_load_cupy_and_dpctl_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cupy = FakeCupyModule()
    fake_dpctl = FakeDpctlTensorModule()

    def fake_import(name: str) -> object:
        if name == "cupy":
            return fake_cupy
        if name == "dpctl.tensor":
            return fake_dpctl
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(oidn.importlib, "import_module", fake_import)
    oidn._cupy_module = None
    oidn._dpctl_tensor_module = None

    assert oidn._load_cupy() is fake_cupy
    assert oidn._load_cupy() is fake_cupy
    assert oidn._load_dpctl_tensor() is fake_dpctl
    assert oidn._load_dpctl_tensor() is fake_dpctl


def test_require_helpers() -> None:
    module = SimpleNamespace(value=1, func=lambda: 3, cls=int)
    assert oidn._require_attr(module, "value") == 1
    assert oidn._require_callable(module, "func")() == 3
    assert oidn._require_type(module, "cls") is int
    with pytest.raises(RuntimeError):
        oidn._require_attr(module, "missing")
    with pytest.raises(RuntimeError):
        oidn._require_callable(module, "value")
    with pytest.raises(RuntimeError):
        oidn._require_type(module, "value")


def test_torch_helpers() -> None:
    torch = FakeTorchModule(cuda_available=True, hip_available=True)
    assert oidn._torch_cuda_available(torch) is True
    assert oidn._torch_hip_available(torch) is True
    assert oidn._torch_dtype(torch, np.dtype("float32")) is torch.float32
    assert oidn._torch_dtype(torch, np.dtype("float16")) is torch.float16
    with pytest.raises(ValueError):
        oidn._torch_dtype(torch, np.dtype("float64"))


def test_resolve_numpy_dtype() -> None:
    assert oidn._resolve_numpy_dtype(np.float32) == np.dtype("float32")
    with pytest.raises(ValueError):
        oidn._resolve_numpy_dtype(np.int32)
    with pytest.raises(TypeError):
        # exercise type validation
        oidn._resolve_numpy_dtype(object())  # type: ignore[arg-type]


def test_parse_interface_helpers() -> None:
    interface = make_interface((2, 3, 4), np.float32)
    spec = oidn._parse_array_interface(interface)
    assert spec.shape == (2, 3, 4)
    assert spec.pointer == 123
    assert spec.dtype == np.dtype("float32")
    mapping_ptr = oidn._parse_pointer({"ptr": 7, "read_only": True})
    assert mapping_ptr == (7, True)
    with pytest.raises(TypeError):
        oidn._tuple_of_ints("bad", name="shape")
    with pytest.raises(TypeError):
        oidn._tuple_of_ints([1, "bad"], name="shape")
    with pytest.raises(TypeError):
        oidn._parse_array_interface({"shape": (1, 2), "data": (1, False)})
    with pytest.raises(TypeError):
        oidn._parse_pointer({"data": 1})
    with pytest.raises(TypeError):
        oidn._parse_pointer(("bad", False))


def test_shape_and_format_helpers() -> None:
    height, width, channels = oidn._infer_image_shape((2, 3), oidn.ChannelOrder.HWC)
    assert (height, width, channels) == (2, 3, 1)
    with pytest.raises(ValueError):
        oidn._infer_image_shape((2, 3, 4), oidn.ChannelOrder.CHW)
    with pytest.raises(ValueError):
        oidn._infer_image_shape((2, 3, 4, 5), oidn.ChannelOrder.HWC)
    assert oidn._format_for_dtype(np.dtype("float32"), 3) == oidn.FORMAT_FLOAT3
    assert oidn._format_for_dtype(np.dtype("float16"), 3) == oidn.FORMAT_HALF3
    with pytest.raises(ValueError):
        oidn._format_for_dtype(np.dtype("float64"), 3)


def test_contiguity_helpers() -> None:
    shape = (2, 3, 4)
    strides = oidn._expected_strides(shape, np.dtype("float32").itemsize)
    assert oidn._is_c_contiguous(shape, strides, np.dtype("float32").itemsize) is True
    assert oidn._is_c_contiguous(shape, None, np.dtype("float32").itemsize) is True
    assert oidn._is_c_contiguous(shape, (-1, 0, 0), np.dtype("float32").itemsize) is False


def test_byte_strides_helper() -> None:
    assert oidn._byte_strides((2, 3), None, 4) == (12, 4)
    assert oidn._byte_strides((2, 3, 4), None, 4) == (48, 16)
    assert oidn._byte_strides((2, 3, 4), (48, 16, 4), 4) == (48, 16)
    assert oidn._byte_strides((2, 3), (12, 4), 4) == (12, 4)


def test_resolve_channel_order() -> None:
    assert oidn._resolve_channel_order(None, False) is oidn.ChannelOrder.HWC
    assert oidn._resolve_channel_order(None, True) is oidn.ChannelOrder.CHW
    with pytest.raises(ValueError):
        oidn._resolve_channel_order(oidn.ChannelOrder.HWC, True)


def test_array_interface_for_backend() -> None:
    device_cpu = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    array = np.zeros((1, 1, 1), dtype=np.float32)
    assert oidn._array_interface_for_backend(device_cpu, array)["shape"] == (1, 1, 1)
    device_hip = cast(oidn.Device, DummyDevice(oidn.Backend.HIP))
    hip_array = FakeCudaArray(make_interface((1, 1, 1), np.float32))
    interface = oidn._array_interface_for_backend(device_hip, hip_array)
    assert interface["shape"] == (1, 1, 1)
    device_sycl = cast(oidn.Device, DummyDevice(oidn.Backend.SYCL))
    with pytest.raises(TypeError):
        oidn._array_interface_for_backend(device_sycl, array)
    device_metal = cast(oidn.Device, DummyDevice(oidn.Backend.METAL))
    with pytest.raises(NotImplementedError):
        oidn._array_interface_for_backend(device_metal, array)


def test_buffer_from_array_cpu() -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    array = np.zeros((2, 3, 3), dtype=np.float32)
    buffer = oidn.Buffer.from_array(device, array)
    assert buffer.width == 3
    assert buffer.height == 2
    assert buffer.channels == 3
    assert buffer.format == oidn.FORMAT_FLOAT3
    assert buffer.data_ptr == array.__array_interface__["data"][0]
    assert buffer.channel_first is False


def test_buffer_from_array_errors() -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    bad_dtype = np.zeros((2, 2, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        oidn.Buffer.from_array(device, bad_dtype)
    non_contig = np.zeros((2, 2, 3), dtype=np.float32).transpose(1, 0, 2)
    with pytest.raises(ValueError):
        oidn.Buffer.from_array(device, non_contig)
    with pytest.raises(ValueError):
        oidn.Buffer.from_array(device, np.zeros((2, 2, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        oidn.Buffer.from_array(device, np.zeros((2, 2, 3), dtype=np.float32), channel_order="chw")
    with pytest.raises(ValueError):
        oidn.Buffer.from_array(device, np.zeros((2, 2, 3), dtype=np.float32), channel_first=True)

    with pytest.raises(TypeError):
        # exercise type validation
        oidn.Buffer.from_array(
            device,
            np.zeros((2, 2, 3), dtype=np.float32),
            channel_order=123,  # type: ignore[arg-type]
        )


def test_buffer_from_array_gpu_interfaces() -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CUDA))
    cuda_array = FakeCudaArray(make_interface((2, 2, 3), np.float32))
    buffer = oidn.Buffer.from_array(device, cuda_array)
    assert buffer.channels == 3

    device_hip = cast(oidn.Device, DummyDevice(oidn.Backend.HIP))
    hip_array = FakeHipArray(make_interface((2, 2, 3), np.float32))
    buffer = oidn.Buffer.from_array(device_hip, hip_array)
    assert buffer.channels == 3

    device_sycl = cast(oidn.Device, DummyDevice(oidn.Backend.SYCL))
    sycl_array = FakeSyclArray(make_interface((2, 2, 3), np.float32))
    buffer = oidn.Buffer.from_array(device_sycl, sycl_array)
    assert buffer.channels == 3


def test_buffer_create_cpu() -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    buffer = oidn.Buffer.create(2, 3, device=device, dtype=np.float16)
    assert isinstance(buffer.buffer_delegate, np.ndarray)
    assert buffer.dtype == np.dtype("float16")
    with pytest.raises(ValueError):
        oidn.Buffer.create(2, 3, device=None)
    with pytest.raises(ValueError):
        oidn.Buffer.create(2, 3, device=device, channels=5)
    with pytest.raises(ValueError):
        oidn.Buffer.create(2, 3, device=device, channel_order="chw")


def test_buffer_create_cuda_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CUDA))
    buffer = oidn.Buffer.create(1, 2, device=device, dtype=np.float32)
    assert buffer.channels == 3

    fake_cupy = FakeCupyModule()
    monkeypatch.setattr(oidn, "_load_cupy", lambda: fake_cupy)
    buffer = oidn.Buffer.create(1, 2, device=device, dtype=np.float32, use_cupy=True)
    assert buffer.channels == 3

    fake_torch_no_cuda = FakeTorchModule(cuda_available=False, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch_no_cuda)
    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available"):
        oidn.Buffer.create(1, 2, device=device, dtype=np.float32)


def test_buffer_create_hip_requires_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.HIP))
    with pytest.raises(RuntimeError, match="ROCm"):
        oidn.Buffer.create(1, 2, device=device, dtype=np.float32)


def test_buffer_create_sycl(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dpctl = FakeDpctlTensorModule()
    monkeypatch.setattr(oidn, "_load_dpctl_tensor", lambda: fake_dpctl)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.SYCL))
    buffer = oidn.Buffer.create(1, 2, device=device, dtype=np.float32)
    assert buffer.channels == 3


def test_buffer_create_metal_unsupported() -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.METAL))
    with pytest.raises(NotImplementedError):
        oidn.Buffer.create(1, 2, device=device, dtype=np.float32)


def test_buffer_load_numpy_cpu() -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    array = np.zeros((2, 2, 3), dtype=np.float32)
    buffer = oidn.Buffer.load(device, array, normalize=False)
    assert buffer.width == 2


def test_buffer_load_rejects_normalize_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    array = np.zeros((1, 1, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="normalize=True requires copy_data=True"):
        oidn.Buffer.load(device, array, normalize=True, copy_data=False)


def test_buffer_load_numpy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=True)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CUDA))
    array = np.zeros((2, 2, 3), dtype=np.float32)
    buffer = oidn.Buffer.load(device, array, normalize=False)
    assert buffer.channels == 3

    fake_cupy = FakeCupyModule()
    monkeypatch.setattr(oidn, "_load_cupy", lambda: fake_cupy)
    buffer = oidn.Buffer.load(device, array, normalize=False, use_cupy=True)
    assert buffer.channels == 3


def test_buffer_load_sycl_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_dpctl = FakeDpctlTensorModule()
    monkeypatch.setattr(oidn, "_load_dpctl_tensor", lambda: fake_dpctl)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.SYCL))
    array = np.zeros((2, 2, 3), dtype=np.float32)
    buffer = oidn.Buffer.load(device, array, normalize=False)
    assert buffer.channels == 3


def test_buffer_load_torch_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    interface = make_interface((1, 1, 3), np.float32)
    tensor = FakeTorchTensor(interface, fake_torch.uint8, is_cuda=False)
    buffer = oidn.Buffer.load(device, tensor, normalize=False)
    assert buffer.channels == 3


def test_buffer_load_torch_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CUDA))
    interface = make_interface((1, 1, 3), np.float32)
    tensor = FakeTorchTensor(interface, fake_torch.uint8, is_cuda=True)
    buffer = oidn.Buffer.load(device, tensor, normalize=True)
    assert buffer.channels == 3


def test_buffer_to_array_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CUDA))
    interface = make_interface((1, 1, 3), np.float32)
    tensor = FakeTorchTensor(interface, fake_torch.float32, is_cuda=True)
    buffer = oidn.Buffer.from_array(device, tensor)
    assert isinstance(buffer.to_array(), np.ndarray)


def test_buffer_to_tensor_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = FakeTorchModule(cuda_available=True, hip_available=False)
    monkeypatch.setattr(oidn, "_load_torch", lambda: fake_torch)
    device = cast(oidn.Device, DummyDevice(oidn.Backend.CPU))
    array = np.zeros((1, 1, 3), dtype=np.float32)
    buffer = oidn.Buffer.from_array(device, array)
    with pytest.raises(RuntimeError):
        buffer.to_tensor()
