from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
from PIL import Image

from oidn import _backends, _ffi
from oidn._backends import (  # noqa: F401 - re-exported public API
    Backend,
    DeviceOptions,
    available_backends,
    is_backend_available,
)
from oidn.capi import *

oidn_version = _ffi.loaded_library_version() or _ffi.packaged_library_version() or "unknown"
oidn_py_version = "0.4"


class AutoReleaseByContextManaeger:
    def __enter__(self):
        return self

    def __exit__(self, _1, _2, _3):
        self.release()


class Device(AutoReleaseByContextManaeger):
    def __init__(
        self,
        backend: Backend | str | None = None,
        *,
        device_type: Backend | str | None = None,
        options: DeviceOptions | None = None,
    ) -> None:
        r"""
        Create an OIDN device.

        Args:
            backend: backend selection (cpu/cuda/hip/sycl/metal/auto)
            device_type: legacy alias for backend selection
            options: device configuration options
        """
        if backend is None and device_type is None:
            resolved = Backend.CPU
        elif backend is not None and device_type is not None:
            raise ValueError("Specify backend or device_type, not both.")
        elif backend is not None:
            resolved = Backend.parse(backend)
        else:
            resolved = Backend.parse(device_type)
        availability = _backends.backend_availability(resolved)
        if not availability.available:
            reason = availability.reason or "Backend is unavailable."
            raise RuntimeError(reason)

        self.backend = resolved
        device_type_value = _backends.device_type_for_backend(resolved)
        self.type = device_type_value
        self.__device_handle = NewDevice(device_type_value)
        if self.__device_handle == 0:
            raise RuntimeError("Failed to create device.")
        if options is not None:
            _backends.apply_device_options(self.device_handle, options)
        CommitDevice(self.device_handle)

    @property
    def error(self):
        """
        Returns a tuple[error_code, error_message], the same as oidn.GetDeviceError.
        """
        return GetDeviceError(self.device_handle)

    def raise_if_error(self):
        """
        Raise a RuntimeError if an error occured.
        """
        err = self.error
        if err is not None:
            if err[0] != 0:
                raise RuntimeError(err[1])

    def release(self):
        """
        Call ReleaseDevice with self.device_handle
        """
        if self.device_handle:  # not 0, not None
            ReleaseDevice(self.device_handle)
        self.native_handle = 0

    @property
    def is_cpu(self):
        """
        Indicate whether it is a CPU device.
        """
        return self.backend is Backend.CPU

    @property
    def is_cuda(self):
        """
        Indicate wheter it is a CUDA device.
        """
        return self.backend is Backend.CUDA

    @property
    def is_hip(self):
        """
        Indicate whether it is a HIP device.
        """
        return self.backend is Backend.HIP

    @property
    def is_sycl(self):
        """
        Indicate whether it is a SYCL device.
        """
        return self.backend is Backend.SYCL

    @property
    def is_metal(self):
        """
        Indicate whether it is a Metal device.
        """
        return self.backend is Backend.METAL

    @property
    def device_handle(self):
        """
        Returns the device handle
        """
        return self.__device_handle


_torch_module: object | None = None
_cupy_module: object | None = None
_dpctl_tensor_module: object | None = None


def _load_module(name: str, *, reason: str) -> object:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(reason) from exc


def _load_torch() -> object:
    global _torch_module
    if _torch_module is None:
        _torch_module = _load_module(
            "torch",
            reason="CUDA/HIP backends require torch to be installed.",
        )
    return _torch_module


def _load_cupy() -> object:
    global _cupy_module
    if _cupy_module is None:
        _cupy_module = _load_module(
            "cupy",
            reason="CuPy is required to allocate CUDA/HIP buffers when use_cupy=True.",
        )
    return _cupy_module


def _load_dpctl_tensor() -> object:
    global _dpctl_tensor_module
    if _dpctl_tensor_module is None:
        _dpctl_tensor_module = _load_module(
            "dpctl.tensor",
            reason="SYCL backend requires dpctl for buffer allocation.",
        )
    return _dpctl_tensor_module


def _require_callable(module: object, name: str) -> Callable[..., object]:
    value = getattr(module, name, None)
    if not callable(value):
        raise RuntimeError(f"Module {module!r} is missing callable {name}.")
    return cast(Callable[..., object], value)


def _require_attr(module: object, name: str) -> object:
    value = getattr(module, name, None)
    if value is None:
        raise RuntimeError(f"Module {module!r} is missing attribute {name}.")
    return value


def _require_type(module: object, name: str) -> type:
    value = _require_attr(module, name)
    if not isinstance(value, type):
        raise RuntimeError(f"Module {module!r} attribute {name} is not a type.")
    return value


def _torch_cuda_available(torch_module: object) -> bool:
    cuda = _require_attr(torch_module, "cuda")
    is_available = _require_callable(cuda, "is_available")
    return bool(is_available())


def _torch_hip_available(torch_module: object) -> bool:
    version = getattr(torch_module, "version", None)
    if version is None:
        return False
    return getattr(version, "hip", None) is not None


def _torch_dtype(torch_module: object, dtype: np.dtype) -> object:
    if dtype == np.dtype("float32"):
        return _require_attr(torch_module, "float32")
    if dtype == np.dtype("float16"):
        return _require_attr(torch_module, "float16")
    raise ValueError(f"Unsupported dtype: {dtype}")


def _resolve_numpy_dtype(dtype: object) -> np.dtype:
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:
        raise TypeError(f"Unsupported dtype: {dtype}") from exc
    if resolved not in (np.dtype("float32"), np.dtype("float16")):
        raise ValueError(f"Unsupported dtype: {resolved}")
    return resolved


class ChannelOrder(str, Enum):
    HWC = "hwc"
    CHW = "chw"

    @classmethod
    def parse(cls, value: ChannelOrder | str) -> ChannelOrder:
        if isinstance(value, ChannelOrder):
            return value
        if not isinstance(value, str):
            raise TypeError("channel_order must be a ChannelOrder or str")
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unsupported channel order: {value}")


@dataclass(frozen=True, slots=True)
class _ArraySpec:
    pointer: int
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    dtype: np.dtype
    read_only: bool


def _tuple_of_ints(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    items: list[int] = []
    for item in value:
        if not isinstance(item, (int, np.integer)):
            raise TypeError(f"{name} must contain integers.")
        items.append(int(item))
    return tuple(items)


def _parse_pointer(data: object) -> tuple[int, bool]:
    if isinstance(data, (tuple, list)) and len(data) == 2:
        ptr, read_only = data
    elif isinstance(data, Mapping):
        if "ptr" in data:
            ptr = data["ptr"]
            read_only = data.get("read_only", False)
        else:
            raise TypeError("Array interface data mapping must include 'ptr'.")
    else:
        raise TypeError("Array interface data must be a tuple or mapping.")
    if not isinstance(ptr, (int, np.integer)):
        raise TypeError("Array interface pointer must be an integer.")
    return int(ptr), bool(read_only)


def _parse_array_interface(interface: Mapping[str, object]) -> _ArraySpec:
    if "typestr" not in interface:
        raise TypeError("Array interface missing 'typestr'.")
    dtype = np.dtype(interface["typestr"])
    shape = _tuple_of_ints(interface.get("shape"), name="shape")
    strides_obj = interface.get("strides")
    strides = None
    if strides_obj is not None:
        strides = _tuple_of_ints(strides_obj, name="strides")
    pointer, read_only = _parse_pointer(interface.get("data"))
    return _ArraySpec(pointer=pointer, shape=shape, strides=strides, dtype=dtype, read_only=read_only)


def _expected_strides(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    stride = itemsize
    expected: list[int] = []
    for dim in reversed(shape):
        expected.append(stride)
        stride *= dim
    return tuple(reversed(expected))


def _is_c_contiguous(
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None,
    itemsize: int,
) -> bool:
    if strides is None:
        return True
    if any(value < 0 for value in strides):
        return False
    return strides == _expected_strides(shape, itemsize)


def _resolve_channel_order(
    channel_order: ChannelOrder | str | None,
    channel_first: bool,
) -> ChannelOrder:
    if channel_order is None:
        return ChannelOrder.CHW if channel_first else ChannelOrder.HWC
    if channel_first:
        raise ValueError("Specify channel_order or channel_first, not both.")
    return ChannelOrder.parse(channel_order)


def _infer_image_shape(
    shape: tuple[int, ...],
    channel_order: ChannelOrder,
) -> tuple[int, int, int]:
    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3:
        if channel_order is not ChannelOrder.HWC:
            raise ValueError("Channel order CHW is not supported.")
        height, width, channels = shape
    else:
        raise ValueError("Array must have shape (H, W) or (H, W, C).")
    if height <= 0 or width <= 0:
        raise ValueError("Width and height must be positive.")
    if channels not in (1, 2, 3, 4):
        raise ValueError("Channels must be 1, 2, 3, or 4.")
    return height, width, channels


def _format_for_dtype(dtype: np.dtype, channels: int) -> int:
    if dtype == np.dtype("float32"):
        formats = {
            1: FORMAT_FLOAT,
            2: FORMAT_FLOAT2,
            3: FORMAT_FLOAT3,
            4: FORMAT_FLOAT4,
        }
    elif dtype == np.dtype("float16"):
        formats = {
            1: FORMAT_HALF,
            2: FORMAT_HALF2,
            3: FORMAT_HALF3,
            4: FORMAT_HALF4,
        }
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return formats[channels]


def _array_interface_for_backend(device: Device, array: object) -> Mapping[str, object]:
    backend = device.backend
    if backend is Backend.CPU:
        attr = "__array_interface__"
    elif backend is Backend.CUDA:
        attr = "__cuda_array_interface__"
    elif backend is Backend.HIP:
        if hasattr(array, "__hip_array_interface__"):
            attr = "__hip_array_interface__"
        else:
            attr = "__cuda_array_interface__"
    elif backend is Backend.SYCL:
        attr = "__sycl_usm_array_interface__"
    else:
        raise NotImplementedError(f"Backend {backend.value} buffer interface is not supported.")
    interface = getattr(array, attr, None)
    if not isinstance(interface, Mapping):
        raise TypeError(f"Object does not expose {attr}.")
    return interface


def _byte_strides(
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None,
    itemsize: int,
) -> tuple[int, int]:
    if strides is None:
        if len(shape) == 2:
            height, width = shape
            return width * itemsize, itemsize
        height, width, channels = shape
        return width * channels * itemsize, channels * itemsize
    if len(strides) == 2:
        return strides[0], strides[1]
    return strides[0], strides[1]


@dataclass(slots=True)
class Buffer(AutoReleaseByContextManaeger):
    device: Device
    buffer_delegate: object
    format: int
    channel_order: ChannelOrder
    width: int
    height: int
    channels: int
    byte_offset: int
    byte_pixel_stride: int
    byte_row_stride: int
    data_ptr: int
    dtype: np.dtype

    @property
    def channel_first(self) -> bool:
        """
        Indicate whether the buffer is channel-first.
        """
        return self.channel_order is ChannelOrder.CHW

    def release(self) -> None:
        """
        Buffer data is owned by the backing array; no explicit release required.
        """
        return None

    @classmethod
    def from_array(
        cls,
        device: Device,
        array: object,
        *,
        channel_order: ChannelOrder | str | None = None,
        channel_first: bool = False,
    ) -> Buffer:
        """
        Wrap an existing array buffer.
        """
        order = _resolve_channel_order(channel_order, channel_first)
        interface = _array_interface_for_backend(device, array)
        spec = _parse_array_interface(interface)
        dtype = _resolve_numpy_dtype(spec.dtype)
        height, width, channels = _infer_image_shape(spec.shape, order)
        if not _is_c_contiguous(spec.shape, spec.strides, dtype.itemsize):
            raise ValueError("Array must be C-contiguous.")
        byte_row_stride, byte_pixel_stride = _byte_strides(spec.shape, spec.strides, dtype.itemsize)
        format_value = _format_for_dtype(dtype, channels)
        return cls(
            device=device,
            buffer_delegate=array,
            format=format_value,
            channel_order=order,
            width=width,
            height=height,
            channels=channels,
            byte_offset=0,
            byte_pixel_stride=byte_pixel_stride,
            byte_row_stride=byte_row_stride,
            data_ptr=spec.pointer,
            dtype=dtype,
        )

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        channels: int = 3,
        channel_first: bool = False,
        device: Device | None = None,
        use_cupy: bool = False,
        dtype: object = np.float32,
        channel_order: ChannelOrder | str | None = None,
    ) -> Buffer:
        """
        Create a new buffer with allocated storage.
        """
        if device is None:
            raise ValueError("device must be provided.")
        order = _resolve_channel_order(channel_order, channel_first)
        if order is ChannelOrder.CHW:
            raise ValueError("Channel order CHW is not supported.")
        resolved_dtype = _resolve_numpy_dtype(dtype)
        channels_value = 1 if channels in (0, None) else channels
        if channels_value not in (1, 2, 3, 4):
            raise ValueError("channels must be 1, 2, 3, or 4.")
        shape = (height, width) if channels_value == 1 else (height, width, channels_value)

        if device.backend is Backend.CPU:
            buffer = np.zeros(shape=shape, dtype=resolved_dtype)
        elif device.backend in (Backend.CUDA, Backend.HIP):
            if use_cupy:
                cupy = _load_cupy()
                zeros = _require_callable(cupy, "zeros")
                buffer = zeros(shape, dtype=resolved_dtype)
            else:
                torch = _load_torch()
                if not _torch_cuda_available(torch):
                    raise RuntimeError("torch.cuda.is_available() is False.")
                if device.backend is Backend.HIP and not _torch_hip_available(torch):
                    raise RuntimeError("HIP backend requires a ROCm-enabled torch build.")
                device_name = "cuda"
                torch_dtype = _torch_dtype(torch, resolved_dtype)
                zeros = _require_callable(torch, "zeros")
                buffer = zeros(shape, dtype=torch_dtype, device=device_name)
        elif device.backend is Backend.SYCL:
            dpctl_tensor = _load_dpctl_tensor()
            zeros = _require_callable(dpctl_tensor, "zeros")
            buffer = zeros(shape, dtype=resolved_dtype)
        else:
            raise NotImplementedError(f"Backend {device.backend.value} buffer allocation is not supported.")

        return cls.from_array(device, buffer, channel_order=order)

    @classmethod
    def load(
        cls,
        device: Device,
        source: object,
        normalize: bool,
        copy_data: bool = True,
        *,
        channel_order: ChannelOrder | str | None = None,
        channel_first: bool = False,
        use_cupy: bool = False,
    ) -> Buffer:
        """
        Create a buffer from a PIL image, numpy array, or torch tensor.
        """
        order = _resolve_channel_order(channel_order, channel_first)
        if normalize and not copy_data:
            raise RuntimeError("normalize=True requires copy_data=True.")

        if isinstance(source, Image.Image):
            array = np.array(source)
            if normalize:
                if array.dtype == np.uint8:
                    array = array.astype(np.float32) / 255.0
                elif array.dtype == np.uint16:
                    array = array.astype(np.float32) / 65535.0
        elif isinstance(source, np.ndarray):
            array = source if not copy_data else np.array(source)
            if normalize:
                if array.dtype == np.uint8:
                    array = array.astype(np.float32) / 255.0
                elif array.dtype == np.uint16:
                    array = array.astype(np.float32) / 65535.0
        else:
            torch = _load_torch()
            tensor_type = _require_type(torch, "Tensor")
            if isinstance(source, tensor_type):
                if device.backend is Backend.CPU:
                    array = source.detach().cpu().numpy()
                    if normalize:
                        array = array.astype(np.float32) / 255.0 if array.dtype == np.uint8 else array
                else:
                    if not getattr(source, "is_cuda", False):
                        raise RuntimeError("Torch tensor must be on CUDA for GPU backends.")
                    tensor = source if not copy_data else _require_callable(torch, "tensor")(source)
                    if normalize:
                        dtype = getattr(tensor, "dtype", None)
                        uint8_dtype = getattr(torch, "uint8", None)
                        int16_dtype = getattr(torch, "int16", None) or getattr(torch, "short", None)
                        if uint8_dtype is not None and dtype == uint8_dtype:
                            tensor = tensor.float() / 255.0
                        elif int16_dtype is not None and dtype == int16_dtype:
                            tensor = tensor.float() / 65535.0
                    array = tensor
            else:
                raise NotImplementedError(f"Not implemented sharing buffer from {type(source)}")

        if device.backend is Backend.CPU:
            return cls.from_array(device, array, channel_order=order)
        if device.backend in (Backend.CUDA, Backend.HIP):
            if isinstance(array, np.ndarray):
                if use_cupy:
                    cupy = _load_cupy()
                    asarray = _require_callable(cupy, "asarray")
                    array = asarray(array, dtype=_resolve_numpy_dtype(array.dtype))
                else:
                    torch = _load_torch()
                    if not _torch_cuda_available(torch):
                        raise RuntimeError("torch.cuda.is_available() is False.")
                    if device.backend is Backend.HIP and not _torch_hip_available(torch):
                        raise RuntimeError("HIP backend requires a ROCm-enabled torch build.")
                    torch_dtype = _torch_dtype(torch, _resolve_numpy_dtype(array.dtype))
                    array = _require_callable(torch, "tensor")(array, device="cuda", dtype=torch_dtype)
            return cls.from_array(device, array, channel_order=order)
        if device.backend is Backend.SYCL:
            if isinstance(array, np.ndarray):
                dpctl_tensor = _load_dpctl_tensor()
                asarray = _require_callable(dpctl_tensor, "asarray")
                array = asarray(array, dtype=_resolve_numpy_dtype(array.dtype))
            return cls.from_array(device, array, channel_order=order)
        raise NotImplementedError(f"Backend {device.backend.value} buffer loading is not supported.")

    def to_tensor(self) -> object:
        """
        Returns the backing torch.Tensor when available.
        """
        torch = _load_torch()
        tensor_type = _require_type(torch, "Tensor")
        if isinstance(self.buffer_delegate, tensor_type):
            return self.buffer_delegate
        raise RuntimeError("Buffer is not backed by a torch.Tensor.")

    def to_array(self) -> np.ndarray:
        """
        Returns a numpy.ndarray copy of the buffer when possible.
        """
        if isinstance(self.buffer_delegate, np.ndarray):
            return self.buffer_delegate
        torch = _load_torch()
        tensor_type = _require_type(torch, "Tensor")
        if isinstance(self.buffer_delegate, tensor_type):
            return self.buffer_delegate.detach().cpu().numpy()
        raise RuntimeError("Buffer cannot be converted to a numpy array.")


class Filter(AutoReleaseByContextManaeger):
    def __init__(self, device: Device, type: str) -> None:
        r"""
        Args:
            device : oidn.Device
            type   : 'RT' or 'RTLightmap'
        """
        self.device = device
        self.type = type
        self.__filter_handle = NewFilter(device_handle=device.device_handle, type=type)
        if not self.__filter_handle:
            raise RuntimeError("Can't create filter")
        device.raise_if_error()
        CommitFilter(self.__filter_handle)
        self._image_size: tuple[int, int] | None = None

    @property
    def filter_handle(self) -> int:
        r"""
        Returns the handle of filter.
        """
        return self.__filter_handle

    def release(self) -> None:
        r"""
        Call ReleaseFilter with self.filter_handle
        """
        if self.__filter_handle:
            ReleaseFilter(self.__filter_handle)
        self.__filter_handle = 0

    def set_image(self, name: str, buffer: Buffer) -> None:
        r"""
        Set image buffer for the filter.

        Args:
            name    : color/albedo/normal/output
            buffer  : Buffer object
        """
        name_value = name.strip().lower()
        if name_value not in {"color", "albedo", "normal", "output"}:
            raise ValueError(f"Unsupported image name: {name}")
        if name_value in {"albedo", "normal"} and self.type != "RT":
            raise RuntimeError(f"{name_value} is only supported for RT filters.")
        if buffer.device.backend is not self.device.backend:
            raise RuntimeError("Buffer backend does not match device backend.")
        if buffer.channel_order is not ChannelOrder.HWC:
            raise RuntimeError("Buffer must use HWC channel order.")
        if buffer.channels != 3:
            raise RuntimeError("Buffers must have 3 channels for OIDN filters.")

        if self._image_size is None:
            self._image_size = (buffer.width, buffer.height)
        if self._image_size != (buffer.width, buffer.height):
            raise RuntimeError("All filter images must share the same dimensions.")

        functions = _ffi.get_functions()
        functions.oidnSetSharedFilterImage(
            self.__filter_handle,
            name_value.encode("ascii"),
            buffer.data_ptr,
            buffer.format,
            buffer.width,
            buffer.height,
            buffer.byte_offset,
            buffer.byte_pixel_stride,
            buffer.byte_row_stride,
        )

    def set_images(
        self,
        *,
        color: Buffer,
        output: Buffer,
        albedo: Buffer | None = None,
        normal: Buffer | None = None,
    ) -> None:
        """
        Set the filter images in a single call.
        """
        self.set_image("color", color)
        if albedo is not None:
            self.set_image("albedo", albedo)
        if normal is not None:
            self.set_image("normal", normal)
        self.set_image("output", output)

    def execute(self) -> None:
        r"""
        Run the filter, wait until finished.
        """
        if not self.__filter_handle:
            raise RuntimeError("Invalid filter handle")
        CommitFilter(self.__filter_handle)
        ExecuteFilter(self.__filter_handle)
        self.device.raise_if_error()


def _ensure_buffer(
    device: Device,
    value: Buffer | object,
    *,
    channel_order: ChannelOrder | str | None,
    channel_first: bool,
) -> Buffer:
    if isinstance(value, Buffer):
        return value
    return Buffer.from_array(
        device,
        value,
        channel_order=channel_order,
        channel_first=channel_first,
    )


def denoise(
    color: Buffer | object,
    *,
    albedo: Buffer | object | None = None,
    normal: Buffer | object | None = None,
    output: Buffer | object | None = None,
    device: Device | None = None,
    backend: Backend | str | None = None,
    options: DeviceOptions | None = None,
    filter_type: str = "RT",
    channel_order: ChannelOrder | str | None = None,
    channel_first: bool = False,
) -> Buffer:
    """
    Convenience API for denoising with optional auxiliary images.
    """
    created_device = False
    if device is None:
        device = Device(backend=backend, options=options)
        created_device = True

    try:
        color_buffer = _ensure_buffer(
            device,
            color,
            channel_order=channel_order,
            channel_first=channel_first,
        )
        albedo_buffer = (
            _ensure_buffer(device, albedo, channel_order=channel_order, channel_first=channel_first)
            if albedo is not None
            else None
        )
        normal_buffer = (
            _ensure_buffer(device, normal, channel_order=channel_order, channel_first=channel_first)
            if normal is not None
            else None
        )
        if output is None:
            output_buffer = Buffer.create(
                color_buffer.width,
                color_buffer.height,
                channels=color_buffer.channels,
                device=device,
                dtype=color_buffer.dtype,
                channel_order=channel_order,
                channel_first=channel_first,
            )
        else:
            output_buffer = _ensure_buffer(
                device,
                output,
                channel_order=channel_order,
                channel_first=channel_first,
            )

        with Filter(device, filter_type) as filter_obj:
            filter_obj.set_images(
                color=color_buffer,
                output=output_buffer,
                albedo=albedo_buffer,
                normal=normal_buffer,
            )
            filter_obj.execute()
        return output_buffer
    finally:
        if created_device:
            device.release()
