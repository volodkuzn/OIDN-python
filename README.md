# OIDN-python
Python binding for [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn).

The latest OIDN source is pinned as a git submodule under `oidn_cpp`, and wheels bundle
the corresponding native libraries. Backend availability depends on the packaged libs
and the runtime toolchains on your machine.

## Install

```
pip install oidn
```

Optional runtime dependencies (only needed for the backend you use):
- CUDA/HIP: `torch` (install with `pip install oidn[cuda]` or `pip install oidn[hip]`).
- CUDA/HIP with CuPy: install the appropriate `cupy` package for your CUDA version.
- SYCL: `dpctl` (install with `pip install oidn[sycl]`).

## Quick start

```python
import numpy as np
import oidn

color = np.zeros((256, 256, 3), dtype=np.float32)
albedo = np.zeros_like(color)
normal = np.zeros_like(color)

output = oidn.denoise(color, albedo=albedo, normal=normal, backend="cpu")
```

## Backend selection

```python
import oidn

print(oidn.available_backends())
print(oidn.is_backend_available("cuda"))

device = oidn.Device(backend="cuda")
```

## Buffers and layout

Buffers must be HWC (height, width, channels) and C-contiguous. Use:
- `Buffer.from_array(device, array)` to wrap an existing array.
- `Buffer.create(...)` to allocate a new buffer.
- `Buffer.load(...)` for PIL images or numpy arrays (with optional normalization).

## Auxiliary images

```python
import numpy as np
import oidn

device = oidn.Device("cpu")
color = oidn.Buffer.from_array(device, np.zeros((64, 64, 3), dtype=np.float32))
albedo = oidn.Buffer.from_array(device, np.zeros((64, 64, 3), dtype=np.float32))
normal = oidn.Buffer.from_array(device, np.zeros((64, 64, 3), dtype=np.float32))
output = oidn.Buffer.create(64, 64, device=device)

with oidn.Filter(device, "RT") as filter_obj:
    filter_obj.set_images(color=color, albedo=albedo, normal=normal, output=output)
    filter_obj.execute()
```

## Architecture overview

- `src/oidn/_ffi.py` binds the OIDN C API via ctypes and loads shared libraries.
- `src/oidn/_backends.py` provides backend discovery and runtime checks.
- `src/oidn/__init__.py` exposes the public API (`Device`, `Buffer`, `Filter`, `denoise`).
- `generate_doc.py` regenerates `APIs.md` via module introspection.

## Development

### Submodule setup

```
git lfs install
git submodule update --init --recursive
```

### Update the OIDN submodule

```
git -C oidn_cpp fetch --tags
git -C oidn_cpp checkout <tag-or-commit>
git add oidn_cpp
```

### Build and stage native libraries

```
python scripts/build_oidn.py --backends cpu --stage
```

To enable GPU backends, add `cuda`, `hip`, `sycl`, or `metal` to `--backends` and
install the required toolchains described in `oidn_cpp/doc/compilation.md`.
Use `python scripts/stage_oidn_libs.py --clean` to refresh the staged libraries.
By default, build outputs are written under `oidn_cpp/build` and `oidn_cpp/install`.
Metal builds require the Xcode Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`).

### API docs

```
python generate_doc.py
```

## Known limitations
- HWC layout only (CHW is not supported).
- Metal buffer allocation is not exposed yet.
- GPU backends require the matching runtime/toolkit on the host.

## License

Apache 2.0

## API Document

See [here](APIs.md).
