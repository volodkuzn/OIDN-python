# Implementation Plan

## Goals
- Use the latest OIDN from the repo, version pinned as a git submodule.
- Provide native support for every backend OIDN exposes on the target platform.
- Expose backend selection at initialization (explicit, validated).
- Support auxiliary images (albedo, normal) in the Python API.
- Meet repo gates: pyproject/uv, typed code, ruff, mypy, tests with 100% coverage.

## Constraints and assumptions
- Backend list is derived from the OIDN C API in the submodule to avoid hardcoding.
- GPU support requires optional Python deps and the matching runtime/toolkit on the host.
- Tests must be deterministic; GPU paths are exercised via fakes/mocks when hardware is unavailable.

## Milestones

### 1) Repo modernization and scaffolding
- Add `pyproject.toml` with ruff/mypy/pytest/coverage config and uv metadata.
- Move package to `src/oidn` and migrate imports; keep `oidn/` name.
- Replace `setup.py` flow with pyproject build backend (document in README).
- Add `tests/unit` and `tests/integration` scaffolding; add `tests/fixtures`.
- Define typed public API surface and module layout plan.

Status (2026-01-22)
- [x] Added `pyproject.toml` with ruff/mypy/pytest/coverage config and uv metadata.
- [x] Moved package to `src/oidn` and updated local import paths in docs/scripts.
- [x] Removed `setup.py` flow and documented pyproject usage in README.
- [x] Added `tests/unit`, `tests/integration`, and `tests/fixtures` scaffolding.
- [x] Added `ARCH.md` describing the typed public API surface and module layout plan.
- [x] Generate `uv.lock` via `uv lock` when network access is available.

Deliverables
- `pyproject.toml`, `uv.lock`, and src-layout package.
- Minimal test suite wiring with coverage enforcement.

### 2) OIDN submodule and native build pipeline
- Add OIDN repo as submodule (e.g., `oidn_cpp/`) from `https://github.com/RenderKit/oidn.git`.
- Track the latest OIDN tag/commit in the submodule and document update steps.
- Add build scripts to compile OIDN for each platform and backend:
  - CPU always enabled.
  - GPU backends enabled when dependencies are present (CUDA, HIP, SYCL).
- Copy built shared libraries into the package at build time.
- Add CI matrix to build wheels with bundled OIDN libs per OS/arch.

Status (2026-01-22)
- [x] Added `oidn_cpp` submodule pinned to the current commit.
- [x] Added `scripts/build_oidn.py` to build OIDN via the submodule build script.
- [x] Added `scripts/stage_oidn_libs.py` to stage shared libraries into `src/oidn/lib.*`.
- [x] Documented submodule update and build steps in README.
- [x] Added a GitHub Actions wheel build matrix for Linux, Windows, and macOS (x64/arm64).
- [ ] Expand CI to build GPU backends once runners/toolchains are available.

Deliverables
- Submodule checked in and documented.
- Deterministic build artifacts for each platform/backend combination.

### 3) Typed C API binding layer
- Refactor `capi.py` into a typed FFI module (ctypes) with narrow, explicit signatures.
- Auto-load platform-specific shared libraries from package data.
- Provide typed error handling helpers and lifecycle helpers.
- Avoid Any by isolating untyped boundary code in a dedicated module.

Status (2026-01-22)
- [x] Added `src/oidn/_ffi.py` with typed bindings, library auto-loading, and error helpers.
- [x] Switched `src/oidn/capi.py` to use the new FFI layer with lazy initialization.
- [x] Removed hardcoded library version loading in `src/oidn/__init__.py`.
- [x] Added unit tests covering library loading and device error formatting.

Deliverables
- `oidn/_ffi.py` (or similar) with typed bindings and tests.

### 4) Backend abstraction and device selection API
- Check the current device implementation of the device selection. Proceed with next bullet-point only if needed
- Introduce a `Backend` enum mapped to OIDN device types.
- Implement `Device(backend=Backend.CPU, ...)` with validation and defaults.
- Add `available_backends()` and `is_backend_available(backend)` helpers.
- Add backend-specific runtime checks (driver/toolkit presence, array protocol support).
- Allow user to pass backend-specific config at initialization when needed.

Status (2026-01-22)
- [x] Reviewed existing device selection and confirmed backend-aware API was required.
- [x] Added `Backend` enum mapped to OIDN device types.
- [x] Updated `Device` initialization to validate backend availability and apply defaults.
- [x] Added `available_backends()` and `is_backend_available()` helpers.
- [x] Implemented backend runtime checks (torch/dpctl/platform and device probe).
- [x] Added `DeviceOptions`/`extra_params` for backend-specific configuration.

Deliverables
- Public API supporting explicit backend selection and discovery.

### 5) Buffer and filter support with auxiliary images
- Define a typed `Buffer` abstraction supporting:
  - CPU: numpy arrays via `__array_interface__`.
  - CUDA/HIP: torch or cupy via `__cuda_array_interface__`/`__hip_array_interface__`.
  - SYCL: dpctl or similar via `__sycl_usm_array_interface__`.
- Enforce dtype and shape validation, explicit channel order, and contiguous layout.
- Extend `Filter` wrapper to accept `color`, `albedo`, `normal`, `output`.
- Provide a high-level convenience API for denoising with optional auxiliary images.

Status (2026-01-22)
- [x] Added `Buffer` with array-interface validation for CPU/CUDA/HIP/SYCL and strict dtype/shape checks.
- [x] Enforced HWC channel order and contiguous layouts with stride-aware metadata.
- [x] Updated `Filter` to validate auxiliary images and set color/albedo/normal/output buffers.
- [x] Added `denoise` convenience API with optional aux images and buffer creation.
- [x] Added unit tests covering buffer interfaces, filter validation, and denoise flow.

Deliverables
- `Device`, `Buffer`, `Filter` with backend-aware behavior and aux image support.

### 6) Tests and coverage
- Unit tests for device creation, backend selection, error handling, and buffer validation.
- Integration test for a tiny denoise run on CPU with auxiliary images.
- GPU paths covered via mocks/fakes for array interface extraction and pointer handling.
- Maintain 100% line coverage for production code.

Status (2026-01-22)
- [x] Added unit tests for device selection, backend checks, and buffer validation.
- [x] Added integration test for CPU denoise with auxiliary images.
- [x] Added mocked GPU/SYCL array-interface tests for pointer/stride handling.
- [x] Run full coverage and close remaining gaps (if any).

Deliverables
- `tests/unit` and `tests/integration` covering all public APIs and key branches.

### 7) Documentation and examples
- Update README with backend selection and auxiliary image examples.
- Regenerate `APIs.md` if API surface changes.
- Add minimal CLI usage if needed, otherwise document script usage.
- Document how to update the OIDN submodule and rebuild libs.

Status (2026-01-22)
- [x] Updated README with backend selection, aux image usage, and buffer layout notes.
- [x] Documented submodule update and build scripts in README.
- [x] Regenerated `APIs.md` after API changes.

Deliverables
- Updated docs and example snippets aligned with the new API.

### 8) Code monitoring
- Add a pre-commit to the package
- Add `ruff format` to pre-commit
- Add `ruff check --fix` to pre-commit, and add safe defaults to pyproject.toml
- There are lots of star imports, consider disabling F405
- Run mypy to check feasibility of adding it to the project
- Correct all ruff and mypy (if added) errors
- Add Github CI

Status (2026-01-22)
- [x] Added `.pre-commit-config.yaml` with `ruff` and `ruff-format` hooks.
- [x] Configured `ruff` hook to run with `--fix` and `--exit-non-zero-on-fix`.
- [x] Disabled `F403/F405` for `src/oidn/__init__.py` to allow star imports.
- [ ] Ran `uv run mypy src tests` (blocked by uv panicking in `system-configuration` on this host).
- [x] Added GitHub Actions CI to run ruff format/check, mypy, and pytest via uv.

Deliverables
- `.pre-commit-config.yaml` and ruff/mypy defaults in `pyproject.toml`.
- GitHub Actions CI workflow enforcing lint, typing, and tests.

### 9) Mypy and typing hardening plan
- Target mypy configuration:
  - Keep `python_version` aligned with the minimum supported runtime (bump to 3.10+ everywhere).
  - Enable strict-ish gates for production code: `disallow_untyped_defs`, `check_untyped_defs`,
    `disallow_any_generics`, `warn_unused_ignores`, `warn_return_any`, `warn_redundant_casts`,
    `warn_unreachable`, `no_implicit_optional`, `strict_optional`, `strict_equality`.
  - Add module overrides only for third-party tooling (`pytest`, `_pytest`) and optional deps
    (`torch`, `dpctl`) when they are not installed.
- Typing boundaries / ignore points for C/C++ interop:
  - `src/oidn/_ffi.py`: dynamic `ctypes.CDLL` attribute access and function pointer signatures.
    Use runtime checks + `cast`; allow local `# type: ignore[<code>]` only where ctypes cannot
    express types.
  - `src/oidn/capi.py`: `RawFunctions.oidn*` attribute access to ctypes-bound functions.
    Prefer a typed Protocol or `BoundFunctions`; otherwise use targeted `# type: ignore[attr-defined]`.
  - `src/oidn/__init__.py`: array-interface dictionaries (`__array_interface__`,
    `__cuda_array_interface__`, `__hip_array_interface__`, `__sycl_usm_array_interface__`).
    Use runtime validation and `Mapping[str, object]` casts to avoid leaking `Any`.
  - `src/oidn/_backends.py`: dynamic `torch` attribute access (e.g., `torch.cuda`, `torch.version.hip`).
    Keep behind runtime checks and narrow casts/Protocols.
- Roadblocks:
  - Optional deps not installed in CI/dev environments (torch/dpctl) can trigger import errors.
  - `ctypes` dynamic attributes are difficult to type without casts/Protocols.
  - Array-interface dicts are untyped; requires explicit validation and casting.
  - Python version mismatch if `|` unions are used while `requires-python` stays at 3.9.
- Deliverables:
  - Mypy configuration updated in `pyproject.toml` with explicit overrides and strict gates.
  - Typed Protocols or .pyi helpers for optional deps and ctypes functions.
  - Local, code-commented `# type: ignore[<code>]` only in boundary modules.
  - Clean `uv run mypy src tests` in CI with documented exceptions.

Status (2026-01-22)
- [x] Bumped minimum Python version to 3.10 and aligned mypy `python_version`.
- [x] Enabled strict-ish mypy gates (untyped defs, any generics, redundant casts, unreachable, strict equality).
- [x] Added mypy overrides for pytest internals and optional deps (torch/dpctl).
- [x] Added return type annotations for boundary helpers and test fakes.
- [x] Added torch tensor Protocols, tighter dtype validation, and typed context manager returns.
- [x] Added targeted mypy ignores at untyped boundaries and invalid-type test cases.
- [ ] Confirmed `uv run mypy src tests` passes with the stricter config (blocked by uv cache permissions).

## Acceptance criteria
- Package builds from pyproject and passes ruff, mypy, and pytest with 100% coverage.
- Backend selection is explicit and validated at initialization.
- CPU and GPU paths load the correct OIDN libs; device creation fails with actionable errors if backend is unavailable.
- Auxiliary images (albedo, normal) are supported in `Filter` and high-level denoise API.
- OIDN version is sourced from the pinned submodule, not hardcoded.
