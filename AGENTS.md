# AGENTS.md

This repo reimplements a scientific paper as a production-ready Python package **and** a reproducible research artifact.

Everything in this file is written for “agents” (humans or AI) doing changes in the repo: how to structure code, how to verify correctness, and what “done” means.

---

## Project snapshot

- **Project name:** `Python Binding of Intel Open Image Denoise`
- **Goal:** provide a portable and concise pythonic library for oidn usage which supports all devices supported by original library and uses the latest version of oidn
- **Non-goals (explicit):** `provide a full python API for the library`

---

## Hard requirements (do not violate)

1. **pyproject.toml-based** repository (no setup.py legacy flow).
2. **All code typed.** Untyped code is allowed *only* at boundaries with untyped third-party libs (and must be isolated).
3. **Type checking:** `mypy` (treat as a gate).
4. **Style/lint/format:** `ruff` (treat as a gate).
5. **Dependencies:** managed with **uv** (lock + sync; do not hand-edit venvs).
6. **TDD:** tests first (or at least test + implementation in the same diff with a clear red→green story).
7. **100% test coverage** for production code (line coverage; add branch coverage if feasible).

If you think a requirement blocks a necessary change, create an ADR (see below) and propose an alternative—don’t silently weaken gates.

---

## Repo layout (proposed)

.
├─ pyproject.toml
├─ uv.lock
├─ README.md
├─ AGENTS.md
├─ ARCH.md
├─ IMPLEMENTATION_PLAN.md
├─ TESTING.md
├─ oidn/
├─ oidn_cpp/
├─ tasks/
│  ├─ 000-repository-setup.md
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ fixtures/
├─ docs/
│  └─ adr/
└─ .gitignore

## Repo layout (current)

.
├─ AGENTS.md
├─ APIs.md
├─ ARCH.md
├─ .github/
│  └─ workflows/
├─ IMPLEMENTATION_PLAN.md
├─ LICENSE
├─ README.md
├─ build_support.py
├─ generate_doc.py
├─ oidn_cpp/
├─ pyproject.toml
├─ scripts/
├─ src/
│  └─ oidn/
│     ├─ __init__.py
│     ├─ __main__.py
│     ├─ capi.py
│     ├─ constants.py
│     ├─ lib.linux.x64/
│     ├─ lib.macos.aarch64/
│     ├─ lib.macos.x64/
│     └─ lib.win.x64/
└─ tests/
   ├─ DenoiseCornellBox/
   │  ├─ DenoiseCornellBox.py
   │  ├─ DenoiseCornellBox2.py
   │  ├─ CornellBoxNoisy.png
   │  └─ CornellBoxDenoisedAsExample.png
   ├─ fixtures/
   ├─ integration/
   └─ unit/

---

## Architecture
- `src/oidn/capi.py` is the ctypes binding layer: it loads function pointers into `RawFunctions` and exposes thin, typed-ish wrappers around the OIDN C API, including helpers for numpy buffers and generic array-interface buffers.
- `src/oidn/__init__.py` is the package entrypoint: it loads the platform-specific shared libraries from `src/oidn/lib.*`, initializes the binding layer, and re-exports the C API wrapper functions.
- Pythonic API lives in `src/oidn/__init__.py` as `Device`, `Filter`, and `Buffer` classes; they wrap raw handles, enforce device/buffer compatibility, and manage lifetimes via context managers.
- Buffer storage is CPU-first with NumPy arrays; CUDA buffers are supported via optional Torch tensors and `__cuda_array_interface__` to pass device pointers to the C API.
- `src/oidn/constants.py` defines the public enum-like constants used by both the bindings and the higher-level API.
- `src/oidn/__main__.py` is a stub CLI entrypoint; example usage lives under `tests/DenoiseCornellBox/`.
- `generate_doc.py` introspects the module to regenerate `APIs.md`.

---

## CLI and scripts
- Prefer console entry points in pyproject.toml
- Keep `scripts/` as thin, typed wrappers for convenience; no business logic.


---

## Use uv for everything

```
# tests
uv run pytest

# tests + coverage
uv run pytest --cov=src/<package_name> --cov-report=term-missing

# typing
uv run mypy src tests

# static analysis
uv run pyscn analyze --json .

# lint
uv run ruff check .

# safe autofix
uv run ruff check . --fix

# format
uv run ruff format .
```

### Pre-commit
- Run `uv run pre-commit run --all-files` on task completion and fix any issues it reports.
- Run pre-commit as the agent; developers do not need to check the results.
- Do not commit changes unless explicitly requested.

---

## Typing policy

Defaults:
  - Prefer small, typed, mostly-pure functions.
  - Avoid Any. If Any comes from an untyped dependency, contain it in _boundaries/ and convert to typed models.
  - Prefer dataclass(slots=True, frozen=True) (or similar) for stable domain models.
  - Do not use dataclasses with a single field and no methods; use the underlying type or a type alias instead (do not add methods just to bypass this rule).
  - Prefer dataclasses over dicts when passing objects between functions
  - Prefer PyTorch for tensor manipulation and data processing; avoid hand-written implementations when library primitives exist, and document any necessary hand-rolled operations.
  - Prefer Protocol for pluggable components.
  - Prefer numpy.typing and/or typing_extensions where appropriate.
  - For _boundaries with untyped libraries: check return types at runtime before casting, cast to concrete types (no cast to Any), and use local # type: ignore[<code>] only when a concrete type cannot be expressed.
  - Document shape of the arrays
  - Add typings for all bindings

Allowed:
  - Thin wrapper module inside binding code.
  - Narrow cast() at the boundary after a runtime type check (no cast to Any).
  - Local # type: ignore[<code>] with a short explanation.

Not allowed:
  - Any leaking into core/.
  - Blanket type: ignore without an error code.
  - Loosening mypy globally to accommodate one library.
  - Numeric / stochastic code
  - Make randomness explicit (pass seed or RNG).
  - Use tolerances for floating comparisons with rationale.
  - Prefer deterministic tests; keep stochastic tests tightly bounded.

---

## Ruff policy
- Ruff is both formatter and linter.
- Keep configuration centralized in pyproject.toml.
- If ignoring a rule, do it narrowly (single line) and explain why.

---

## What to test
- Unit tests: equations, invariants, edge cases.
- Integration tests: smallest end-to-end pipeline on tiny fixtures.
- Regression tests: every bugfix gets a test that fails before the fix.
- Repro checks (lightweight): repro scripts run and output expected keys/shapes (not full-scale training unless required).
- Coverage policy
- Production code in src/material_reconstruction must remain 100% covered.
- Avoid # pragma: no cover. If unavoidable, justify it and record in an ADR.

---

## ADRs

Create ADRs in docs/adr/NNNN-title.md:
- Context
- Decision
- Alternatives considered
- Consequences

Write an ADR when choosing:
- compute stack (numpy vs torch vs jax),
- config approach (dataclasses vs pydantic vs hydra),
- CLI (none vs argparse vs typer),
- serialization formats,
- any exception to typing/coverage gates.

---

## Commit checklist
Copy into commit description:
  -[ ] tasks/xxx-taskname.md completed
  -[ ] Tests added/updated (TDD slice)
  -[ ] uv run ruff format .
  -[ ] uv run ruff check .
  -[ ] uv run mypy src tests
  -[ ] uv run pyscn analyze --json .
  -[ ] uv run pytest --cov=src/<package_name> --cov-report=term-missing
  -[ ] ADR added (if architecture/behavior decision)
