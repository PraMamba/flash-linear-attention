# Repository Guidelines

## Project Structure & Module Organization

`fla/` contains the Python package. Core Triton/PyTorch operators live in `fla/ops/`, reusable neural-network modules in `fla/modules/`, model definitions in `fla/models/`, and higher-level layers in `fla/layers/`. Tests mirror these areas under `tests/ops/`, `tests/modules/`, `tests/models/`, and `tests/layers/`; shared pytest setup is in `tests/conftest.py`. Use `benchmarks/` for performance scripts, `evals/` for evaluation harnesses, `examples/` for usage notes, and `scripts/` for maintenance utilities.

## Build, Test, and Development Commands

- `pip install -e '.[test]'` installs the package in editable mode with pytest.
- `pip install pre-commit && pre-commit install` enables local lint/format hooks.
- `pre-commit run --all-files` runs repository linting, formatting, YAML/TOML checks, and safety hooks.
- `pytest tests/` runs the full test suite; most tests require a GPU with Triton support.
- `pytest tests/ops/test_delta.py::test_chunk -v` runs a focused test while developing one operator.

## Coding Style & Naming Conventions

Target Python 3.10+. Ruff and autopep8 enforce style through pre-commit; line length is 127 characters, and `fla` is treated as first-party for import sorting. Use PascalCase for classes, `snake_case` for functions, UPPER_SNAKE_CASE for constants, and leading underscores for private helpers. New source files should include the project copyright header shown in `CONTRIBUTING.md`.

## Testing Guidelines

Pytest is the test framework. Add tests with the matching area, such as `tests/ops/test_<op_name>.py` for new kernels. Operator tests should compare optimized implementations against naive or recurrent PyTorch references, cover forward and backward gradients, use `torch.manual_seed(42)`, and prefer `fla.utils.assert_close`, `device`, and `device_platform`. Include non-power-of-two sequence lengths and skip unsupported platforms explicitly.

## Commit & Pull Request Guidelines

Use square-bracket commit tags seen in history and `CONTRIBUTING.md`, for example `[Fix]`, `[Docs]`, `[CI]`, `[Test]`, `[Perf]`, `[Ops]`, `[Model]`, or component tags such as `[KDA]` and `[Conv]`. Keep PRs focused. PR descriptions should include a summary, test plan, and breaking-change notes when relevant. Include tests for code changes, run `pre-commit run --files <your_files>`, and use `[skip test]` only for documentation-only commits that can skip GPU tests.

## Configuration & Environment Notes

See `ENVs.md` for supported environment variables. Do not commit local caches, build outputs, virtual environments, or `.omx/` agent state.
