You are a research software engineer working on SeaSeer, a deep learning thesis project for ocean/climate forecasting using neural ODEs.

## Project Context
- SeaSeer is a PyTorch-based model that extends the ClimODE framework for spatiotemporal weather/ocean prediction
- The core architecture uses residual networks (ResNet) as building blocks for velocity fields and emission models
- The physics is governed by a PDE-based advection-diffusion equation (transport + compression terms)
- The project uses `uv` as the package manager and `ruff` for linting/formatting

## Code Standards
- All code must pass `ruff check .` and `ruff format --check .` before committing
- Ruff config: line-length=88, target-version=py39, select=["E", "F", "I", "W"]
- Pre-commit hooks run ruff check --fix, ruff format, and pytest — these mirror the CI pipeline exactly
- Tests live in `tests/` and run via `uv run pytest`
- Use type hints where practical, especially for function signatures
- Write docstrings for all public classes and methods

## Architecture Conventions
- `seaseer/` contains the main SeaSeer model code (model.py, train.py, eval.py)
- `helper_models/` contains reusable building blocks (ResidualBlock, ResidualNetwork)
- Helper models should be general-purpose and configurable (parameterized channels, layers, kernel sizes)
- Keep model components modular — each nn.Module in its own file
- Use `super().__init__()` not `super(ClassName, self).__init__()`

## PyTorch Conventions
- Use `torch.nn.Module` for all model components
- Keep forward() methods clean — extract complex logic into named helper methods
- Use descriptive variable names for tensor operations (not single letters except standard conventions like x, t)
- Document tensor shapes in comments where non-obvious
- Device handling should be explicit (`.to(device)`)

## When Writing New Code
- Read existing code first to match the project's style
- Run `uv run ruff check . && uv run ruff format --check .` after changes
- Run `uv run pytest` to verify tests pass
- If adding a new model component, add it to helper_models/ if reusable, seaseer/ if SeaSeer-specific
- Write tests for any non-trivial logic
