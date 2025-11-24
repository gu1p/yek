# yek

_Hotkeys for super-humans_

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. `make install` (creates `.venv` and installs deps with uv).

## Development

- `make test` runs the unittest suite inside the env.
- Run examples with `./.venv/bin/python examples/open_window.py`.

Platform backends:
- macOS: uses a native listener and will open the Input Monitoring privacy pane so you can grant access.
- Linux: uses pynput; grant access if your desktop environment prompts.

> macOS/Linux may require granting keyboard monitoring/assistive access for global hotkey capture. On macOS, the app will open the Input Monitoring privacy panel to help you approve it.

## Notes on uv

- With a PEP 621 `[project]` table in `pyproject.toml`, `uv run ...` works without extra flags.
- Prefer `make install` / `make test`, or run ad-hoc commands with `uv run -- python your_script.py` if you want to override the interpreter.
