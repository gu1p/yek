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

## Avoiding collisions

- The engine rejects overlapping routes at registration time. If two shortcuts could fire on the same keys (including holds), a `ValueError` is raised with the colliding matcher details and signatures.
- Prefer `Hold(..., only=True)` for modifiers, or add an extra key to disambiguate similar combos.
- You can override ordering with `@app.on(..., priority=10)` if you intentionally want a general route to sit behind a specific one.
- Validate a routes file without running it: `python -m yek.shortcuts examples/open_window.py` (exits non-zero on collision) or `make check-shortcuts FILE=path/to/routes.py`.

## Feature examples

```python
from yek import App
from yek.keys import Cmd, Shift, Ctrl, Left, Char, String
from yek.matchers import Hold
from yek.time import Wait

app = App()

# Simple chord: Cmd+Shift+O
@app.on(Cmd / Shift / Char("o"))
def open_file(_): ...

# Case-insensitive char: matches "a" or "A"
@app.on(Char("a"))  # default case=False
def any_a(_): ...

# Exact string typed in order (skips non-char keys)
@app.on(String("hello"))
def greet(_): ...

# Hold-only guard (prevents collisions): Ctrl must be the only key held, then Left tap
@app.on(Hold(Ctrl, only=True) / Left)
def ctrl_left_only(_): ...

# Timed press-release: tap Left for between 50–250ms
@app.on(Left @ (0.05, 0.25))
def left_tap(_): ...

# Sequence with timing gaps: Shift then Left within 500ms (0.5s)
@app.on((Shift / Left) @ Wait(0.5))
def nudge_left(_): ...

# Alternation: Cmd+Shift+S or Cmd+Shift+Alt+S
@app.on((Cmd / Shift / Char("s")) | (Cmd / Shift / Char("s") / Char("alt")))
def save(_): ...

# Priority override: force this to run before shorter/same-length routes
@app.on(Cmd / Shift / Char("z"), priority=20)
def undo_special(_): ...

# Held combo that repeats but fires every 200ms at most
@app.on(Hold(Cmd, Shift).throttle(every_ms=200))
def scrub(_): ...

if __name__ == "__main__":
    app()
```

## Notes on uv

- With a PEP 621 `[project]` table in `pyproject.toml`, `uv run ...` works without extra flags.
- Prefer `make install` / `make test`, or run ad-hoc commands with `uv run -- python your_script.py` if you want to override the interpreter.
