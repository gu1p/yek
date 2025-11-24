"""Factory for platform-specific keyboard state implementations."""

import sys

from yek.platforms.base import KeyboardState
from yek.platforms.common import PynputKeyboardState
from yek.platforms.darwin import MacKeyboardState
from yek.platforms.linux import LinuxKeyboardState


def create_keyboard_state(max_buffer_len: int = 100) -> KeyboardState:
    """Return a keyboard state suitable for the current platform."""
    if sys.platform == "darwin":
        return MacKeyboardState(max_buffer_len)
    if sys.platform.startswith("linux"):
        return LinuxKeyboardState(max_buffer_len)

    # Fallback: use generic pynput-backed implementation
    return PynputKeyboardState(max_buffer_len)


__all__ = ["KeyboardState", "create_keyboard_state"]
