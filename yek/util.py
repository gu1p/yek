# pylint: disable=duplicate-code
"""Utility helpers for keyboard state and function metadata."""

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable

from yek.events import KeyEvent
from yek.platforms import KeyboardState, create_keyboard_state

__all__ = ["KeyboardStateMap", "KeyEvent"]

_DEBUG = os.environ.get("DEBUG", False)


def _debug(msg):
    if _DEBUG:
        print(msg)


class KeyboardStateMap(KeyboardState):
    """
    Factory wrapper returning a platform-specific keyboard state implementation.
    """

    def __new__(cls, max_buffer_len: int = 100):  # type: ignore[override]
        return create_keyboard_state(max_buffer_len)


@dataclass
class FunctionDetails:
    """Captured metadata about a callable."""
    name: str
    line: int
    file: str


def get_function_details(obj: Callable[..., Any]) -> FunctionDetails:
    """Extract function name, line number, and file path."""
    # Check if the object is a function or method
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        name = obj.__name__
        line_number = inspect.getsourcelines(obj)[1]
        file_name = inspect.getfile(obj)
        return FunctionDetails(name=name, line=line_number, file=file_name)

    raise ValueError(f"Unsupported object type {type(obj)} - expected function or method")
