import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from yek.events import KeyEvent
from yek.platforms import create_keyboard_state, KeyboardState

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
    name: str
    line: int
    file: str


def get_function_details(obj: Callable[[...], Any]) -> FunctionDetails:
    # Check if the object is a function or method
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        name = obj.__name__
        line_number = inspect.getsourcelines(obj)[1]
        file_name = inspect.getfile(obj)
        return FunctionDetails(name=name, line=line_number, file=file_name)

    raise ValueError(f"Unsupported object type {type(obj)} - expected function or method")
