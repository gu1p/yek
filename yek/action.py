"""Actions and execution context for shortcuts."""

# pylint: disable=missing-class-docstring,missing-function-docstring

import abc
import inspect
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from pynput.keyboard import Key, KeyCode  # pylint: disable=import-error

from yek.util import KeyboardStateMap, KeyEvent, get_function_details


class Action(abc.ABC):
    @abc.abstractmethod
    def execute(self, context: "Context"):
        pass

    @abc.abstractmethod
    def is_running(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def id(self) -> str:
        pass

    def __eq__(self, other: "Action"):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self) -> str:
        return self.id

    @property
    @abc.abstractmethod
    def properties(self) -> Optional[Any]:
        pass


class SimpleAction(Action):
    def __init__(self, func: Callable[..., None]):
        self._id = str(uuid.uuid4())
        desc = get_function_details(func)
        self._props = {
            "file": desc.file,
            "name": desc.name,
            "line": desc.line
        }
        self._func = func
        self._execution = None
        self._lock = threading.Lock()

    def execute(self, context: "Context"):
        with self._lock:
            if self.is_running():
                return
            self._execution = threading.Thread(target=self._func, args=(context,))
            self._execution.start()

    def is_running(self) -> bool:
        if self._execution is None:
            return False
        return self._execution.is_alive()

    @property
    def id(self) -> str:
        return self._id

    @property
    def properties(self) -> Optional[Any]:
        return self._props

    def __str__(self) -> str:
        name, file, line = self._props["name"], self._props["file"], self._props["line"]
        return f"SimpleAction(func={name}, defined_at={file}:{line})"


class Context:
    def __init__(self, keyboard_state: KeyboardStateMap, match_result=None):
        self.__keyboard_state = keyboard_state
        self.__started_at = time.time()
        self.__payload = {}
        self.__match_result = match_result

    def is_pressed(self, key: Key):
        return self.__keyboard_state.is_pressed(key)

    def is_only_pressed(self, *keys: Union[KeyCode, KeyEvent, Key]):
        return self.__keyboard_state.is_only_pressed(*keys)

    def was_pressed_other_than(self, *keys: Union[KeyCode, KeyEvent, Key]):
        return self.__keyboard_state.other_than_was_pressed_since(
            self.__started_at, *keys
        )

    def keys(self) -> List[KeyEvent]:
        return self.__keyboard_state.get_since(self.__started_at)

    @property
    def started_at(self):
        return self.__started_at

    @property
    def payload(self) -> Dict[str, Any]:
        return self.__payload

    @property
    def match(self):
        return self.__match_result

    @property
    def match_end(self):
        if self.__match_result and self.__match_result.value:
            return self.__match_result.value.end
        return None


def get_function_name(obj: Callable[..., Any]) -> Optional[str]:
    # Check if the object is a function or method
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return obj.__name__
    return None
