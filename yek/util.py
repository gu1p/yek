import inspect
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import List, Union, Dict, Any

import pynput
import pynput.keyboard

__all__ = ["KeyboardStateMap", "KeyEvent"]

from pynput.keyboard import Key

from yek.data_structures import ThreadSafeBuffer
from yek.events import KeyEvent, KeyEventKind
from yek.matchers import Key

_DEBUG = os.environ.get("DEBUG", False)


def _debug(msg):
    if _DEBUG:
        print(msg)


_KeyImpl = Union[str, pynput.keyboard.Key, pynput.keyboard.KeyCode, "KeyEvent", Key]


def get_key_code(k: _KeyImpl) -> pynput.keyboard.KeyCode:
    if isinstance(k, str):
        if len(k) != 1: raise ValueError(f"Only single character strings are supported, got {k}")
        return pynput.keyboard.KeyCode.from_char(k)

    if isinstance(k, pynput.keyboard.Key):
        return k.value

    if isinstance(k, pynput.keyboard.KeyCode):
        return k

    if isinstance(k, Key):
        return k.code

    raise ValueError(f"Unsupported key type {type(k)}")


def get_event(k: _KeyImpl, kind: KeyEventKind) -> KeyEvent:
    return KeyEvent(key=get_key_code(k), kind=kind)


class KeyboardStateMap:
    def __init__(self, max_buffer_len: int = 100):
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._listener = None
        self._map: Dict[str, bool] = {}
        self._buffer = ThreadSafeBuffer[KeyEvent](max_len=max_buffer_len)

    def start(self):
        self._running.set()
        self._listener = pynput.keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()

    def stop(self):
        self._running.clear()
        if self._listener:
            self._listener.stop()

    def _on_press(self, key: Union[pynput.keyboard.Key, pynput.keyboard.KeyCode]):
        _debug(f"Pressed: {key}")
        event = get_event(key, kind=KeyEventKind.PRESSED)
        with self._lock:
            self._buffer.append(event)
            self._map[event.code] = True

    def _on_release(self, key: Union[pynput.keyboard.Key, pynput.keyboard.KeyCode]):
        _debug(f"Released: {key}")
        event = get_event(key, kind=KeyEventKind.RELEASED)
        with self._lock:
            self._buffer.append(event)
            self._map[event.code] = False

    def is_pressed(self, key: Key) -> bool:
        with self._lock:
            return self._map.get(key.code, False)

    def is_pressed_any(self, *keys: Key) -> bool:
        codes = set([k.code for k in keys])
        with self._lock:
            return any([self._map.get(code, False) for code in codes])

    def is_only_pressed(self, *keys: Key) -> bool:
        keys = set([k.code for k in keys])
        with self._lock:
            return keys == set([k for k, v in self._map.items() if v])

    def was_pressed_any_since(self, since: float, *keys: Key) -> bool:
        keys = set([get_event(k) for k in keys])
        buffer = set(self._buffer.get_since(since))
        return bool(buffer.intersection(keys))

    def other_than_was_pressed_since(
            self, since: float, *keys: Key
    ) -> bool:
        keys = set([get_event(k) for k in keys])
        buffer = set(self._buffer.get_since(since))
        return bool(buffer.difference(keys))

    def get_since(self, since: float) -> List["KeyEvent"]:
        return self._buffer.get_since(since)

    def get_in_the_last(self, seconds: float) -> List["KeyEvent"]:
        return self._buffer.get_since(time.time() - seconds)


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
