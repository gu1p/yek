"""Shared pynput-backed keyboard state implementation."""

# pylint: disable=missing-function-docstring,import-error

import threading
import time
from typing import Dict, List, Union

import pynput
import pynput.keyboard

from yek.data_structures import ThreadSafeBuffer
from yek.events import KeyEvent, KeyEventKind
from yek.key_utils import get_event
from yek.platforms.base import KeyboardState


class PynputKeyboardState(KeyboardState):
    """Keyboard state backed by pynput listeners."""

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
        event = get_event(key, kind=KeyEventKind.PRESSED)
        with self._lock:
            self._buffer.append(event)
            self._map[event.code] = True

    def _on_release(self, key: Union[pynput.keyboard.Key, pynput.keyboard.KeyCode]):
        event = get_event(key, kind=KeyEventKind.RELEASED)
        with self._lock:
            self._buffer.append(event)
            self._map[event.code] = False

    def is_pressed(self, key) -> bool:
        with self._lock:
            return self._map.get(key.code, False)

    def is_pressed_any(self, *keys) -> bool:
        codes = {k.code for k in keys}
        with self._lock:
            return any(self._map.get(code, False) for code in codes)

    def is_only_pressed(self, *keys) -> bool:
        keys = {k.code for k in keys}
        with self._lock:
            return keys == {k for k, v in self._map.items() if v}

    def was_pressed_any_since(self, since: float, *keys) -> bool:
        keys = {get_event(k, kind=KeyEventKind.PRESSED) for k in keys}
        buffer = set(self._buffer.get_since(since))
        return bool(buffer.intersection(keys))

    def other_than_was_pressed_since(self, since: float, *keys) -> bool:
        keys = {get_event(k, kind=KeyEventKind.PRESSED) for k in keys}
        buffer = set(self._buffer.get_since(since))
        return bool(buffer.difference(keys))

    def get_since(self, since: float) -> List["KeyEvent"]:
        return self._buffer.get_since(since)

    def get_in_the_last(self, seconds: float) -> List["KeyEvent"]:
        return self._buffer.get_since(time.time() - seconds)
