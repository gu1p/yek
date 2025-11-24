"""Platform-specific keyboard state abstraction."""

# pylint: disable=missing-function-docstring

import abc
from typing import List

from yek.events import KeyEvent
from yek.matchers import Key


class KeyboardState(abc.ABC):
    """Interface implemented by each platform backend."""

    @abc.abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_pressed(self, key: Key) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_pressed_any(self, *keys: Key) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def is_only_pressed(self, *keys: Key) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def was_pressed_any_since(self, since: float, *keys: Key) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def other_than_was_pressed_since(self, since: float, *keys: Key) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_since(self, since: float) -> List["KeyEvent"]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_in_the_last(self, seconds: float) -> List["KeyEvent"]:
        raise NotImplementedError
