from typing import Union

import pynput
import pynput.keyboard

from yek.events import KeyEvent, KeyEventKind
from yek.matchers import Key as MatcherKey

_KeyImpl = Union[str, pynput.keyboard.Key, pynput.keyboard.KeyCode, "KeyEvent", MatcherKey]


def get_key_code(k: _KeyImpl) -> Union[pynput.keyboard.KeyCode, str]:
    if isinstance(k, str):
        if len(k) != 1:
            raise ValueError(f"Only single character strings are supported, got {k}")
        return pynput.keyboard.KeyCode.from_char(k)

    if isinstance(k, pynput.keyboard.Key):
        return k.value

    if isinstance(k, pynput.keyboard.KeyCode):
        return k

    if isinstance(k, MatcherKey):
        return k.code

    raise ValueError(f"Unsupported key type {type(k)}")


def get_event(k: _KeyImpl, kind: KeyEventKind) -> KeyEvent:
    return KeyEvent(key=get_key_code(k), kind=kind)
