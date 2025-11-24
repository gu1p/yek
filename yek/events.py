import time
import uuid
from typing import Sequence, Optional, Union
from enum import Enum

import pynput


class KeyEventKind(Enum):
    PRESSED = "PRESSED"
    RELEASED = "RELEASED"


class KeyEvent:
    def __init__(self, key: Union[pynput.keyboard.KeyCode, str], kind: KeyEventKind):
        self.id = uuid.uuid4()
        self.key = key
        self.pressed_at = time.time()
        self.kind = kind

    def __eq__(self, other: "KeyEvent"):
        if not isinstance(other, KeyEvent):
            return False

        return (
            self.code == other.code
            and self.kind == other.kind
        )

    @property
    def code(self) -> str:
        return self.key if isinstance(self.key, str) else repr(self.key)

    def get_key_next_event(self, events: Sequence["KeyEvent"]) -> Optional["KeyEvent"]:
        for n, event in enumerate(events):
            if event.pressed_at > self.pressed_at and event.key == self.key:
                return event
        return None

    def __repr__(self):
        return f"KeyEvent({self.key} - {self.kind} - {self.pressed_at})"

    def __hash__(self):
        return hash((self.code, self.kind))

    def __str__(self):
        return str(self.key)
