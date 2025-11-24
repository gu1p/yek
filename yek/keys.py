"""Convenience constructors for common keyboard matcher keys."""

# pylint: disable=invalid-name,import-error

from contextlib import suppress

import pynput
import pynput.keyboard

from yek.matchers import Key, Matcher, _Or, StringMatcher

__all__ = [
    "Char",
    "String",
    "Alt",
    "AltL",
    "AltR",
    "AltGr",
    "Backspace",
    "CapsLock",
    "Cmd",
    "CmdL",
    "CmdR",
    "Ctrl",
    "CtrlL",
    "CtrlR",
    "Delete",
    "Down",
    "End",
    "Enter",
    "Esc",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "Home",
    "Left",
    "PageDown",
    "PageUp",
    "Right",
    "Shift",
    "ShiftL",
    "ShiftR",
    "Space",
    "Tab",
    "Up",
    "MediaPlayPause",
    "MediaVolumeMute",
    "MediaVolumeDown",
    "MediaVolumeUp",
    "MediaPrevious",
]


def Char(char: str, case: bool = False) -> Matcher:
    """Create a matcher for a single character."""
    if not isinstance(char, str) and len(char) != 1:
        raise ValueError(f"We only support single character strings, got {char}")

    if case:
        return Key(name=char, code=repr(pynput.keyboard.KeyCode.from_char(char)))

    lower_case = Key(name=char, code=repr(pynput.keyboard.KeyCode.from_char(char.lower())))
    upper_case = Key(name=char, code=repr(pynput.keyboard.KeyCode.from_char(char.upper())))

    if lower_case != upper_case:
        return _Or(lower_case, upper_case)

    return lower_case


def String(s: str, case: bool = False) -> Matcher:
    """Create a matcher for an exact string sequence."""
    return StringMatcher(s, case=case)


Alt = Key(name="Alt", code=repr(pynput.keyboard.Key.alt.value))
AltL = Key(name="AltL", code=repr(pynput.keyboard.Key.alt_l.value))
AltR = Key(name="AltR", code=repr(pynput.keyboard.Key.alt_r.value))
AltGr = Key(name="AltGr", code=repr(pynput.keyboard.Key.alt_gr.value))
Backspace = Key(name="Backspace", code=repr(pynput.keyboard.Key.backspace.value))
CapsLock = Key(name="CapsLock", code=repr(pynput.keyboard.Key.caps_lock.value))
Cmd = Key(name="Cmd", code=repr(pynput.keyboard.Key.cmd.value))
CmdL = Key(name="CmdL", code=repr(pynput.keyboard.Key.cmd_l.value))
CmdR = Key(name="CmdR", code=repr(pynput.keyboard.Key.cmd_r.value))
Ctrl = Key(name="Ctrl", code=repr(pynput.keyboard.Key.ctrl.value))
CtrlL = Key(name="CtrlL", code=repr(pynput.keyboard.Key.ctrl_l.value))
CtrlR = Key(name="CtrlR", code=repr(pynput.keyboard.Key.ctrl_r.value))
Delete = Key(name="Delete", code=repr(pynput.keyboard.Key.delete.value))
Down = Key(name="Down", code=repr(pynput.keyboard.Key.down.value))
End = Key(name="End", code=repr(pynput.keyboard.Key.end.value))
Enter = Key(name="Enter", code=repr(pynput.keyboard.Key.enter.value))
Esc = Key(name="Esc", code=repr(pynput.keyboard.Key.esc.value))
F1 = Key(name="F1", code=repr(pynput.keyboard.Key.f1.value))
F2 = Key(name="F2", code=repr(pynput.keyboard.Key.f2.value))
F3 = Key(name="F3", code=repr(pynput.keyboard.Key.f3.value))
F4 = Key(name="F4", code=repr(pynput.keyboard.Key.f4.value))
F5 = Key(name="F5", code=repr(pynput.keyboard.Key.f5.value))
F6 = Key(name="F6", code=repr(pynput.keyboard.Key.f6.value))
F7 = Key(name="F7", code=repr(pynput.keyboard.Key.f7.value))
F8 = Key(name="F8", code=repr(pynput.keyboard.Key.f8.value))
F9 = Key(name="F9", code=repr(pynput.keyboard.Key.f9.value))
F10 = Key(name="F10", code=repr(pynput.keyboard.Key.f10.value))
F11 = Key(name="F11", code=repr(pynput.keyboard.Key.f11.value))
F12 = Key(name="F12", code=repr(pynput.keyboard.Key.f12.value))
F13 = Key(name="F13", code=repr(pynput.keyboard.Key.f13.value))
F14 = Key(name="F14", code=repr(pynput.keyboard.Key.f14.value))
F15 = Key(name="F15", code=repr(pynput.keyboard.Key.f15.value))
F16 = Key(name="F16", code=repr(pynput.keyboard.Key.f16.value))
F17 = Key(name="F17", code=repr(pynput.keyboard.Key.f17.value))
F18 = Key(name="F18", code=repr(pynput.keyboard.Key.f18.value))
F19 = Key(name="F19", code=repr(pynput.keyboard.Key.f19.value))
F20 = Key(name="F20", code=repr(pynput.keyboard.Key.f20.value))
Home = Key(name="Home", code=repr(pynput.keyboard.Key.home.value))
Left = Key(name="Left", code=repr(pynput.keyboard.Key.left.value))
PageDown = Key(name="PageDown", code=repr(pynput.keyboard.Key.page_down.value))
PageUp = Key(name="PageUp", code=repr(pynput.keyboard.Key.page_up.value))
Right = Key(name="Right", code=repr(pynput.keyboard.Key.right.value))
Shift = Key(name="Shift", code=repr(pynput.keyboard.Key.shift.value))
ShiftL = Key(name="ShiftL", code=repr(pynput.keyboard.Key.shift_l.value))
ShiftR = Key(name="ShiftR", code=repr(pynput.keyboard.Key.shift_r.value))
Space = Key(name="Space", code=repr(pynput.keyboard.Key.space.value))
Tab = Key(name="Tab", code=repr(pynput.keyboard.Key.tab.value))
Up = Key(name="Up", code=repr(pynput.keyboard.Key.up.value))
MediaPlayPause = Key(
    name="MediaPlayPause", code=repr(pynput.keyboard.Key.media_play_pause.value)
)
MediaVolumeMute = Key(
    name="MediaVolumeMute", code=repr(pynput.keyboard.Key.media_volume_mute.value)
)
MediaVolumeDown = Key(
    name="MediaVolumeDown", code=repr(pynput.keyboard.Key.media_volume_down.value)
)
MediaVolumeUp = Key(
    name="MediaVolumeUp", code=repr(pynput.keyboard.Key.media_volume_up.value)
)
MediaPrevious = Key(
    name="MediaPrevious", code=repr(pynput.keyboard.Key.media_previous.value)
)

with suppress(AttributeError):
    MediaNext = Key(name="MediaNext", code=repr(pynput.keyboard.Key.media_next.value))
    __all__.append("MediaNext")

with suppress(AttributeError):
    Insert = Key(name="Insert", code=repr(pynput.keyboard.Key.insert.value))
    __all__.append("Insert")

with suppress(AttributeError):
    Menu = Key(name="Menu", code=repr(pynput.keyboard.Key.menu.value))
    __all__.append("Menu")

with suppress(AttributeError):
    NumLock = Key(name="NumLock", code=repr(pynput.keyboard.Key.num_lock.value))
    __all__.append("NumLock")

with suppress(AttributeError):
    Pause = Key(name="Pause", code=repr(pynput.keyboard.Key.pause.value))
    __all__.append("Pause")

with suppress(AttributeError):
    PrintScreen = Key(
        name="PrintScreen", code=repr(pynput.keyboard.Key.print_screen.value)
    )
    __all__.append("PrintScreen")

with suppress(AttributeError):
    ScrollLock = Key(name="ScrollLock", code=repr(pynput.keyboard.Key.scroll_lock.value))
    __all__.append("ScrollLock")
