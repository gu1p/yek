import time
import unittest

try:
    import pynput  # type: ignore
except ImportError:
    pynput = None

if pynput is not None:
    from yek.events import KeyEvent, KeyEventKind
    from yek.keys import Char
    from yek.time import Wait
else:
    KeyEvent = KeyEventKind = Char = Wait = None  # type: ignore


@unittest.skipIf(pynput is None, "pynput not installed")
class MatcherTests(unittest.TestCase):
    def _event(self, char: str, kind: KeyEventKind, ts: float) -> KeyEvent:
        event = KeyEvent(
            key=pynput.keyboard.KeyCode.from_char(char),
            kind=kind,
        )
        event.pressed_at = ts
        return event

    def test_key_match_by_code(self):
        key = Char("a", case=True)
        event = KeyEvent(
            key=pynput.keyboard.KeyCode.from_char("a"),
            kind=KeyEventKind.PRESSED,
        )

        result = key.match([event])

        self.assertTrue(result)
        self.assertEqual(result.value.start, event)

    def test_char_matches_upper_and_lower(self):
        key = Char("a", case=False)
        event_upper = KeyEvent(
            key=pynput.keyboard.KeyCode.from_char("A"),
            kind=KeyEventKind.PRESSED,
        )

        result = key.match([event_upper])

        self.assertTrue(result)

    def test_timed_press_release_with_wait(self):
        matcher = Char("a", case=True) @ Wait(seconds=1)

        press = self._event("a", KeyEventKind.PRESSED, ts=10.0)
        release = self._event("a", KeyEventKind.RELEASED, ts=10.5)

        result = matcher.match([press, release])

        self.assertTrue(result)
        self.assertEqual(result.value.start, press)
        self.assertEqual(result.value.end, release)

    def test_key_events_compare_on_code(self):
        first = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        second = self._event("a", KeyEventKind.PRESSED, ts=2.0)

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))


if __name__ == "__main__":
    unittest.main()
