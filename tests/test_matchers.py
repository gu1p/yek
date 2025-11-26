"""Tests for matcher combinations."""

import unittest

from tests.pynput_utils import require_pynput

pynput = require_pynput()

# pylint: disable=wrong-import-position
# pylint: disable=duplicate-code
from yek.events import KeyEvent, KeyEventKind
from yek.keys import Char, Ctrl, Shift, Left
from yek.matchers import Hold, Loop, Matcher, Throttle
from yek.time import Wait


class MatcherTests(unittest.TestCase):
    """Exercises matcher combinations when pynput is available."""

    def _event(self, char: str, kind: KeyEventKind, ts: float) -> KeyEvent:
        """Build a KeyEvent with a fixed timestamp."""
        event = KeyEvent(
            key=pynput.keyboard.KeyCode.from_char(char),
            kind=kind,
        )
        event.pressed_at = ts
        return event

    def test_key_match_by_code(self):
        """Single key matches when codes align."""
        key = Char("a", case=True)
        event = KeyEvent(
            key=pynput.keyboard.KeyCode.from_char("a"),
            kind=KeyEventKind.PRESSED,
        )

        result = key.match([event])

        self.assertTrue(result)
        self.assertEqual(result.value.start, event)

    def test_char_matches_upper_and_lower(self):
        """Lowercase matcher accepts either case when case-insensitive."""
        key = Char("a", case=False)
        event_upper = KeyEvent(
            key=pynput.keyboard.KeyCode.from_char("A"),
            kind=KeyEventKind.PRESSED,
        )

        result = key.match([event_upper])

        self.assertTrue(result)

    def test_timed_press_release_with_wait(self):
        """Two events within window satisfy timed matcher."""
        matcher = Char("a", case=True) @ Wait(seconds=1)

        press = self._event("a", KeyEventKind.PRESSED, ts=10.0)
        release = self._event("a", KeyEventKind.RELEASED, ts=10.5)

        result = matcher.match([press, release])

        self.assertTrue(result)
        self.assertEqual(result.value.start, press)
        self.assertEqual(result.value.end, release)

    def test_key_events_compare_on_code(self):
        """Key events compare by code regardless of timestamp."""
        first = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        second = self._event("a", KeyEventKind.PRESSED, ts=2.0)

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_long_sequence_matches_in_order(self):
        """Sequence operator (/) walks through events cumulatively."""
        matcher = (
            Char("a", case=True)
            / Char("b", case=True)
            / Char("c", case=True)
            / Char("d", case=True)
        )

        events = [
            self._event("a", KeyEventKind.PRESSED, ts=1.0),
            self._event("b", KeyEventKind.PRESSED, ts=1.2),
            self._event("c", KeyEventKind.PRESSED, ts=1.3),
            self._event("d", KeyEventKind.PRESSED, ts=1.4),
        ]

        result = matcher.match(events)

        self.assertTrue(result)
        self.assertEqual(result.value.start, events[0])
        self.assertEqual(result.value.end, events[-1])

    def test_sequence_allows_releases_of_previous_keys(self):
        """Releases of already-matched keys between steps should be ignored."""
        matcher = Char("a", case=True) / Char("b", case=True)
        events = [
            self._event("a", KeyEventKind.PRESSED, ts=1.0),
            self._event("a", KeyEventKind.RELEASED, ts=1.05),
            self._event("b", KeyEventKind.PRESSED, ts=1.1),
        ]

        result = matcher.match(events)

        self.assertTrue(result)
        self.assertEqual(result.value.start, events[0])
        self.assertEqual(result.value.end, events[-1])

    def test_sequence_rejects_skipped_events(self):
        """Sequences require adjacent events; extra keys in between fail."""
        matcher = Char("a", case=True) / Char("b", case=True)
        events = [
            self._event("a", KeyEventKind.PRESSED, ts=1.0),
            self._event("x", KeyEventKind.PRESSED, ts=1.1),
            self._event("b", KeyEventKind.PRESSED, ts=1.2),
        ]
        self.assertFalse(matcher.match(events))

    def test_hold_matches_on_state_and_repeats(self):
        """Hold modifier keeps matching across repeated taps."""
        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            """Minimal keyboard mock."""

            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                """Return True if key code is marked as pressed."""
                return key.code in self._pressed

        keyboard = _KB()
        keyboard._pressed.update({Ctrl.code, Shift.code})  # pylint: disable=protected-access

        class _App:  # pylint: disable=too-few-public-methods
            """Minimal app stub carrying keyboard state."""
            _keyboard = keyboard

        Matcher.app = _App()

        matcher = Hold(Ctrl, Shift) / Left

        tap1 = self._event("x", KeyEventKind.PRESSED, ts=1.0)
        tap1.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]
        tap1.pressed_at = 1.0

        tap2 = self._event("y", KeyEventKind.PRESSED, ts=2.0)
        tap2.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]
        tap2.pressed_at = 2.0

        self.assertTrue(matcher.match([tap1]))
        self.assertTrue(matcher.match([tap2]))

        keyboard._pressed.clear()  # pylint: disable=protected-access
        self.assertFalse(matcher.match([tap2]))

        Matcher.app = prev_app

    def test_hold_only_requires_no_extra_keys(self):
        """Hold(..., only=True) fails if extra keys are down."""
        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            """Minimal keyboard mock."""

            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                """Return True if key code is pressed."""
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                """Return True if only provided keys are pressed."""
                return {k.code for k in keys} == set(self._pressed)

        keyboard = _KB()
        keyboard._pressed.update({Ctrl.code, Shift.code})  # pylint: disable=protected-access

        class _App:  # pylint: disable=too-few-public-methods
            """Minimal app stub carrying keyboard state."""
            _keyboard = keyboard

        Matcher.app = _App()

        matcher = Hold(Ctrl, Shift, only=True) / Left

        tap = self._event("x", KeyEventKind.PRESSED, ts=1.0)
        tap.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]
        tap.pressed_at = 1.0

        self.assertTrue(matcher.match([tap]))

        keyboard._pressed.add(Char("z", case=True).code)  # pylint: disable=protected-access
        self.assertFalse(matcher.match([tap]))

        Matcher.app = prev_app

    def test_sequence_then_hold_arms_once(self):
        """Prefix match followed by hold tail repeats without prefix event again."""
        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            """Minimal keyboard mock."""

            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                """Return True if key code is pressed."""
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                """Return True if only provided keys are pressed."""
                return {k.code for k in keys} == set(self._pressed)

        keyboard = _KB()

        class _App:  # pylint: disable=too-few-public-methods
            _keyboard = keyboard

        Matcher.app = _App()

        matcher = (Char("a", case=True) / Hold(Left, only=True))

        # First match: 'a' event plus Left held (and only Left held)
        keyboard._pressed.update({Left.code})  # pylint: disable=protected-access
        tap_a = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        tap_a.key = pynput.keyboard.KeyCode.from_char("a")
        tap_left = self._event("x", KeyEventKind.PRESSED, ts=1.0)
        tap_left.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]

        self.assertTrue(matcher.match([tap_a, tap_left]))

        # Subsequent Left tap without another 'a' should still match while Left is held
        tap_left2 = self._event("y", KeyEventKind.PRESSED, ts=1.5)
        tap_left2.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]
        self.assertTrue(matcher.match([tap_left2]))

        # Clear held keys; matcher should fail until re-armed
        keyboard._pressed.clear()  # pylint: disable=protected-access
        self.assertFalse(matcher.match([tap_left2]))

        Matcher.app = prev_app

    def test_throttle_limits_frequency(self):
        """Throttle wrapper limits how often a held combo matches."""
        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            """Minimal keyboard mock for throttling tests."""

            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                """Return True if key code is pressed."""
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                """Return True if only provided keys are pressed."""
                return {k.code for k in keys} == set(self._pressed)

        keyboard = _KB()
        keyboard._pressed.update({Ctrl.code, Shift.code})  # pylint: disable=protected-access

        class _App:  # pylint: disable=too-few-public-methods
            _keyboard = keyboard

        Matcher.app = _App()

        throttled = Throttle(Hold(Ctrl, Shift), every_ms=100)

        # Patch time to control elapsed intervals
        class FakeTime:  # pylint: disable=too-few-public-methods
            """Deterministic time stub."""

            def __init__(self, values):
                self.values = list(values)

            def time(self):
                """Pop the next timestamp."""
                return self.values.pop(0)

        base_time = 1.0
        fake = FakeTime(
            [
                base_time,          # Hold call 1
                base_time,          # Throttle call 1
                base_time + 0.05,   # Hold call 2
                base_time + 0.05,   # Throttle call 2 (blocked)
                base_time + 0.11,   # Hold call 3
                base_time + 0.11,   # Throttle call 3 (allowed)
            ]
        )
        original_throttle_time = Throttle.__dict__["match"].__globals__["time"]
        original_hold_time = Hold.__dict__["match"].__globals__["time"]

        try:
            Throttle.__dict__["match"].__globals__["time"] = fake
            Hold.__dict__["match"].__globals__["time"] = fake
            tap = self._event("x", KeyEventKind.PRESSED, ts=0)
            tap.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]
            tap.pressed_at = 0

            self.assertTrue(throttled.match([tap]))
            self.assertFalse(throttled.match([tap]))
            self.assertTrue(throttled.match([tap]))
        finally:
            Throttle.__dict__["match"].__globals__["time"] = original_throttle_time
            Hold.__dict__["match"].__globals__["time"] = original_hold_time
            Matcher.app = prev_app

    def test_hold_throttle_helper(self):
        """Hold.throttle helper returns a throttled matcher that respects interval."""
        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            """Minimal keyboard mock for helper tests."""

            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                """Return True if key code is pressed."""
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                """Return True if only provided keys are pressed."""
                return {k.code for k in keys} == set(self._pressed)

        keyboard = _KB()
        keyboard._pressed.update({Ctrl.code, Shift.code})  # pylint: disable=protected-access

        class _App:  # pylint: disable=too-few-public-methods
            _keyboard = keyboard

        Matcher.app = _App()

        throttled = Hold(Ctrl, Shift).throttle(every_ms=50)

        class FakeTime:  # pylint: disable=too-few-public-methods
            """Deterministic time stub."""

            def __init__(self, values):
                self.values = list(values)

            def time(self):
                """Pop the next timestamp."""
                return self.values.pop(0)

        fake = FakeTime(
            [
                1.0,  # Hold call 1
                1.0,  # Throttle call 1
                1.02,  # Hold call 2 (too soon)
                1.02,  # Throttle call 2 blocked
                1.06,  # Hold call 3 (enough time passed)
                1.06,  # Throttle call 3 allowed
            ]
        )

        original_throttle_time = Throttle.__dict__["match"].__globals__["time"]
        original_hold_time = Hold.__dict__["match"].__globals__["time"]

        try:
            Throttle.__dict__["match"].__globals__["time"] = fake
            Hold.__dict__["match"].__globals__["time"] = fake

            tap = self._event("x", KeyEventKind.PRESSED, ts=0)
            tap.key = pynput.keyboard.Key.left.value  # type: ignore[attr-defined]
            tap.pressed_at = 0

            self.assertTrue(throttled.match([tap]))
            self.assertFalse(throttled.match([tap]))
            self.assertTrue(throttled.match([tap]))
        finally:
            Throttle.__dict__["match"].__globals__["time"] = original_throttle_time
            Hold.__dict__["match"].__globals__["time"] = original_hold_time
            Matcher.app = prev_app

    def test_loop_sequence_renews_and_expires(self):
        """Loop arms after prefix, renews deadline, and expires when idle."""
        matcher = Char("a", case=True) / Loop(Char("b", case=True), tolerance_ms=1000, renew=True)

        tap_a = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        tap_b1 = self._event("b", KeyEventKind.PRESSED, ts=1.1)
        self.assertTrue(matcher.match([tap_a, tap_b1]))

        tap_b2 = self._event("b", KeyEventKind.PRESSED, ts=2.0)  # within renewed window
        self.assertTrue(matcher.match([tap_b2]))

        tap_b3 = self._event("b", KeyEventKind.PRESSED, ts=4.2)  # after expiry
        self.assertFalse(matcher.match([tap_b3]))

        tap_a2 = self._event("a", KeyEventKind.PRESSED, ts=4.3)
        tap_b4 = self._event("b", KeyEventKind.PRESSED, ts=4.35)
        self.assertTrue(matcher.match([tap_a2, tap_b4]))

    def test_loop_without_renew_expires_at_deadline(self):
        """Loop without renew stops matching after fixed deadline."""
        matcher = Char("a", case=True) / Loop(Char("b", case=True), tolerance_ms=500, renew=False)

        tap_a = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        tap_b1 = self._event("b", KeyEventKind.PRESSED, ts=1.05)
        self.assertTrue(matcher.match([tap_a, tap_b1]))

        tap_b2 = self._event("b", KeyEventKind.PRESSED, ts=1.7)  # after deadline
        self.assertFalse(matcher.match([tap_b2]))

        tap_a2 = self._event("a", KeyEventKind.PRESSED, ts=1.8)
        tap_b3 = self._event("b", KeyEventKind.PRESSED, ts=1.85)
        self.assertTrue(matcher.match([tap_a2, tap_b3]))

    def test_loop_must_be_last_in_sequence(self):
        """Loop in the middle of a sequence should raise."""
        with self.assertRaises(ValueError):
            _ = Char("a", case=True) / Loop(Char("b", case=True)) / Char("c", case=True)

    def test_exclusive_chord_rejects_extra_keys(self):
        """Exclusive chord should fail when another key is pressed in the overlap window."""
        chord = Char("a", case=True) & Char("b", case=True)
        a_press = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        b_press = self._event("b", KeyEventKind.PRESSED, ts=1.01)
        c_press = self._event("c", KeyEventKind.PRESSED, ts=1.02)
        self.assertFalse(chord.match([a_press, b_press, c_press]))

    def test_nonexclusive_chord_allows_extra_keys(self):
        """Non-exclusive chord should match even if extra keys are pressed."""
        chord = (Char("a", case=True) + Char("b", case=True))
        a_press = self._event("a", KeyEventKind.PRESSED, ts=1.0)
        b_press = self._event("b", KeyEventKind.PRESSED, ts=1.01)
        c_press = self._event("c", KeyEventKind.PRESSED, ts=1.02)
        result = chord.match([a_press, b_press, c_press])
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
