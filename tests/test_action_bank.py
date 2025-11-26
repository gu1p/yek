"""Tests for action bank evaluating multiple shortcuts."""

import os
import string
import tempfile
import textwrap
import unittest
from collections import deque
from time import time

from tests.pynput_utils import require_pynput

pynput = require_pynput()

# pylint: disable=wrong-import-position
# pylint: disable=line-too-long,protected-access,too-many-public-methods,too-many-locals
# pylint: disable=too-many-branches,too-many-statements,missing-function-docstring
# pylint: disable=multiple-statements,duplicate-code
from yek.action import SimpleAction
from yek.events import KeyEventKind
from yek.key_utils import get_event
from yek.keys import (
    Alt,
    AltL,
    AltR,
    Cmd,
    Ctrl,
    CtrlL,
    CtrlR,
    Down,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    F13,
    F14,
    F15,
    F16,
    F17,
    F18,
    F19,
    F20,
    Left,
    Home,
    PageUp,
    PageDown,
    End,
    Right,
    Shift,
    ShiftL,
    ShiftR,
    Space,
    Enter,
    Esc,
    Backspace,
    Delete,
    Tab,
    Up,
    Char,
    Chord,
    String,
)
from yek.matchers import Hold, Loop, Matcher, Key
from yek.shortcuts import ActionBankTrigger, ActionTrigger, check_routes
from yek.action import Context
from yek.time import Wait


class ActionBankTests(unittest.TestCase):
    """Action bank should evaluate each registered trigger."""

    def _timestamped_seq(self, seq, offset: float):
        for idx, event in enumerate(seq):
            event.pressed_at = offset + idx * 0.01
        return seq

    def test_multiple_routes_match_independently(self):
        """Each registered trigger should be evaluated, not just the first one."""

        def one(_): ...
        def two(_): ...
        def three(_): ...

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd / Shift / Char("s"), SimpleAction(one)))
        bank.register(ActionTrigger(Cmd / Shift / Char("o"), SimpleAction(two)))
        bank.register(ActionTrigger(Cmd / Shift / Char("t"), SimpleAction(three)))

        seqs = [
            self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED),
                 get_event(Shift, KeyEventKind.PRESSED),
                 get_event("s", KeyEventKind.PRESSED)],
                time(),
            ),
            self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED),
                 get_event(Shift, KeyEventKind.PRESSED),
                 get_event("o", KeyEventKind.PRESSED)],
                time() + 1,
            ),
            self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED),
                 get_event(Shift, KeyEventKind.PRESSED),
                 get_event("t", KeyEventKind.PRESSED)],
                time() + 2,
            ),
        ]

        names = []
        for seq in seqs:
            actions = bank.match(seq)
            names.extend([a.properties["name"] for a, _ in actions])

        self.assertEqual(names, ["one", "two", "three"])

    def test_triggers_sorted_by_specificity(self):
        """More specific (longer) matchers run before shorter ones."""

        def short(_): ...
        def long(_): ...

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Char("x", case=True) / Char("y", case=True), SimpleAction(short)))
        bank.register(ActionTrigger(Cmd / Shift / Char("z", case=True), SimpleAction(long)))

        events = self._timestamped_seq(
            [
                get_event(Cmd, KeyEventKind.PRESSED),
                get_event(Shift, KeyEventKind.PRESSED),
                get_event("z", KeyEventKind.PRESSED),
                get_event("x", KeyEventKind.PRESSED),
                get_event("y", KeyEventKind.PRESSED),
            ],
            time(),
        )

        actions = bank.match(events)
        names = [a.properties["name"] for a, _ in actions]

        # Remaining events should be eligible in a subsequent pass once the buffer advances.
        if actions:
            _, res = actions[0]
            remaining = events[res.stop_position + 1 :]
            actions.extend(bank.match(remaining))
            names = [a.properties["name"] for a, _ in actions]

        self.assertEqual(names, ["long", "short"])

    def test_overlapping_routes_raise_collision(self):
        """Conflicting shortcuts are rejected at registration time."""

        def hold_left(_): ...
        def cmd_shift_left(_): ...

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Shift / Hold(Left), SimpleAction(hold_left)))

        with self.assertRaises(ValueError):
            bank.register(ActionTrigger(Cmd / Shift / Left, SimpleAction(cmd_shift_left)))

    def test_timed_windows_overlap_collision(self):
        """Timed press-release on same key with overlapping windows should collide."""

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd @ (0.2, 2), SimpleAction(lambda _: None)))

        with self.assertRaises(ValueError):
            bank.register(ActionTrigger(Cmd @ (0.3, 4), SimpleAction(lambda _: None)))

    def test_timed_windows_disjoint_allowed(self):
        """Disjoint timed windows on same key should be allowed."""

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd @ (0.0, 0.1), SimpleAction(lambda _: None)))

        # Non-overlapping window
        bank.register(ActionTrigger(Cmd @ (0.2, 0.3), SimpleAction(lambda _: None)))

    def test_timed_vs_untimed_collision(self):
        """Timed press-release and untimed key on same key collide."""

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd @ (0.1, 0.2), SimpleAction(lambda _: None)))

        with self.assertRaises(ValueError):
            bank.register(ActionTrigger(Cmd, SimpleAction(lambda _: None)))

    def test_exclusive_chord_and_sequence_do_not_collide(self):
        """Exclusive chord should not collide with a sequence that requires an extra key."""

        def chord_fn(_): ...
        def seq_fn(_): ...

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd & Shift & Space, SimpleAction(chord_fn)))
        bank.register(ActionTrigger(Cmd / Shift / Right, SimpleAction(seq_fn)))

    def test_non_exclusive_chord_collides_with_sequence(self):
        """Non-exclusive chord should collide with sequence sharing its keys."""

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Chord(Cmd, Shift, Space, only=False), SimpleAction(lambda _: None)))

        with self.assertRaises(ValueError):
            bank.register(ActionTrigger(Cmd / Shift / Right, SimpleAction(lambda _: None)))

    def test_exclusive_hold_sequences_do_not_collide(self):
        """Distinct hold-only sequences should coexist when modifiers diverge."""

        def left_fn(_): ...
        def right_fn(_): ...

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(CtrlL / Hold(Left, only=True).throttle(50), SimpleAction(left_fn)))
        bank.register(ActionTrigger(CtrlL / Hold(Right, only=True).throttle(50), SimpleAction(right_fn)))

    def test_superset_route_wins_over_subset(self):
        """Superset match should be chosen over subset in the same buffer."""

        bank = ActionBankTrigger()

        superset_hit = []
        subset_hit = []

        def sup_action(_): superset_hit.append("super")
        def sub_action(_): subset_hit.append("sub")

        bank.register(ActionTrigger(CtrlL / Shift / Space / Right, SimpleAction(sup_action)))
        bank.register(ActionTrigger(CtrlL / Shift / Right, SimpleAction(sub_action)))

        base = time()
        events = self._timestamped_seq(
            [
                get_event(CtrlL, KeyEventKind.PRESSED),
                get_event(Shift, KeyEventKind.PRESSED),
                get_event(Space, KeyEventKind.PRESSED),
                get_event(Right, KeyEventKind.PRESSED),
            ],
            base,
        )

        actions = bank.match(events)
        self.assertEqual([a.properties["name"] for a, _ in actions], ["sup_action"])
        self.assertEqual(superset_hit, [])
        self.assertEqual(subset_hit, [])

        # Execute selected action to ensure it was the superset.
        action, result = actions[0]
        action.execute(Context(None, match_result=result))  # type: ignore[arg-type]
        action._execution.join(timeout=1)  # type: ignore[attr-defined]
        self.assertEqual(superset_hit, ["super"])
        self.assertEqual(subset_hit, [])

    def test_consumed_events_not_reused_for_second_route(self):
        """Once a route is chosen, remaining events from that slice should be processed later."""

        bank = ActionBankTrigger()

        first = []
        second = []

        def first_action(_): first.append("first")
        def second_action(_): second.append("second")

        bank.register(ActionTrigger(CtrlL / Shift / Right, SimpleAction(first_action)))
        bank.register(ActionTrigger(Shift / Right, SimpleAction(second_action)))

        base = time()
        events = self._timestamped_seq(
            [
                get_event(CtrlL, KeyEventKind.PRESSED),
                get_event(Shift, KeyEventKind.PRESSED),
                get_event(Right, KeyEventKind.PRESSED),
                get_event(Shift, KeyEventKind.PRESSED),
                get_event(Right, KeyEventKind.PRESSED),
            ],
            base,
        )

        actions = bank.match(events)
        self.assertEqual([a.properties["name"] for a, _ in actions], ["first_action"])

        # Remaining events should still allow the second route in a subsequent pass.
        _, res = actions[0]
        remaining = events[res.stop_position + 1 :]
        follow_up = bank.match(remaining)
        self.assertEqual([a.properties["name"] for a, _ in follow_up], ["second_action"])

        for action, result in actions + follow_up:
            action.execute(Context(None, match_result=result))  # type: ignore[arg-type]
            action._execution.join(timeout=1)  # type: ignore[attr-defined]

        self.assertEqual(first, ["first"])
        self.assertEqual(second, ["second"])

    def test_loop_selection_blocks_other_routes(self):
        """Looped route should win over a single key while it remains active."""

        bank = ActionBankTrigger()

        loop_hits = []
        right_hits = []

        loop_matcher = CtrlL / Loop(Right, tolerance_ms=500, renew=True)
        def loop_action(_): loop_hits.append("loop")
        def right_action(_): right_hits.append("right")

        bank.register(ActionTrigger(loop_matcher, SimpleAction(loop_action)))
        bank.register(ActionTrigger(Right, SimpleAction(right_action)))

        base = time()
        arming_events = self._timestamped_seq(
            [get_event(CtrlL, KeyEventKind.PRESSED), get_event(Right, KeyEventKind.PRESSED)],
            base,
        )

        actions = bank.match(arming_events)
        self.assertEqual([a.properties["name"] for a, _ in actions], ["loop_action"])

        # Keep loop active and ensure subsequent Right is still claimed by the loop.
        follow_up_events = self._timestamped_seq(
            [get_event(Right, KeyEventKind.PRESSED)],
            base + 0.2,
        )
        follow_up = bank.match(follow_up_events)
        self.assertEqual([a.properties["name"] for a, _ in follow_up], ["loop_action"])

        for action, result in actions + follow_up:
            action.execute(Context(None, match_result=result))  # type: ignore[arg-type]
            action._execution.join(timeout=1)  # type: ignore[attr-defined]

        self.assertEqual(loop_hits, ["loop", "loop"])
        self.assertEqual(right_hits, [])

    def test_loop_drops_on_unrelated_key(self):
        """Loop should reset when a non-matching key is pressed."""

        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                return {k.code for k in keys} == self._pressed

        keyboard = _KB()
        keyboard._pressed.update({CtrlL.code, Alt.code, Right.code})  # pylint: disable=protected-access

        class _App:  # pylint: disable=too-few-public-methods
            _keyboard = keyboard

        Matcher.app = _App()

        try:
            loop_hits = []
            other_hits = []

            bank = ActionBankTrigger()
            loop_matcher = CtrlL / Alt / Loop(
                Hold(Right).throttle(80) | Hold(Left).throttle(80),
                tolerance_ms=500,
                renew=True,
            )
            bank.register(ActionTrigger(loop_matcher, SimpleAction(lambda _: loop_hits.append("loop"))))
            bank.register(ActionTrigger(Cmd, SimpleAction(lambda _: other_hits.append("cmd"))))

            base = time()
            arming_events = self._timestamped_seq(
                [get_event(CtrlL, KeyEventKind.PRESSED), get_event(Alt, KeyEventKind.PRESSED), get_event(Right, KeyEventKind.PRESSED)],
                base,
            )
            bank.match(arming_events)  # arms the loop

            # Now press Cmd; loop should reset and allow Cmd route.
            keyboard._pressed.add(Cmd.code)  # pylint: disable=protected-access
            cmd_events = self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED)],
                base + 0.2,
            )
            actions = bank.match(cmd_events)

            self.assertEqual([a.properties["name"] for a, _ in actions], ["<lambda>"])
            for action, result in actions:
                action.execute(Context(None, match_result=result))  # type: ignore[arg-type]
                action._execution.join(timeout=1)  # type: ignore[attr-defined]

            self.assertEqual(loop_hits, [])
            self.assertEqual(other_hits, ["cmd"])
        finally:
            Matcher.app = prev_app

    def test_match_end_propagated_in_context(self):
        """Context.match_end should expose the last event of a match."""
        seen = []

        def handler(ctx):
            seen.append(ctx.match_end.code)

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd / Loop(Right | Left, tolerance_ms=500), SimpleAction(handler)))

        base = time()
        events = self._timestamped_seq(
            [get_event(Cmd, KeyEventKind.PRESSED), get_event(Right, KeyEventKind.PRESSED)],
            base,
        )
        actions = bank.match(events)
        self.assertEqual(len(actions), 1)
        action, result = actions[0]
        action.execute(Context(None, match_result=result))  # type: ignore[arg-type]
        action._execution.join(timeout=1)  # type: ignore[attr-defined]
        self.assertEqual(seen, [Right.code])

    def test_loop_requires_prefix_guardrail(self):
        """Loop without a prefix should be rejected."""
        bank = ActionBankTrigger()
        with self.assertRaises(ValueError):
            bank.register(ActionTrigger(Loop(Char("x", case=True)), SimpleAction(lambda _: None)))

    def test_loop_collides_with_plain_route(self):
        """Loop route should collide with an equivalent non-loop route."""
        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Cmd / Cmd / Loop(Left, tolerance_ms=500), SimpleAction(lambda _: None)))

        with self.assertRaises(ValueError):
            bank.register(ActionTrigger(Cmd / Cmd / Left, SimpleAction(lambda _: None)))

    def test_loop_dispatches_actions_while_active(self):
        """Loop should fire actions within tolerance and stop after expiry until re-armed."""
        hits = []

        def handler(_):
            hits.append("hit")

        bank = ActionBankTrigger()
        matcher = Char("a", case=True) / Loop(Char("b", case=True), tolerance_ms=300, renew=True)
        bank.register(ActionTrigger(matcher, SimpleAction(handler)))

        base = time() + 10

        seq1 = self._timestamped_seq(
            [get_event("a", KeyEventKind.PRESSED), get_event("b", KeyEventKind.PRESSED)],
            base,
        )
        actions = bank.match(seq1)
        self.assertEqual(len(actions), 1)
        action, _ = actions[0]
        action.execute(None)
        action._execution.join(timeout=1)  # type: ignore[attr-defined]
        self.assertEqual(len(hits), 1)

        seq2 = self._timestamped_seq(
            [get_event("b", KeyEventKind.PRESSED)],
            base + 0.1,
        )
        actions = bank.match(seq2)
        self.assertEqual(len(actions), 1)
        action, _ = actions[0]
        action.execute(None)
        action._execution.join(timeout=1)  # type: ignore[attr-defined]
        self.assertEqual(len(hits), 2)

        seq3 = self._timestamped_seq(
            [get_event("b", KeyEventKind.PRESSED)],
            base + 0.7,  # beyond tolerance
        )
        actions = bank.match(seq3)
        self.assertEqual(len(actions), 0)
        self.assertEqual(len(hits), 2)

        seq4 = self._timestamped_seq(
            [get_event("a", KeyEventKind.PRESSED), get_event("b", KeyEventKind.PRESSED)],
            base + 1.0,
        )
        actions = bank.match(seq4)
        self.assertEqual(len(actions), 1)
        action, _ = actions[0]
        action.execute(None)
        action._execution.join(timeout=1)  # type: ignore[attr-defined]
        self.assertEqual(len(hits), 3)

    def test_priority_override_wins(self):
        """Explicit priority overrides specificity tie-breaking."""

        def low(_): ...
        def high(_): ...

        bank = ActionBankTrigger()
        bank.register(ActionTrigger(Char("a", case=True), SimpleAction(low), priority=1))
        bank.register(ActionTrigger(Cmd / Char("b", case=True), SimpleAction(high), priority=10))

        events = self._timestamped_seq(
            [
                get_event(Cmd, KeyEventKind.PRESSED),
                get_event("b", KeyEventKind.PRESSED),
                get_event("a", KeyEventKind.PRESSED),
            ],
            time(),
        )

        actions = bank.match(events)
        names = [a.properties["name"] for a, _ in actions]

        if actions:
            _, res = actions[0]
            remaining = events[res.stop_position + 1 :]
            actions.extend(bank.match(remaining))
            names = [a.properties["name"] for a, _ in actions]

        self.assertEqual(names, ["high", "low"])

    def test_check_routes_detects_collision(self):
        """check_routes helper raises for overlapping routes."""

        content = textwrap.dedent(
            """
            from yek.keys import Char
            from yek.shortcuts import App

            app = App()

            @app.on(Char("a", case=True))
            def first(_): ...

            @app.on(Char("a", case=True))
            def second(_): ...
            """
        )

        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as handle:
            handle.write(content)
            path = handle.name

        try:
            with self.assertRaises(ValueError):
                check_routes(path)
        finally:
            os.unlink(path)

    def test_routes_fire_across_many_matchers(self):
        """Integration-style check that varied matchers dispatch correctly."""

        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                return {k.code for k in keys} == set(self._pressed)

        keyboard = _KB()

        class _App:  # pylint: disable=too-few-public-methods
            _keyboard = keyboard

        Matcher.app = _App()

        try:
            def open_fn(_): ...
            def string_fn(_): ...
            def tap_fn(_): ...
            def ctrl_left_fn(_): ...
            def save_fn(_): ...

            bank = ActionBankTrigger()
            bank.register(ActionTrigger(Cmd / Shift / Char("o"), SimpleAction(open_fn)))
            bank.register(ActionTrigger(String("hi"), SimpleAction(string_fn)))
            bank.register(ActionTrigger(Right @ (0.05, 0.25), SimpleAction(tap_fn)))
            bank.register(ActionTrigger(Hold(Ctrl, only=True) / Left, SimpleAction(ctrl_left_fn)))
            bank.register(ActionTrigger((Cmd / Shift / Char("s")) | (Cmd / Shift / Char("a")), SimpleAction(save_fn)))

            base = time() + 100

            open_events = self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED),
                 get_event(Shift, KeyEventKind.PRESSED),
                 get_event("o", KeyEventKind.PRESSED)],
                base,
            )
            self.assertEqual(
                [a.properties["name"] for a, _ in bank.match(open_events)],
                ["open_fn"],
            )

            string_events = self._timestamped_seq(
                [get_event("h", KeyEventKind.PRESSED),
                 get_event("i", KeyEventKind.PRESSED)],
                base + 100,
            )
            self.assertEqual(
                [a.properties["name"] for a, _ in bank.match(string_events)],
                ["string_fn"],
            )

            tap_events = self._timestamped_seq(
                [get_event(Right, KeyEventKind.PRESSED),
                 get_event(Right, KeyEventKind.RELEASED)],
                base + 200,
            )
            # Ensure release happens within 0.05-0.25s window
            tap_events[1].pressed_at = tap_events[0].pressed_at + 0.1
            self.assertEqual(
                [a.properties["name"] for a, _ in bank.match(tap_events)],
                ["tap_fn"],
            )

            keyboard._pressed = {Ctrl.code}  # pylint: disable=protected-access
            ctrl_left_events = self._timestamped_seq(
                [get_event(Left, KeyEventKind.PRESSED)],
                base + 300,
            )
            self.assertEqual(
                [a.properties["name"] for a, _ in bank.match(ctrl_left_events)],
                ["ctrl_left_fn"],
            )
            keyboard._pressed.clear()  # pylint: disable=protected-access

            save_branch_one = self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED),
                 get_event(Shift, KeyEventKind.PRESSED),
                 get_event("s", KeyEventKind.PRESSED)],
                base + 400,
            )
            self.assertEqual(
                [a.properties["name"] for a, _ in bank.match(save_branch_one)],
                ["save_fn"],
            )

            save_branch_two = self._timestamped_seq(
                [get_event(Cmd, KeyEventKind.PRESSED),
                 get_event(Shift, KeyEventKind.PRESSED),
                 get_event("a", KeyEventKind.PRESSED)],
                base + 500,
            )
            self.assertEqual(
                [a.properties["name"] for a, _ in bank.match(save_branch_two)],
                ["save_fn"],
            )
        finally:
            Matcher.app = prev_app

    def test_bulk_routes_matrix(self):
        """Generate a large set of varied routes and ensure each triggers correctly."""

        prev_app = Matcher.app

        class _KB:  # pylint: disable=too-few-public-methods
            def __init__(self):
                self._pressed = set()

            def is_pressed(self, key):
                return key.code in self._pressed

            def is_only_pressed(self, *keys):
                return {k.code for k in keys} == set(self._pressed)

        keyboard = _KB()

        class _App:  # pylint: disable=too-few-public-methods
            _keyboard = keyboard

        Matcher.app = _App()

        try:
            bank = ActionBankTrigger()
            cases = []
            used_chars = set()

            # Pools to avoid collisions: key-based matchers use key_pool; string matchers use string_pool
            key_chars = string.ascii_letters + string.digits
            key_pool = deque([Char(c, case=True) for c in key_chars])
            key_pool.extend([
                Alt, AltL, AltR, Ctrl, CtrlL, CtrlR, Shift, ShiftL, ShiftR,
                Tab, Left, Right, Up, Down, Home, PageUp, PageDown, End,
                Space, Enter, Esc, Backspace, Delete,
                F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
                F13, F14, F15, F16, F17, F18, F19, F20,
            ])
            safe_punct = [c for c in string.punctuation if c not in {"'", '"', "\\", "`"}]
            string_pool = deque(list(string.ascii_letters + string.digits) + safe_punct)

            def next_key():
                if not key_pool:
                    return None
                return key_pool.popleft()

            def next_string_char():
                if not string_pool:
                    return None
                return string_pool.popleft()

            def add_case(matcher, events, pressed=None):
                idx = len(cases)

                def _fn(_): ...

                _fn.__name__ = f"handler_{idx}"
                bank.register(ActionTrigger(matcher, SimpleAction(_fn)))
                cases.append({"name": _fn.__name__, "events": events, "pressed": pressed or set()})
                # Track single-character keys used so far to avoid string collisions later
                if isinstance(matcher, (Key,)):
                    char = _code_to_char(getattr(matcher, "_code", ""))
                    if char:
                        used_chars.add(char)

            base = time()
            start_ts = base

            def ts():
                nonlocal start_ts
                start_ts += 1
                return start_ts

            def _code_to_char(code: str) -> str:
                if "'" in code:
                    try:
                        return code.split("'")[1]
                    except IndexError:
                        return ""
                return ""

            # Allocate target counts
            target_counts = {
                "simple": 20,
                "cmd": 15,
                "timed": 15,
                "hold": 6,
                "throttle": 2,
                "alt": 0,
                "chord": 0,
            }

            for _ in range(target_counts["simple"]):
                k = next_key()
                if k is None:
                    break
                t0 = ts()
                events = self._timestamped_seq([get_event(k, KeyEventKind.PRESSED)], t0)
                add_case(k, events)

            for _ in range(target_counts["cmd"]):
                k = next_key()
                if k is None:
                    break
                t0 = ts()
                events = self._timestamped_seq(
                    [get_event(Cmd, KeyEventKind.PRESSED), get_event(k, KeyEventKind.PRESSED)],
                    t0,
                )
                add_case(Cmd / k, events)

            for i in range(target_counts["timed"]):
                k = next_key()
                if k is None:
                    break
                t0 = ts()
                press = get_event(k, KeyEventKind.PRESSED)
                release = get_event(k, KeyEventKind.RELEASED)
                press.pressed_at = t0
                release.pressed_at = t0 + 0.1
                matcher = k @ (0.05, 0.2) if i % 2 == 0 else k @ Wait(seconds=0.2)
                add_case(matcher, [press, release])

            hold_added = 0
            while hold_added < target_counts["hold"]:
                hold_key = next_key()
                if hold_key is None:
                    break
                tap_key = next_key()
                if tap_key is None:
                    break
                t0 = ts()
                keyboard._pressed = {hold_key.code}  # pylint: disable=protected-access
                events = self._timestamped_seq([get_event(tap_key, KeyEventKind.PRESSED)], t0)
                try:
                    add_case(Hold(hold_key, only=True) / tap_key, events, pressed={hold_key.code})
                except ValueError:
                    continue
                else:
                    hold_added += 1
                finally:
                    keyboard._pressed.clear()  # pylint: disable=protected-access

            throttle_added = 0
            while throttle_added < target_counts["throttle"]:
                hold_key = next_key()
                if hold_key is None:
                    break
                t0 = ts()
                keyboard._pressed = {hold_key.code}  # pylint: disable=protected-access
                events = self._timestamped_seq([get_event(hold_key, KeyEventKind.PRESSED)], t0)
                try:
                    add_case(Hold(hold_key, only=True).throttle(every_ms=10), events, pressed={hold_key.code})
                except ValueError:
                    continue
                else:
                    throttle_added += 1
                finally:
                    keyboard._pressed.clear()  # pylint: disable=protected-access

            alt_added = 0
            while alt_added < target_counts["alt"]:
                k1 = next_key()
                if k1 is None:
                    break
                k2 = next_key()
                if k2 is None:
                    break
                t0 = ts()
                events = self._timestamped_seq(
                    [get_event(Cmd, KeyEventKind.PRESSED), get_event(k1, KeyEventKind.PRESSED)],
                    t0,
                )
                try:
                    add_case((Cmd / k1) | (Shift / k2), events)
                except ValueError:
                    continue
                else:
                    alt_added += 1

            for _ in range(target_counts["chord"]):
                k1 = next_key()
                if k1 is None:
                    break
                k2 = next_key()
                if k2 is None:
                    break
                t0 = ts()
                press1 = get_event(k1, KeyEventKind.PRESSED)
                press2 = get_event(k2, KeyEventKind.PRESSED)
                release2 = get_event(k2, KeyEventKind.RELEASED)
                release1 = get_event(k1, KeyEventKind.RELEASED)
                press1.pressed_at = t0
                press2.pressed_at = t0 + 0.01
                release2.pressed_at = t0 + 0.02
                release1.pressed_at = t0 + 0.03
                add_case(k1 & k2, [press1, press2, release2, release1])

            # Pre-compute chars already used by existing routes to avoid collisions with String matchers.
            for trig in bank._pairs.values():
                for sig in trig.signatures:
                    for tok in sig:
                        if tok[0] == "char":
                            used_chars.add(tok[1])
                        if tok[0] == "key":
                            char = _code_to_char(tok[1])
                            if char:
                                used_chars.add(char)
                        if tok[0] == "press_release":
                            char = _code_to_char(tok[1])
                            if char:
                                used_chars.add(char)

            # Fill the rest with string matchers (single-character strings)
            while len(cases) < 80:
                ch = next_string_char()
                if ch is None:
                    break
                if ch in used_chars:
                    continue
                t0 = ts()
                events = self._timestamped_seq(
                    [get_event(ch, KeyEventKind.PRESSED)],
                    t0,
                )
                add_case(String(ch, case=True), events)
                used_chars.add(ch)

            # Validate dispatch
            for case in cases:
                keyboard._pressed = set(case["pressed"])  # pylint: disable=protected-access
                actions = bank.match(case["events"])
                names = [a.properties["name"] for a, _ in actions]
                self.assertEqual(names, [case["name"]])

        finally:
            keyboard._pressed.clear()  # pylint: disable=protected-access
            Matcher.app = prev_app


if __name__ == "__main__":
    unittest.main()
