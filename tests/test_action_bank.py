"""Tests for action bank evaluating multiple shortcuts."""

import unittest
from time import time

# pylint: disable=duplicate-code,invalid-name
try:
    import pynput  # type: ignore
except ImportError:
    pynput = None

if pynput is not None:
    from yek.action import SimpleAction
    from yek.events import KeyEventKind
    from yek.key_utils import get_event
    from yek.keys import Cmd, Shift, Char
    from yek.shortcuts import ActionBankTrigger, ActionTrigger
else:
    SimpleAction = (
        KeyEventKind
    ) = (
        get_event
    ) = (
        Cmd
    ) = Shift = Char = ActionBankTrigger = ActionTrigger = None  # type: ignore  # pylint: disable=invalid-name


@unittest.skipIf(pynput is None, "pynput not installed")
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
            names.extend([a.properties["name"] for a in actions])

        self.assertEqual(names, ["one", "two", "three"])


if __name__ == "__main__":
    unittest.main()
