"""Tests for macOS permission prompting behavior."""

import sys
import unittest
from unittest.mock import patch

from tests.pynput_utils import require_pynput

require_pynput()

# pylint: disable=wrong-import-position
from yek.platforms.darwin import MacKeyboardState


class MacPermissionPromptTests(unittest.TestCase):
    """macOS keyboard permissions should behave differently in daemons."""

    def setUp(self):
        MacKeyboardState._prompted = False  # pylint: disable=protected-access

    def tearDown(self):
        MacKeyboardState._prompted = False  # pylint: disable=protected-access

    def test_interactive_sessions_open_settings(self):
        """When attached to a TTY, we should open System Settings to request permission."""
        keyboard = MacKeyboardState()

        with patch("builtins.print") as print_mock, \
                patch("yek.platforms.darwin.subprocess.Popen") as popen_mock, \
                patch("yek.platforms.darwin._running_interactively", return_value=True):
            keyboard._prompt_permissions(Exception("boom"))  # pylint: disable=protected-access

        popen_mock.assert_called_once()
        print_mock.assert_called_once()
        self.assertIn("Input Monitoring", print_mock.call_args.args[0])

    def test_daemon_mode_logs_without_prompting(self):
        """Without a TTY, warn to stderr but do not try to open settings UI."""
        keyboard = MacKeyboardState()

        with patch("builtins.print") as print_mock, \
                patch("yek.platforms.darwin.subprocess.Popen") as popen_mock, \
                patch("yek.platforms.darwin._running_interactively", return_value=False):
            keyboard._prompt_permissions(Exception("boom"))  # pylint: disable=protected-access

        popen_mock.assert_not_called()
        print_mock.assert_called_once()
        args, kwargs = print_mock.call_args
        self.assertIn("Input Monitoring", args[0])
        self.assertEqual(kwargs.get("file"), sys.stderr)


if __name__ == "__main__":
    unittest.main()
