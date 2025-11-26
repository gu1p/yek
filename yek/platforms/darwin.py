"""macOS-specific keyboard state helpers."""

import os
import subprocess
import sys
from contextlib import suppress

from yek.platforms.common import PynputKeyboardState


def _running_interactively() -> bool:
    """Return True if stdout/stderr are attached to a TTY."""
    return sys.stdout.isatty() or sys.stderr.isatty()


class MacKeyboardState(PynputKeyboardState):
    """
    macOS keyboard state that nudges the user to grant Input Monitoring permission.
    """

    _prompted = False

    def start(self):
        try:
            super().start()
        except Exception as exc:
            self._prompt_permissions(exc)
            raise

    def _prompt_permissions(self, exc: Exception):
        del exc
        if MacKeyboardState._prompted:
            return
        if os.environ.get("YEK_SKIP_MAC_PROMPT"):
            return

        MacKeyboardState._prompted = True
        with suppress(Exception):
            message = (
                "yek could not start keyboard listener "
                "(macOS Input Monitoring permission may be missing). "
                "Grant access in System Settings > Privacy & Security > Input Monitoring for this "
                "executable."
            )
            if _running_interactively():
                print(
                    f"{message} Opening Input Monitoring settings...",
                    file=sys.stderr,
                )
                subprocess.Popen(  # pylint: disable=consider-using-with
                    [
                        "open",
                        "x-apple.systempreferences:com.apple.preference.security?Privacy_Keyboard",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                print(message, file=sys.stderr)
