import os
import subprocess
from contextlib import suppress

from yek.platforms.common import PynputKeyboardState


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
        if self.__class__._prompted:
            return
        if os.environ.get("YEK_SKIP_MAC_PROMPT"):
            return

        self.__class__._prompted = True
        with suppress(Exception):
            print(
                "yek could not start keyboard listener (macOS permissions may be missing). "
                "Opening Input Monitoring settings..."
            )
            subprocess.Popen(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Keyboard"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
