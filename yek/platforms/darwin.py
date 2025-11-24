import os
import subprocess
from contextlib import suppress

from yek.platforms.common import PynputKeyboardState


class MacKeyboardState(PynputKeyboardState):
    """
    macOS keyboard state that nudges the user to grant Input Monitoring permission.
    """

    _prompted = False

    def _prompt_permissions(self):
        if self.__class__._prompted:
            return
        if os.environ.get("YEK_SKIP_MAC_PROMPT"):
            return

        self.__class__._prompted = True
        # Open System Settings → Privacy & Security → Keyboard/Input Monitoring
        with suppress(Exception):
            print("yek needs keyboard permissions. Opening macOS Input Monitoring settings...")
            subprocess.Popen(
                ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Keyboard"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def start(self):
        self._prompt_permissions()
        super().start()
