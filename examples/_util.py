"""Helper utilities for example scripts."""

import shutil
import subprocess
import webbrowser
from typing import Optional


def open_as_browser_app(
    url: str, profile: str = "Default", browser: Optional[str] = None
) -> subprocess.Popen:
    """Open the given URL as a standalone browser window using the provided profile."""
    print(f"Opening {url} in browser")
    browser = browser or get_browser()
    if browser is None:
        # Fallback to system default browser
        webbrowser.open(url)
        return None

    chrome_cmd = [
        browser,
        "--app=" + url,
        "--window-size=300,300",
        f"--profile-directory={profile}",
    ]

    process = subprocess.Popen(  # pylint: disable=consider-using-with
        chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return process


def get_browser() -> str:
    """Return a supported browser executable path if available."""
    for browser in ["brave-browser", "brave", "google-chrome", "google-chrome-stable"]:
        browser = shutil.which(browser)
        if browser is not None:
            return browser
    return None
