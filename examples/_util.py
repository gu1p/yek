import shutil
import subprocess
from typing import Optional


def open_as_browser_app(
    url: str, profile: str = "Default", browser: Optional[str] = None
) -> subprocess.Popen:
    print(f"Opening {url} in browser")
    browser = browser or get_browser()
    chrome_cmd = [
        browser,
        "--app=" + url,
        "--window-size=300,300",
        f"--profile-directory={profile}",
    ]

    process = subprocess.Popen(
        chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    return process


def get_browser() -> str:
    for browser in ["brave-browser", "brave", "google-chrome", "google-chrome-stable"]:
        browser = shutil.which(browser)
        if browser is not None:
            return browser

    raise ValueError("browser not found")
