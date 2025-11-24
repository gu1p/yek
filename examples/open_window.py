"""Example shortcut to open ChatGPT in a browser window."""

from examples._util import open_as_browser_app
from yek import App
from yek.keys import Char, Cmd, Shift

app = App()


@app.on(Cmd / Shift / Char("o"))
def openai(_):
    """Open ChatGPT in the default browser profile."""
    open_as_browser_app(url="https://chat.openai.com/chat", profile="Default")


if __name__ == "__main__":
    app()
