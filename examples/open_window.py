from examples._util import open_as_browser_app
from yek import App
from yek.keys import *

app = App()


@app.on(Cmd / Shift / Char("o"))
def openai(_):
    open_as_browser_app(url="https://chat.openai.com/chat", profile="Default")


if __name__ == "__main__":
    app()
