import importlib
import importlib.util
import inspect
import os
import sys
import threading
import time
import uuid
from time import sleep
from typing import Callable, Any, Dict

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from yek.matchers import Matcher, Result
from yek.util import KeyEvent, KeyboardStateMap
from .action import Action, Context, SimpleAction

__all__ = ["App"]


def _debug(msg):
    if os.environ.get("DEBUG", False):
        print(msg)


_BUFFER_LIMIT = 5
_COMMAND_ATTR = "_is_command"
_ROUTE_ATTR = "_route"


class ActionTrigger:
    def __init__(self, matcher: Matcher, action: Action):
        self.action = action
        self.matcher = matcher
        self._last_match = 0

    def match(self, events: list[KeyEvent]) -> Result:
        events = [e for e in events if e.pressed_at > self._last_match]
        match = self.matcher.match(events)
        if match:
            self._last_match = time.time()
        return match


class ActionBankTrigger:
    def __init__(self):
        self._lock = threading.Lock()
        self._pairs: Dict[uuid, ActionTrigger] = {}
        self._properties = {}

    def match(self, events: list[KeyEvent]) -> list[Action]:
        with self._lock:
            actions = []
            for trigger in self._pairs.values():
                if trigger.match(events):
                    actions.append(trigger.action)
            return actions

    def register(self, trigger: ActionTrigger) -> None:
        with self._lock:
            self._pairs[trigger.action.id] = trigger

    def find_by_prop(self, prop: str, value: str) -> list[ActionTrigger]:
        registers = []
        for _, register in self._pairs.items():
            prop_value = register.action.properties.get(prop)
            if prop_value == value:
                registers.append(register)
        return registers

    def remove_by_prop(self, prop: str, value: Any):
        with self._lock:
            for register in self.find_by_prop(prop, value):
                self._pairs.pop(register.action.id, None)


def _load_action_triggers(file_path: str) -> list[ActionTrigger]:
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)

        route_mapping = []
        app_instance = None

        # First, find the App instance
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, App):
                app_instance = obj
                break

        if app_instance is None:
            return []

        # Look for route decorators
        for name, obj in inspect.getmembers(module):
            if isinstance(obj, ActionTrigger):
                route_mapping.append(obj)
        return route_mapping
    except SyntaxError:
        return []
    finally:
        # Clean up the temporary module from sys.modules
        if "module" in sys.modules:
            del sys.modules["module"]


class _HotReloader(FileSystemEventHandler):
    def __init__(self, app: "App"):
        self.app = app
        self.last_modified = 0

    def on_modified(self, event: FileSystemEvent):
        if self.app.action_triggers.find_by_prop("file", event.src_path):
            current_time = time.time()
            if current_time - self.last_modified > 1:  # Debounce
                self.last_modified = current_time
                _debug(f"Detected change in {event.src_path}. Reloading routes...")
                self.reload_routes(event.src_path)

    def reload_routes(self, path: str):
        try:
            triggers = _load_action_triggers(path)
            self.app.action_triggers.remove_by_prop("file", path)
            for trigger in triggers:
                print(f"Registering: {trigger.action} as {trigger.matcher.debug()}")
                self.app.action_triggers.register(trigger)
        except Exception as e:
            print(f"Failed to reload routes: {e}")
        else:
            print("Routes reloaded successfully.")


class App:
    def __init__(self,
                 buffer_timeout: float = _BUFFER_LIMIT,
                 sequence_timeout: float = 2,
                 ):
        self.action_triggers: ActionBankTrigger = ActionBankTrigger()
        self._buffer_timeout = buffer_timeout
        self._sequence_timeout = sequence_timeout
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._last_called: float = 0

        self._keyboard = KeyboardStateMap()

        # Hot-reload
        self._observer = None
        self._reloader = None
        self.routes_file = None

        Matcher.app = self

    def on(self, matcher: Matcher) -> Callable:
        _debug(matcher.debug())

        def _decorator(func: Callable) -> ActionTrigger:
            if not isinstance(func, Callable):
                raise ValueError("on decorator must be used with a callable")

            trigger = ActionTrigger(matcher, SimpleAction(func))
            self.action_triggers.register(trigger)
            return trigger

        return _decorator

    def __call__(self):
        self._keyboard.start()
        self._running.set()

        # Hot-reload
        self._setup_hot_reload()

        try:
            while self._running.is_set():
                # self._process_running_loops()
                self._check_shortcuts()
                sleep(0.01)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self._keyboard.stop()
        self._running.clear()
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def _since(self):
        window = min(self._buffer_timeout, self._sequence_timeout)
        return max(self._last_called, time.time() - window)

    def _check_shortcuts(self):
        keys = self._keyboard.get_since(self._since())
        if len(keys) == 0:
            return

        actions = self.action_triggers.match(keys)
        for action in actions:
            action.execute(Context(self._keyboard))
        if actions:
            self._last_called = time.time()

    def _setup_hot_reload(self):
        # Get the calling module (where the routes are defined)
        calling_frame = inspect.stack()[2]
        module = inspect.getmodule(calling_frame[0])
        self.routes_file = module.__file__

        self._reloader = _HotReloader(self)
        self._observer = Observer()
        self._observer.schedule(
            self._reloader, path=os.path.dirname(self.routes_file), recursive=False
        )
        self._observer.start()
