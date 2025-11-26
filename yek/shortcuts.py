"""Shortcut orchestration and hot-reload utilities."""

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods
# pylint: disable=too-many-instance-attributes,too-many-return-statements,duplicate-code

import importlib
import importlib.util
import inspect
import os
import sys
import threading
import time
import uuid
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple
import argparse

from watchdog.events import FileSystemEventHandler, FileSystemEvent  # pylint: disable=import-error
from watchdog.observers import Observer  # pylint: disable=import-error

from yek.matchers import (
    Matcher,
    Result,
    Loop,
    Throttle,
    collision_report,
    matcher_signatures,
    matcher_specificity,
)
from yek.util import KeyEvent, KeyboardStateMap
from .action import Action, Context, SimpleAction

__all__ = ["App", "check_routes"]


def _debug(msg):
    if os.environ.get("DEBUG", False):
        print(msg)


_BUFFER_LIMIT = 5
_COMMAND_ATTR = "_is_command"
_ROUTE_ATTR = "_route"


def _result_span(result: Result) -> int:
    """Return the inclusive span length consumed by a matcher result."""
    if result.value is None:
        return 0
    if result.stop_position < 0:
        return 0
    start = max(result.start_position, 0)
    stop = max(result.stop_position, start)
    return stop - start + 1


def _signature_tokens_overlap(x, y) -> bool:
    if x == y:
        return True
    if x[0] == "press_release" and y[0] == "press_release" and x[1] == y[1]:
        return not (x[3] < y[2] or y[3] < x[2])  # time windows overlap
    if x[0] == "press_release" and y[0] == "key" and x[1] == y[1]:
        return True
    if y[0] == "press_release" and x[0] == "key" and x[1] == y[1]:
        return True
    return False


def _is_strict_signature_subsequence(smaller: List[tuple], bigger: List[tuple]) -> bool:
    """
    Return True if any signature in smaller is a contiguous subsequence of a longer
    one in bigger.
    """
    for s in smaller:
        for b in bigger:
            if len(s) >= len(b):
                continue
            for i in range(len(b) - len(s) + 1):
                window = b[i:i + len(s)]
                if all(_signature_tokens_overlap(sa, sb) for sa, sb in zip(s, window)):
                    return True
    return False


def _has_special_tokens(sigs: List[tuple]) -> bool:
    """Return True if signatures include non-plain key tokens (hold/chord)."""
    special = {"hold", "hold_only", "chord", "chord_only"}
    return any(tok[0] in special for sig in sigs for tok in sig)


def _matcher_has_active_loop(matcher: Matcher) -> bool:
    """Return True if matcher tree contains an armed Loop."""
    from yek.matchers import (  # pylint: disable=import-outside-toplevel,protected-access
        _MatchSequence,
        _Or,
        _And,
        _AndKeys,
        _AndStartEndMatcher,
    )

    if isinstance(matcher, Loop):
        return matcher.is_active()
    if isinstance(matcher, Throttle):
        return _matcher_has_active_loop(matcher.inner_matcher)
    if isinstance(matcher, _MatchSequence):
        return any(
            _matcher_has_active_loop(child) for child in matcher._matchers  # pylint: disable=protected-access
        )
    if isinstance(matcher, (_Or, _And)):
        return any(
            _matcher_has_active_loop(child) for child in matcher._matchers  # pylint: disable=protected-access
        )
    if isinstance(matcher, _AndKeys):
        return False
    if isinstance(matcher, _AndStartEndMatcher):
        return (
            _matcher_has_active_loop(matcher.a)
            or _matcher_has_active_loop(matcher.b)
        )  # pylint: disable=protected-access
    return False


class ActionTrigger:
    def __init__(self, matcher: Matcher, action: Action, priority: Optional[int] = None):
        self.action = action
        self.matcher = matcher
        self._last_match = 0
        self._last_result: Optional[Result] = None
        self.signatures = matcher_signatures(matcher)
        self.priority = priority if priority is not None else matcher_specificity(matcher)
        self._order = 0  # set by ActionBankTrigger.register

    def match(self, events: List[KeyEvent]) -> Result:
        events = [e for e in events if e.pressed_at > self._last_match]
        match = self.matcher.match(events)
        if match:
            self._last_result = match
            self._last_match = time.time()
        return match


class ActionBankTrigger:
    def __init__(self):
        self._lock = threading.Lock()
        self._pairs: Dict[uuid.UUID, ActionTrigger] = {}
        self._properties = {}
        self._order_counter = 0
        self._active_loop: Optional[uuid.UUID] = None

    def match(self, events: List[KeyEvent]) -> List[Tuple[Action, Result]]:
        with self._lock:
            # If a looping route is already active, let it keep consuming first.
            if self._active_loop is not None:
                trigger = self._pairs.get(self._active_loop)
                if trigger:
                    result = trigger.match(events)
                    still_active = _matcher_has_active_loop(trigger.matcher)
                    if result and still_active:
                        return [(trigger.action, result)]
                    if not result and still_active:
                        # Loop remains armed but nothing new to consume; do not let
                        # competing routes win this slice.
                        return []
                self._active_loop = None

            candidates = []
            for trigger in self._pairs.values():
                result = trigger.match(events)
                if not result:
                    continue

                span = _result_span(result)
                tokens = result.matched_tokens or matcher_specificity(trigger.matcher)
                candidates.append(
                    (
                        -span,  # largest span first
                        -tokens,  # most tokens first
                        -trigger.priority,  # highest priority wins
                        trigger._order,  # pylint: disable=protected-access
                        trigger.action,
                        result,
                    )
                )
                debug_msg = (
                    f"Matched trigger: {trigger.matcher.debug()} ({trigger.action}) "
                    f"span={span} tokens={tokens}"
                )
                _debug(debug_msg)

            if not candidates:
                return []

            candidates.sort()
            _, _, _, _, action, result = candidates[0]

            # Arm the active loop if applicable to avoid competing routes while the loop is active.
            if _matcher_has_active_loop(self._pairs[action.id].matcher):  # type: ignore[index]
                self._active_loop = action.id

            return [(action, result)]

    def register(self, trigger: ActionTrigger) -> None:
        with self._lock:
            self._assert_guardrails(trigger.matcher)
            self._assert_no_collision(trigger)
            trigger._order = self._order_counter  # pylint: disable=protected-access
            self._order_counter += 1
            self._pairs[trigger.action.id] = trigger

    def find_by_prop(self, prop: str, value: str) -> List[ActionTrigger]:
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

    def _assert_no_collision(self, trigger: ActionTrigger) -> None:
        """
        Prevent registering two routes that can match the same key combination.
        Collisions are detected via signature subsequence checks.
        """
        if not trigger.signatures:
            return

        for existing in self._pairs.values():
            if trigger.matcher.collides_with(existing.matcher):
                signatures_overlap = (
                    _is_strict_signature_subsequence(trigger.signatures, existing.signatures)
                    or _is_strict_signature_subsequence(existing.signatures, trigger.signatures)
                )
                has_special_tokens = (
                    _has_special_tokens(trigger.signatures)
                    or _has_special_tokens(existing.signatures)
                )
                if not has_special_tokens and signatures_overlap:
                    continue
                summary = (
                    "Collision summary:\n"
                    f"- {self._route_descriptor(trigger)}\n"
                    f"- {self._route_descriptor(existing)}"
                )
                raise ValueError(
                    "Collision detected between routes:\n"
                    f"- {trigger.matcher.debug()} ({trigger.action}) "
                    f"signatures={trigger.signatures}\n"
                    f"- {existing.matcher.debug()} ({existing.action}) "
                    f"signatures={existing.signatures}\n"
                    f"Reason: {collision_report(trigger.matcher, existing.matcher)}\n"
                    "Adjust the shortcut (e.g., add a modifier, use Hold(..., only=True), "
                    "or set explicit priority to resolve.\n"
                    f"{summary}"
                )

    @staticmethod
    def _route_descriptor(trigger: ActionTrigger) -> str:
        props = trigger.action.properties or {}
        file = props.get("file", "?")
        line = props.get("line", "?")
        return f'"{file}:{line}" - {trigger.matcher.debug()}'

    def _assert_guardrails(self, matcher: Matcher) -> None:
        """
        Enforce structural constraints (e.g., Loop must be prefixed).
        """
        if isinstance(matcher, Loop):
            if not matcher.prefixed:
                raise ValueError(
                    "Loop matcher must follow a prefix and be the last element in a sequence."
                )
            return

        if isinstance(matcher, Throttle):
            self._assert_guardrails(matcher.inner_matcher)
            return

        children = getattr(matcher, "_matchers", None)
        if children:
            for child in children:
                self._assert_guardrails(child)


def _load_action_triggers(file_path: str) -> List[ActionTrigger]:
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)

        route_mapping = []
        app_instance = None

        # First, find the App instance
        for _, obj in inspect.getmembers(module):
            if isinstance(obj, App):
                app_instance = obj
                break

        if app_instance is None:
            return []

        # Look for route decorators
        for _, obj in inspect.getmembers(module):
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Failed to reload routes: {e}")  # pragma: no cover
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

        self._keyboard = KeyboardStateMap()  # pylint: disable=abstract-class-instantiated

        # Hot-reload
        self._observer = None
        self._reloader = None
        self.routes_file = None

        Matcher.app = self

    def on(self, matcher: Matcher, *, priority: Optional[int] = None) -> Callable:
        _debug(matcher.debug())

        def _decorator(func: Callable) -> ActionTrigger:
            if not isinstance(func, Callable):
                raise ValueError("on decorator must be used with a callable")

            trigger = ActionTrigger(matcher, SimpleAction(func), priority=priority)
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
        if not actions:
            return

        action, result = actions[0]
        action.execute(Context(self._keyboard, match_result=result))

        # Advance buffer past consumed events to avoid reusing the same slice.
        consumed_idx = result.stop_position
        if consumed_idx < 0 or consumed_idx >= len(keys):
            consumed_idx = len(keys) - 1
        consumed_idx = max(0, consumed_idx)
        consumed_time = keys[consumed_idx].pressed_at if keys else time.time()
        self._last_called = max(self._last_called, consumed_time + 1e-6)

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


def check_routes(file_path: str) -> List[ActionTrigger]:
    """
    Load and validate routes defined in the given file.

    Returns the registered ActionTriggers or raises on collision.
    """
    triggers = _load_action_triggers(file_path)
    if not triggers:
        raise ValueError(f"No routes found in {file_path}")
    return triggers


def _main():
    parser = argparse.ArgumentParser(description="Validate yek route files for collisions.")
    parser.add_argument("file", help="Path to the Python file that defines routes.")
    args = parser.parse_args()

    try:
        triggers = check_routes(args.file)
        print(f"Routes OK ({len(triggers)} registered) for {args.file}")
    except Exception as exc:  # pragma: no cover  # pylint: disable=broad-exception-caught
        print(exc)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    _main()
