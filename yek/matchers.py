"""Keyboard matchers and combinators."""

# pylint: disable=missing-class-docstring,missing-function-docstring,too-many-lines
# pylint: disable=too-many-return-statements,too-many-branches,too-many-locals,line-too-long
# pylint: disable=duplicate-code

import abc
import math
import time
from dataclasses import dataclass
from itertools import chain, product
from typing import TYPE_CHECKING, List, Optional, Sequence, Set, Tuple, Union

from yek.events import KeyEvent, KeyEventKind
from yek.time import Wait

if TYPE_CHECKING:
    from yek import App

__all__ = [
    "Key",
    "StringMatcher",
    "Matcher",
    "_TimedPressRelease",
    "_MatchSequence",
    "Hold",
    "Loop",
    "Throttle",
    "matcher_signatures",
    "matcher_specificity",
    "collision_report",
]


@dataclass
class Match:
    start: KeyEvent
    end: KeyEvent


@dataclass
class Result:
    stop_position: int
    start_position: int = 0
    value: Optional[Match] = None
    matched_tokens: int = 0

    def __bool__(self) -> bool:
        return self.value is not None


class Matcher(abc.ABC):
    app: "App" = None

    @abc.abstractmethod
    def match(self, events: Sequence[KeyEvent]) -> Result:
        """Return a Result when events satisfy the matcher."""

    @abc.abstractmethod
    def __truediv__(self, other: Union["Matcher", Tuple[int, int]]) -> "Matcher":
        """Compose matchers in sequence."""

    @abc.abstractmethod
    def __or__(self, other: "Matcher") -> "Matcher":
        """Alternate between two matchers."""

    @abc.abstractmethod
    def __and__(self, other: "Matcher") -> "Matcher":
        """Require both matchers."""

    @abc.abstractmethod
    def __matmul__(self, other: Union[float, Wait]) -> "Matcher":
        """Combine with a timing constraint."""

    @abc.abstractmethod
    def debug(self) -> str:
        """Debug description of the matcher."""

    def throttle(self, every_ms: int = 100) -> "Throttle":
        """Wrap this matcher in a Throttle."""
        return Throttle(self, every_ms=every_ms)

    # Collision detection ------------------------------------------------------
    def collides_with(self, other: "Matcher") -> bool:
        """Return True if this matcher can overlap with another."""
        return _collides(self, other)


class Key(Matcher):
    def __init__(
            self,
            name: str,
            code: str,
            char: Optional[str] = None,
            kind: KeyEventKind = KeyEventKind.PRESSED,
    ):
        self._name = name
        self._code = code
        self._char = char
        self._event_kind = kind

    def __hash__(self) -> int:
        return hash(self._code)

    def __repr__(self) -> str:
        return f"Key({self._name})"

    def debug(self) -> str:
        return f"Key({self._name})"

    @property
    def code(self) -> str:
        return self._code

    def match(self, events: Sequence[KeyEvent]) -> Result:
        for n, event in enumerate(events):
            if event.code == self._code and event.kind == self._event_kind:
                return Result(
                    value=Match(start=event, end=event),
                    start_position=n,
                    stop_position=n,
                    matched_tokens=1,
                )
        return Result(stop_position=len(events), start_position=len(events))

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        def _to_seconds(value: Union[float, int, Wait]) -> float:
            if isinstance(value, Wait):
                return value.seconds
            if isinstance(value, (int, float)):
                return float(value)
            raise ValueError(f"Unsupported wait value type {type(value)}")

        if isinstance(other, tuple):
            if len(other) != 2:
                raise ValueError("Timed press-release tuple must have exactly 2 values")
            start, end = (_to_seconds(other[0]), _to_seconds(other[1]))
            if not 0 <= start < end:
                raise ValueError("Timed press-release must satisfy 0 <= start < end")
            return _TimedPressRelease(self, start=start, end=end)

        duration = _to_seconds(other)
        if duration <= 0:
            raise ValueError("Timed press-release duration must be positive")
        return _TimedPressRelease(self, start=0, end=duration)

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> Matcher:
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def __add__(self, other: Matcher) -> Matcher:
        if isinstance(other, Key):
            return _AndKeys(self, other, only=False)
        return _and(self, other)

    def __radd__(self, other: Matcher) -> Matcher:
        if isinstance(other, Key):
            return _AndKeys(other, self, only=False)
        return _and(other, self)

    def collides_with(self, other: Matcher) -> bool:
        return _collides_key(self, other)


class StringMatcher(Matcher):
    def __init__(self, string: str, case: bool = True):
        self._string = string
        self._case = case

    def __hash__(self) -> int:
        return hash(self._string)

    def __repr__(self) -> str:
        return f"String({self._string})"

    def debug(self) -> str:
        return f"String({self._string})"

    def match(self, events: Sequence[KeyEvent]) -> Result:
        matched_indexes = []
        i = 0
        for char in self._string:
            while i < len(events):
                # Hacky way to ignore non-character keys
                event_char = str(events[i].key).strip("'")
                if len(char) > 1:
                    i += 1
                    continue

                event_char = event_char if self._case else event_char.lower()
                normalized = char if self._case else char.lower()

                if event_char == normalized:
                    matched_indexes.append(i)
                    i += 1
                    break

                i += 1
            else:
                return Result(stop_position=i, start_position=i)

        return Result(
            value=Match(
                start=events[matched_indexes[0]],
                end=events[matched_indexes[-1]]
            ),
            start_position=matched_indexes[0],
            stop_position=i,
            matched_tokens=len(self._string),
        )

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        raise ValueError("Cannot chain timed press-release with a string matcher")

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> Matcher:
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def collides_with(self, other: Matcher) -> bool:
        return _collides_string(self, other)


class _TimedPressRelease(Matcher):
    def __init__(self, key: "Key", start: float, end: float):
        self._key = key
        self._start = start
        self._end = end

    @property
    def key(self) -> "Key":
        return self._key

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    def match(self, events: Sequence[KeyEvent]) -> Result:
        for i, event in enumerate(events):
            if event.code == self._key.code and event.kind == KeyEventKind.PRESSED:
                for j, event2 in enumerate(events[i + 1:]):
                    if event2.code == self._key.code and event2.kind == KeyEventKind.RELEASED:
                        if self._start <= event2.pressed_at - event.pressed_at <= self._end:
                            return Result(
                                value=Match(start=event, end=event2),
                                start_position=i,
                                stop_position=i + j,
                                matched_tokens=1,
                            )
        return Result(stop_position=len(events), start_position=len(events))

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        return _MatchSequence(self, other)

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        raise ValueError("Cannot chain timed press-release with another timed press-release")

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def debug(self) -> str:
        return f"TimedPressRelease({self._key.debug()}, {self._start}, {self._end})"

    def collides_with(self, other: Matcher) -> bool:
        return _collides_timed(self, other)


class _MatchSequence(Matcher):
    def __init__(self, *matchers: Matcher):
        self._matchers = []
        for matcher in matchers:
            if isinstance(matcher, _MatchSequence):
                self._matchers.extend(matcher._matchers)
            else:
                self._matchers.append(matcher)

        loop_indices = [i for i, matcher in enumerate(self._matchers) if isinstance(matcher, Loop)]
        if loop_indices:
            if len(loop_indices) > 1:
                raise ValueError("Loop cannot appear more than once in a sequence")
            loop_idx = loop_indices[0]
            if loop_idx != len(self._matchers) - 1:
                raise ValueError("Loop must be the last matcher in a sequence")
            if loop_idx > 0:
                self._matchers[loop_idx].mark_prefixed()  # type: ignore[attr-defined]

        self._hold_tail = _is_hold_like(self._matchers[-1]) if self._matchers else False
        self._armed = False

    def match(self, events: Sequence[KeyEvent]) -> Result:
        if self._hold_tail and self._armed:
            tail = self._matchers[-1]
            tail_result = tail.match(events)
            if tail_result:
                return tail_result
            if isinstance(tail, Loop) and tail.is_active():
                return Result(stop_position=len(events))
            self._armed = False
            return Result(stop_position=len(events))

        position = 0
        matched_codes: Set[str] = set()
        matches = []
        first_start: Optional[int] = None
        last_stop: Optional[int] = None
        tokens = 0
        for idx, matcher in enumerate(self._matchers):
            # Skip releases of keys we already matched; other events break the sequence.
            while position < len(events):
                ev = events[position]
                if ev.kind == KeyEventKind.RELEASED and ev.code in matched_codes:
                    position += 1
                    continue
                break

            result = matcher.match(events[position:])
            if not result:
                return Result(stop_position=len(events))
            if result.start_position != 0:
                if idx == 0:
                    position += result.start_position
                    result = matcher.match(events[position:])
                    if not result or result.start_position != 0:
                        return Result(stop_position=len(events))
                else:
                    skipped = events[position:position + result.start_position]
                    allowed = all(
                        ev.kind == KeyEventKind.RELEASED and ev.code in matched_codes
                        for ev in skipped
                    )
                    if not allowed:
                        return Result(stop_position=len(events))
                    position += result.start_position

            abs_start = position + result.start_position
            abs_stop = position + result.stop_position
            if first_start is None:
                first_start = abs_start
            if result.stop_position >= 0:
                last_stop = abs_stop

            matches.append(result.value)
            tokens += result.matched_tokens
            position += result.stop_position + 1
            matched_codes.update(_matcher_codes(matcher))

        stop_position = last_stop if last_stop is not None else max(position - 1, 0)
        start_position = first_start if first_start is not None else 0
        if self._hold_tail:
            self._armed = True
        return Result(
            value=Match(start=matches[0].start, end=matches[-1].end),
            start_position=start_position,
            stop_position=stop_position,
            matched_tokens=tokens,
        )

    def __matmul__(self, other) -> "Matcher":
        raise ValueError("Cannot chain timed press-release with a match sequence")

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def debug(self) -> str:
        return f"MatchSequence({', '.join([m.debug() for m in self._matchers])})"

    def collides_with(self, other: Matcher) -> bool:
        return _collides_sequence(self, other)


class _Or(Matcher):
    def __init__(self, *matchers: Matcher):
        self._matchers = matchers

    def match(self, events: Sequence[KeyEvent]) -> Result:
        for matcher in self._matchers:
            match = matcher.match(events)
            if match:
                return match
        return Result(stop_position=len(events))

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        return _Or(*[m.__matmul__(other) for m in self._matchers])

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def debug(self) -> str:
        return f"Or({', '.join([m.debug() for m in self._matchers])})"

    def collides_with(self, other: Matcher) -> bool:
        return any(_collides(m, other) for m in self._matchers)


class _And(Matcher):
    def __init__(self, *matchers: Matcher):
        self._matchers = matchers

    def match(self, events: Sequence[KeyEvent]) -> Result:
        matches = []
        stop_position = 0
        start_position = math.inf
        tokens = 0
        for matcher in self._matchers:
            match = matcher.match(events)
            if not match:
                return Result(stop_position=len(events))

            stop_position = max(stop_position, match.stop_position)
            start_position = min(start_position, match.start_position)
            matches.append(match.value)
            tokens += match.matched_tokens
        return Result(
            value=Match(start=matches[0].start, end=matches[-1].end),
            start_position=start_position if start_position is not math.inf else 0,
            stop_position=stop_position,
            matched_tokens=tokens,
        )

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        return _and(*[m.__matmul__(other) for m in self._matchers])

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def debug(self) -> str:
        return f"And({', '.join([m.debug() for m in self._matchers])})"

    def collides_with(self, other: Matcher) -> bool:
        return any(_collides(m, other) for m in self._matchers)


class _AndKeys(Matcher):
    def __init__(self, *keys: Key, only: bool = True):
        for k in keys:
            if not isinstance(k, Key):
                raise ValueError(f"Expected Key instance, got {type(k)}")

        self._keys = keys
        self._only = only

    def match(self, events: Sequence[KeyEvent]) -> Result:
        intervals = []
        min_stop, max_stop = math.inf, 0
        min_start = math.inf
        for key in self._keys:
            match = key.match(events)
            min_stop = min(min_stop, match.stop_position)
            max_stop = max(max_stop, match.stop_position)
            min_start = min(min_start, match.start_position)

            if not match:
                return Result(stop_position=len(events))

            press_event = match.value.start
            release_event = press_event.get_key_next_event(events)

            intervals.append((press_event, release_event))

        intersection = get_common_intersection(intervals)

        if intersection:
            if self._only:
                start_time = intersection[0].pressed_at if intersection[0] else 0
                end_time = intersection[1].pressed_at if intersection[1] else math.inf
                allowed = {k.code for k in self._keys}
                for event in events:
                    if (
                        event.kind == KeyEventKind.PRESSED
                        and start_time <= event.pressed_at <= end_time
                    ):
                        if event.code not in allowed:
                            stop = min_stop if math.isfinite(min_stop) else len(events)
                            return Result(stop_position=stop)
            return Result(
                value=Match(
                    start=intersection[0],
                    end=intersection[1]
                ),
                start_position=min_start if math.isfinite(min_start) else 0,
                stop_position=max_stop,
                matched_tokens=len(self._keys),
            )

        return Result(stop_position=min_stop)

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        return _and(*[k.__matmul__(other) for k in self._keys])

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> Matcher:
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def debug(self) -> str:
        suffix = ", only=False" if not self._only else ""
        return f"AndKeys({', '.join([k.debug() for k in self._keys])}{suffix})"

    def collides_with(self, other: Matcher) -> bool:
        return _collides_chord(self, other)


class _AndStartEndMatcher(Matcher):
    def __init__(self, a: Key, b: Union[_TimedPressRelease, _MatchSequence, StringMatcher]):
        self.a = a
        self.b = b

    def match(self, events: Sequence[KeyEvent]) -> Result:
        match_a = self.a.match(events)
        if not match_a:
            return Result(stop_position=match_a.stop_position)
        end_a = match_a.value.start.get_key_next_event(events)

        match_b = self.b.match(events)
        if not match_b:
            return Result(stop_position=match_b.stop_position)

        intersection = get_common_intersection([
            (match_a.value.start, end_a),
            (match_b.value.start, match_b.value.end)
        ])

        if intersection:
            stop = max(match_a.stop_position, match_b.stop_position)
            return Result(
                value=Match(start=intersection[0], end=intersection[1]),
                start_position=min(match_a.start_position, match_b.start_position),
                stop_position=stop,
                matched_tokens=match_a.matched_tokens + match_b.matched_tokens,
            )

        stop = min(match_a.stop_position, match_b.stop_position)
        return Result(stop_position=stop)

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        return _and(self.a.__matmul__(other), self.b.__matmul__(other))

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> Matcher:
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def debug(self) -> str:
        return f"AndStartEndMatcher({self.a.debug()}, {self.b.debug()})"

    def collides_with(self, other: Matcher) -> bool:
        return _collides(self.a, other) or _collides(self.b, other)


def get_common_intersection(
    events: List[Tuple[Optional[KeyEvent], Optional[KeyEvent]]]
) -> Optional[Tuple[Optional[KeyEvent], Optional[KeyEvent]]]:
    if not events:
        return None

    cache = {}
    # Sort events by start time, treating None as negative infinity
    sorted_events = sorted(events, key=lambda x: x[0].pressed_at if x[0] else 0)

    # Initialize the intersection
    intersection_start = sorted_events[0][0].pressed_at if sorted_events[0][0] else 0
    intersection_end = sorted_events[0][1].pressed_at if sorted_events[0][1] else math.inf

    cache[intersection_start] = sorted_events[0][0]
    cache[intersection_end] = sorted_events[0][1]

    # Iterate through the rest of the events
    for start, end in sorted_events[1:]:
        start_time = start.pressed_at if start else 0
        end_time = end.pressed_at if end else math.inf

        cache[start_time] = start
        cache[end_time] = end

        # If the current event starts after the intersection ends, there's no overlap
        if start_time >= intersection_end:
            return None

        # Update the intersection
        intersection_start = max(intersection_start, start_time)
        intersection_end = min(intersection_end, end_time)

    # Return the final intersection interval
    return cache.get(intersection_start), cache.get(intersection_end)


def _and(*a: Matcher) -> Matcher:
    # pylint: disable=too-many-return-statements,too-many-branches,protected-access
    if len(a) == 0:
        raise ValueError("Cannot AND zero matchers")
    if len(a) == 1:
        return a[0]
    if len(a) > 2:
        return _and(a[0], _and(a[1], *a[2:]))

    a, b = a
    if isinstance(a, Key) and isinstance(b, Key):
        return _AndKeys(a, b)

    if isinstance(a, _AndKeys) and isinstance(b, Key):
        return _AndKeys(*a._keys, b, only=a._only)

    if isinstance(a, Key) and isinstance(b, _AndKeys):
        return _AndKeys(*b._keys, a, only=b._only)

    if isinstance(a, Key) and isinstance(b, (_TimedPressRelease, _MatchSequence, StringMatcher)):
        return _AndStartEndMatcher(a, b)

    if isinstance(a, (_TimedPressRelease, _MatchSequence, StringMatcher)) and isinstance(b, Key):
        return _AndStartEndMatcher(a=b, b=a)

    if isinstance(a, _Or) and isinstance(b, (_TimedPressRelease, _MatchSequence, StringMatcher)):
        return _Or(*[_and(m, b) for m in a._matchers])

    if isinstance(a, (_TimedPressRelease, _MatchSequence, StringMatcher)) and isinstance(b, _Or):
        return _Or(*[_and(a, m) for m in b._matchers])

    if isinstance(a, _Or) and isinstance(b, _Or):
        return _Or(*[_and(m, b) for m in a._matchers])

    if isinstance(a, Key) and isinstance(b, _Or):
        return _Or(*[_and(m, a) for m in b._matchers])

    if isinstance(a, _AndKeys) and isinstance(b, _Or):
        return _Or(*[_and(m, a) for m in b._matchers])

    if isinstance(a, _Or) and isinstance(b, _AndKeys):
        return _Or(*[_and(m, b) for m in a._matchers])

    return _And(a, b)


def _is_hold_like(matcher: Matcher) -> bool:
    if isinstance(matcher, Hold):
        return True
    if isinstance(matcher, Loop):
        return True
    if isinstance(matcher, Throttle):
        return _is_hold_like(matcher.inner_matcher)
    return False


class Hold(Matcher):
    """
    Matches when the given keys are currently held down (live keyboard state),
    without consuming any events from the sequence.
    """

    def __init__(self, *keys: Key, only: bool = False):
        if len(keys) == 0:
            raise ValueError("Hold requires at least one key")
        for key in keys:
            if not isinstance(key, Key):
                raise ValueError(f"Hold expects Key instances, got {type(key)}")
        self._keys = keys
        self._only = only

    def match(self, events: Sequence[KeyEvent]) -> Result:
        keyboard_state = getattr(self.app, "_keyboard", None) if self.app else None
        if keyboard_state is None:
            return Result(stop_position=len(events))

        if self._only:
            if not keyboard_state.is_only_pressed(*self._keys):
                return Result(stop_position=len(events))
        elif not all(keyboard_state.is_pressed(k) for k in self._keys):
            return Result(stop_position=len(events))

        # Emit a synthetic event that carries the first held key code so downstream
        # consumers (e.g., direction detection) can still inspect match_end.code.
        synthetic = KeyEvent(key=self._keys[0].code, kind=KeyEventKind.PRESSED)
        synthetic.pressed_at = time.time()
        return Result(
            value=Match(start=synthetic, end=synthetic),
            stop_position=-1,
            matched_tokens=len(self._keys),
        )

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        raise ValueError("Cannot chain timed press-release with Hold")

    def debug(self) -> str:
        suffix = ", only=True" if self._only else ""
        return f"Hold({', '.join([k.debug() for k in self._keys])}{suffix})"

    def throttle(self, every_ms: int = 100) -> "Throttle":
        """Convenience to wrap this hold in a Throttle."""
        return Throttle(self, every_ms=every_ms)

    def collides_with(self, other: Matcher) -> bool:
        return _collides_hold(self, other)


class Throttle(Matcher):
    """
    Wraps another matcher and limits how often it can return a match.
    Useful for held combos that should only fire every N milliseconds.
    """

    def __init__(self, matcher: Matcher, every_ms: int = 100):
        if every_ms <= 0:
            raise ValueError("every_ms must be positive")
        self._matcher = matcher
        self._every_ms = every_ms
        self._last_fire = 0.0
        self.inner_matcher = matcher  # public alias for composed checks

    def match(self, events: Sequence[KeyEvent]) -> Result:
        result = self._matcher.match(events)
        if not result:
            return result

        now = time.time()
        if now - self._last_fire >= self._every_ms / 1000:
            self._last_fire = now
            return result

        return Result(stop_position=result.stop_position)

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        raise ValueError("Cannot chain timed press-release with Throttle")

    def debug(self) -> str:
        return f"Throttle({self._matcher.debug()}, every_ms={self._every_ms})"

    def collides_with(self, other: Matcher) -> bool:
        return _collides(self._matcher, other)


class Loop(Matcher):
    """
    Arms after a preceding matcher succeeds, then keeps matching the inner matcher
    while within a tolerance window. Optionally renews the deadline after each match.
    """

    def __init__(self, matcher: Matcher, tolerance_ms: int = 2000, renew: bool = True):
        if tolerance_ms <= 0:
            raise ValueError("tolerance_ms must be positive")
        self._matcher = matcher
        self._tolerance = tolerance_ms / 1000
        self._renew = renew
        self._armed = False
        self._deadline = 0.0
        self._prefixed = False

    @property
    def prefixed(self) -> bool:
        return self._prefixed

    def mark_prefixed(self) -> None:
        """Mark this loop as having a prefix in its sequence."""
        self._prefixed = True

    def is_active(self) -> bool:
        """Return True while armed and before the deadline expires."""
        return self._armed

    def reset(self) -> None:
        self._armed = False
        self._deadline = 0.0

    def _ensure_prefixed(self):
        if not self._prefixed:
            raise ValueError("Loop must follow a prefix and be the final matcher in a sequence.")

    def match(self, events: Sequence[KeyEvent]) -> Result:
        self._ensure_prefixed()
        if not events:
            return Result(stop_position=0)

        allowed_codes = _matcher_codes(self._matcher)

        # Expired before processing anything new
        if self._armed and events[0].pressed_at > self._deadline:
            self.reset()
            return Result(stop_position=len(events))

        # If armed and we see an unrelated key press, drop out of the loop so
        # other routes can take over.
        if self._armed and allowed_codes and events:
            last = events[-1]
            if last.kind == KeyEventKind.PRESSED and last.code not in allowed_codes:
                self.reset()
                return Result(stop_position=len(events))

        if not self._armed:
            result = self._matcher.match(events)
            if not result:
                return Result(stop_position=len(events))
            end_time = result.value.end.pressed_at
            self._armed = True
            self._deadline = end_time + self._tolerance
            return result

        result = self._matcher.match(events)
        if not result:
            # Stay armed if still within the window
            if events[-1].pressed_at > self._deadline:
                self.reset()
            return Result(stop_position=len(events))

        end_time = result.value.end.pressed_at
        if end_time > self._deadline:
            self.reset()
            return Result(stop_position=len(events))

        if self._renew:
            self._deadline = end_time + self._tolerance
        return result

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> "Matcher":
        raise ValueError("Loop must be the last matcher in a sequence")

    def __or__(self, other: Matcher) -> "Matcher":
        return _Or(self, other)

    def __and__(self, other: Matcher) -> "Matcher":
        return _and(self, other)

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        raise ValueError("Cannot chain timed press-release with Loop")

    def debug(self) -> str:
        matcher_desc = self._matcher.debug()
        tolerance_ms = int(self._tolerance * 1000)
        return f"Loop({matcher_desc}, tolerance_ms={tolerance_ms}, renew={self._renew})"

    def collides_with(self, other: Matcher) -> bool:
        return _collides_loop(self, other)


# ---- Collision helpers -------------------------------------------------------

def _time_windows_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return not (a_end < b_start or b_end < a_start)


def _collides_key(a: Key, b: Matcher) -> bool:
    if isinstance(b, Key):
        return a.code == b.code
    if isinstance(b, _TimedPressRelease):
        return a.code == b._key.code  # pylint: disable=protected-access
    if isinstance(b, Hold):
        return a.code in {k.code for k in b._keys}  # pylint: disable=protected-access
    if isinstance(b, _AndKeys):
        return a.code in {k.code for k in b._keys}  # pylint: disable=protected-access
    if isinstance(b, StringMatcher):
        chars = set(_string_chars(b))
        return _key_code_as_char(a.code) in chars
    if isinstance(b, _MatchSequence):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, (_Or, _And)):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, _AndStartEndMatcher):
        return _collides(a, b.a) or _collides(a, b.b)  # pylint: disable=protected-access
    if isinstance(b, Throttle):
        return _collides(a, b.inner_matcher)
    return False


def _collides_timed(a: _TimedPressRelease, b: Matcher) -> bool:
    if isinstance(b, _TimedPressRelease):
        if a._key.code != b._key.code:  # pylint: disable=protected-access
            return False
        return _time_windows_overlap(a._start, a._end, b._start, b._end)  # pylint: disable=protected-access
    if isinstance(b, Key):
        return a._key.code == b.code  # pylint: disable=protected-access
    if isinstance(b, Hold):
        return a._key.code in {k.code for k in b._keys}  # pylint: disable=protected-access
    if isinstance(b, _MatchSequence):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, (_Or, _And)):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, _AndKeys):
        return a._key.code in {k.code for k in b._keys}  # pylint: disable=protected-access
    if isinstance(b, _AndStartEndMatcher):
        return _collides(a, b.a) or _collides(a, b.b)  # pylint: disable=protected-access
    if isinstance(b, Throttle):
        return _collides(a, b.inner_matcher)
    if isinstance(b, StringMatcher):
        return _key_code_as_char(a._key.code) in set(_string_chars(b))  # pylint: disable=protected-access
    return False


def _collides_string(a: StringMatcher, b: Matcher) -> bool:
    chars = set(_string_chars(a))
    if isinstance(b, StringMatcher):
        return bool(chars.intersection(_string_chars(b)))
    if isinstance(b, Key):
        return _key_code_as_char(b.code) in chars
    if isinstance(b, _TimedPressRelease):
        return _key_code_as_char(b._key.code) in chars  # pylint: disable=protected-access
    if isinstance(b, _MatchSequence):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, (_Or, _And)):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, Throttle):
        return _collides(a, b.inner_matcher)
    return False


def _collides_hold(a: Hold, b: Matcher) -> bool:
    a_codes = {k.code for k in a._keys}  # pylint: disable=protected-access
    if isinstance(b, Hold):
        b_codes = {k.code for k in b._keys}  # pylint: disable=protected-access
        return bool(a_codes.intersection(b_codes))
    if isinstance(b, Key):
        return b.code in a_codes
    if isinstance(b, _TimedPressRelease):
        return b._key.code in a_codes  # pylint: disable=protected-access
    if isinstance(b, _AndKeys):
        return bool(a_codes.intersection({k.code for k in b._keys}))  # pylint: disable=protected-access
    if isinstance(b, _MatchSequence):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, (_Or, _And)):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, Throttle):
        return _collides(a, b.inner_matcher)
    return False


def _collides_loop(a: Loop, b: Matcher) -> bool:
    if isinstance(b, Loop):
        return _collides(a._matcher, b._matcher)  # pylint: disable=protected-access
    if isinstance(b, Throttle):
        return _collides(a._matcher, b.inner_matcher)  # pylint: disable=protected-access
    if isinstance(b, _MatchSequence):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, (_Or, _And)):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    return _collides(a._matcher, b)  # pylint: disable=protected-access


def _collides_chord(a: _AndKeys, b: Matcher) -> bool:
    a_codes = {k.code for k in a._keys}  # pylint: disable=protected-access
    if isinstance(b, _AndKeys):
        b_codes = {k.code for k in b._keys}  # pylint: disable=protected-access
        if a._only and b._only:  # pylint: disable=protected-access
            return a_codes == b_codes
        if a._only:  # pylint: disable=protected-access
            return b_codes.issubset(a_codes)
        if b._only:  # pylint: disable=protected-access
            return a_codes.issubset(b_codes)
        return bool(a_codes.intersection(b_codes))
    if isinstance(b, Key):
        return b.code in a_codes
    if isinstance(b, _TimedPressRelease):
        return b._key.code in a_codes  # pylint: disable=protected-access
    if isinstance(b, Hold):
        return bool(a_codes.intersection({k.code for k in b._keys}))  # pylint: disable=protected-access
    if isinstance(b, _MatchSequence):
        return _sequence_collides_with_chord(b, a)
    if isinstance(b, (_Or, _And)):
        return any(_collides(a, m) for m in b._matchers)  # pylint: disable=protected-access
    if isinstance(b, Throttle):
        return _collides(a, b.inner_matcher)
    return False


def _collides_sequence(a: _MatchSequence, b: Matcher) -> bool:
    if isinstance(b, _MatchSequence):
        sigs_a = matcher_signatures(a)
        sigs_b = matcher_signatures(b)
        return _signatures_overlap(sigs_a, sigs_b)
    if isinstance(b, Throttle):
        return _collides_sequence(a, b.inner_matcher)
    if isinstance(b, _AndKeys) and b._only:  # pylint: disable=protected-access
        seq_codes = _matcher_codes(a)
        b_codes = {k.code for k in b._keys}  # pylint: disable=protected-access
        return seq_codes.issubset(b_codes) if seq_codes else False
    # For non-sequence, collide if any component overlaps
    return any(_collides(m, b) for m in a._matchers)  # pylint: disable=protected-access


def _sequence_collides_with_chord(sequence: _MatchSequence, chord: _AndKeys) -> bool:
    seq_codes = _matcher_codes(sequence)
    chord_codes = {k.code for k in chord._keys}  # pylint: disable=protected-access
    if chord._only:  # pylint: disable=protected-access
        return seq_codes.issubset(chord_codes) if seq_codes else False
    return bool(seq_codes.intersection(chord_codes))


def _matcher_codes(matcher: Matcher) -> Set[str]:
    """Best-effort collection of key codes a matcher depends on."""
    if isinstance(matcher, Key):
        return {matcher.code}
    if isinstance(matcher, _TimedPressRelease):
        return {matcher._key.code}  # pylint: disable=protected-access
    if isinstance(matcher, Hold):
        return {k.code for k in matcher._keys}  # pylint: disable=protected-access
    if isinstance(matcher, _AndKeys):
        return {k.code for k in matcher._keys}  # pylint: disable=protected-access
    if isinstance(matcher, _MatchSequence):
        codes: Set[str] = set()
        for child in matcher._matchers:  # pylint: disable=protected-access
            codes.update(_matcher_codes(child))
        return codes
    if isinstance(matcher, _Or):
        codes: Set[str] = set()
        for child in matcher._matchers:  # pylint: disable=protected-access
            codes.update(_matcher_codes(child))
        return codes
    if isinstance(matcher, _AndStartEndMatcher):
        return _matcher_codes(matcher.a).union(_matcher_codes(matcher.b))  # pylint: disable=protected-access
    if isinstance(matcher, _And):
        codes: Set[str] = set()
        for child in matcher._matchers:  # pylint: disable=protected-access
            codes.update(_matcher_codes(child))
        return codes
    if isinstance(matcher, Throttle):
        return _matcher_codes(matcher.inner_matcher)
    if isinstance(matcher, Loop):
        return _matcher_codes(matcher._matcher)  # pylint: disable=protected-access
    return set()


def _signatures_overlap(sigs_a: List[tuple], sigs_b: List[tuple]) -> bool:
    def tokens_overlap(x, y) -> bool:
        if x == y:
            return True
        if x[0] == "press_release" and y[0] == "press_release" and x[1] == y[1]:
            return _time_windows_overlap(x[2], x[3], y[2], y[3])
        if x[0] == "press_release" and y[0] == "key" and x[1] == y[1]:
            return True
        if y[0] == "press_release" and x[0] == "key" and x[1] == y[1]:
            return True
        return False

    def is_contiguous_subseq(smaller, bigger):
        if len(smaller) > len(bigger):
            return False
        for i in range(len(bigger) - len(smaller) + 1):
            window = bigger[i:i + len(smaller)]
            if all(tokens_overlap(s, w) for s, w in zip(smaller, window)):
                return True
        return False

    for a in sigs_a:
        for b in sigs_b:
            if is_contiguous_subseq(a, b) or is_contiguous_subseq(b, a):
                return True
    return False


def _key_code_as_char(code: str) -> str:
    # repr(KeyCode.from_char('a')) -> "KeyCode.from_char('a')"
    # naive extraction of last single character between quotes
    if "'" in code:
        try:
            return code.split("'")[1]
        except IndexError:
            return ""
    return ""


def _string_chars(m: StringMatcher) -> List[str]:
    return [c if m._case else c.lower() for c in m._string]  # pylint: disable=protected-access


def _collides(a: Matcher, b: Matcher) -> bool:
    # Symmetric dispatch
    if isinstance(a, Key):
        return _collides_key(a, b)
    if isinstance(a, _TimedPressRelease):
        return _collides_timed(a, b)
    if isinstance(a, Hold):
        return _collides_hold(a, b)
    if isinstance(a, StringMatcher):
        return _collides_string(a, b)
    if isinstance(a, _MatchSequence):
        return _collides_sequence(a, b)
    if isinstance(a, _Or):
        return any(_collides(m, b) for m in a._matchers)  # pylint: disable=protected-access
    if isinstance(a, _And):
        return any(_collides(m, b) for m in a._matchers)  # pylint: disable=protected-access
    if isinstance(a, _AndKeys):
        return _collides_chord(a, b)
    if isinstance(a, _AndStartEndMatcher):
        return _collides(a.a, b) or _collides(a.b, b)  # pylint: disable=protected-access
    if isinstance(a, Throttle):
        return _collides(a.inner_matcher, b)
    if isinstance(a, Loop):
        return _collides_loop(a, b)
    # fallback: try reversing dispatch
    if a is not b:
        return _collides(b, a)  # pylint: disable=arguments-out-of-order
    return False


def collision_report(a: Matcher, b: Matcher) -> str:
    """Best-effort description of why two matchers collide."""
    # Timed overlaps
    if (
        isinstance(a, _TimedPressRelease)
        and isinstance(b, _TimedPressRelease)
        and a.key.code == b.key.code
    ):
        return (
            "Timed press-release on "
            f"{a.key.debug()} overlaps: [{a.start}, {a.end}] vs [{b.start}, {b.end}]"
        )
    # Key overlaps
    if isinstance(a, Key) and isinstance(b, Key) and a.code == b.code:
        return f"Same key: {a.debug()}"
    if isinstance(a, Hold) and isinstance(b, Hold):
        overlap = {k.code for k in a._keys}.intersection({k.code for k in b._keys})  # pylint: disable=protected-access
        return f"Holds share keys: {overlap}"
    if isinstance(a, Hold) and isinstance(b, Key) and any(k.code == b.code for k in a._keys):  # pylint: disable=protected-access
        return f"Key {b.debug()} is part of hold {a.debug()}"
    if isinstance(b, Hold) and isinstance(a, Key) and any(k.code == a.code for k in b._keys):  # pylint: disable=protected-access
        return f"Key {a.debug()} is part of hold {b.debug()}"
    if isinstance(a, StringMatcher) and isinstance(b, StringMatcher):
        overlap = set(_string_chars(a)).intersection(_string_chars(b))
        return f"Strings share chars: {sorted(overlap)}"
    if isinstance(a, StringMatcher) and isinstance(b, Key):
        return f"String contains key {b.debug()}"
    if isinstance(b, StringMatcher) and isinstance(a, Key):
        return f"String contains key {a.debug()}"

    # Sequence/Or/And: fallback to signatures to surface token overlap
    sigs_a = matcher_signatures(a)
    sigs_b = matcher_signatures(b)
    if sigs_a and sigs_b:
        for sig_a in sigs_a:
            for sig_b in sigs_b:
                common = [tok for tok in sig_a if tok in sig_b]
                if common:
                    return f"Common signature tokens: {common}"

    # Default
    return f"{a.debug()} overlaps with {b.debug()}"


# ---- Matcher introspection helpers -------------------------------------------------

def _product_concat(options: List[List[tuple]]) -> List[tuple]:
    """Cartesian product helper that flattens each combination."""
    combos = []
    for combo in product(*options):
        combos.append(tuple(chain.from_iterable(combo)))
    return combos


def matcher_signatures(matcher: Matcher) -> List[tuple]:
    """
    Return lightweight signature variants for a matcher.

    The signatures are best-effort, meant for detecting obvious collisions
    (e.g., overlapping sequences) and for estimating matcher specificity.
    When a matcher cannot be reduced to a signature, an empty list is returned.
    """
    def _sig(m: Matcher) -> List[tuple]:
        if isinstance(m, Key):
            return [(("key", m.code),)]

        if isinstance(m, _TimedPressRelease):
            timed = (("press_release", m._key.code, m._start, m._end),)  # pylint: disable=protected-access
            return [timed]

        if isinstance(m, Hold):
            codes = tuple(sorted(k.code for k in m._keys))  # pylint: disable=protected-access
            only = m._only  # pylint: disable=protected-access
            signatures = [(("hold_only" if only else "hold", codes),)]
            if not only:
                # Expand non-exclusive holds into individual key presses for collision checks
                signatures.append(tuple(("key", code) for code in codes))
            return signatures

        if isinstance(m, Throttle):
            return _sig(m.inner_matcher)

        if isinstance(m, Loop):
            return _sig(m._matcher)  # pylint: disable=protected-access

        if isinstance(m, StringMatcher):
            return [tuple(("char", c if m._case else c.lower()) for c in m._string)]  # pylint: disable=protected-access

        if isinstance(m, _MatchSequence):
            parts = [_sig(child) for child in m._matchers]  # pylint: disable=protected-access
            if any(len(p) == 0 for p in parts):
                return []
            return _product_concat(parts)

        if isinstance(m, _Or):
            variants: List[tuple] = []
            for child in m._matchers:  # pylint: disable=protected-access
                variants.extend(_sig(child))
            return variants

        if isinstance(m, _AndKeys):
            codes = tuple(sorted(k.code for k in m._keys))  # pylint: disable=protected-access
            token = "chord_only" if m._only else "chord"  # pylint: disable=protected-access
            return [((token, codes),)]

        if isinstance(m, _AndStartEndMatcher):
            left = _sig(m.a)  # pylint: disable=protected-access
            right = _sig(m.b)  # pylint: disable=protected-access
            combos = []
            for la in left:
                for rb in right:
                    combos.append(tuple(sorted(set(la + rb))))
            return combos

        if isinstance(m, _And):
            parts = [_sig(child) for child in m._matchers]  # pylint: disable=protected-access
            if any(len(p) == 0 for p in parts):
                return []
            combos = []
            for combo in _product_concat(parts):
                combos.append(tuple(sorted(set(combo))))
            return combos

        return []

    return _sig(matcher)


def matcher_specificity(matcher: Matcher) -> int:
    """
    Return an integer representing how specific a matcher is.

    We prefer longer sequences (more required steps) when ordering matchers.
    """
    signatures = matcher_signatures(matcher)
    if signatures:
        return max(len(sig) for sig in signatures)

    # Fallback approximations for matchers we cannot reduce
    if isinstance(matcher, Throttle):
        return matcher_specificity(matcher.inner_matcher)

    if isinstance(matcher, Hold):
        return len(matcher._keys)  # pylint: disable=protected-access

    if isinstance(matcher, _MatchSequence):
        return sum(matcher_specificity(m) for m in matcher._matchers)  # pylint: disable=protected-access

    if isinstance(matcher, _Or):
        return max(matcher_specificity(m) for m in matcher._matchers)  # pylint: disable=protected-access

    if isinstance(matcher, _And):
        return sum(matcher_specificity(m) for m in matcher._matchers)  # pylint: disable=protected-access

    return 1
