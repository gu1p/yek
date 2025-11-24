"""Keyboard matchers and combinators."""

# pylint: disable=missing-class-docstring,missing-function-docstring

import abc
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

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
]


@dataclass
class Match:
    start: KeyEvent
    end: KeyEvent


@dataclass
class Result:
    stop_position: int
    value: Optional[Match] = None

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
                return Result(value=Match(start=event, end=event), stop_position=n)
        return Result(stop_position=len(events))

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
                return Result(stop_position=i)

        return Result(
            value=Match(
                start=events[matched_indexes[0]],
                end=events[matched_indexes[-1]]
            ),
            stop_position=i
        )

    def __matmul__(self, other: Union[float, Wait]) -> Matcher:
        raise ValueError("Cannot chain timed press-release with a string matcher")

    def __truediv__(self, other: Union[Matcher, Tuple[int, int]]) -> Matcher:
        return _MatchSequence(self, other)

    def __or__(self, other: Matcher) -> Matcher:
        return _Or(self, other)

    def __and__(self, other: Matcher) -> Matcher:
        return _and(self, other)


class _TimedPressRelease(Matcher):
    def __init__(self, key: "Key", start: float, end: float):
        self._key = key
        self._start = start
        self._end = end

    def match(self, events: Sequence[KeyEvent]) -> Result:
        for i, event in enumerate(events):
            if event.code == self._key.code and event.kind == KeyEventKind.PRESSED:
                for j, event2 in enumerate(events[i + 1:]):
                    if event2.code == self._key.code and event2.kind == KeyEventKind.RELEASED:
                        if self._start <= event2.pressed_at - event.pressed_at <= self._end:
                            return Result(value=Match(start=event, end=event2), stop_position=i + j)
        return Result(stop_position=len(events))

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


class _MatchSequence(Matcher):
    def __init__(self, *matchers: Matcher):
        self._matchers = []
        for matcher in matchers:
            if isinstance(matcher, _MatchSequence):
                self._matchers.extend(matcher._matchers)
            else:
                self._matchers.append(matcher)

    def match(self, events: Sequence[KeyEvent]) -> Result:
        position = 0
        matches = []
        for matcher in self._matchers:
            result = matcher.match(events[position:])
            if not result:
                return Result(stop_position=len(events))

            matches.append(result.value)
            position = result.stop_position + 1

        return Result(
            value=Match(start=matches[0].start, end=matches[-1].end),
            stop_position=position,
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


class _And(Matcher):
    def __init__(self, *matchers: Matcher):
        self._matchers = matchers

    def match(self, events: Sequence[KeyEvent]) -> Result:
        matches = []
        stop_position = 0
        for matcher in self._matchers:
            match = matcher.match(events)
            if not match:
                return Result(stop_position=len(events))

            stop_position = max(stop_position, match.stop_position)
            matches.append(match.value)
        return Result(
            value=Match(start=matches[0].start, end=matches[-1].end),
            stop_position=stop_position,
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


class _AndKeys(Matcher):
    def __init__(self, *keys: Key):
        for k in keys:
            if not isinstance(k, Key):
                raise ValueError(f"Expected Key instance, got {type(k)}")

        self._keys = keys

    def match(self, events: Sequence[KeyEvent]) -> Result:
        intervals = []
        min_stop, max_stop = math.inf, 0
        for key in self._keys:
            match = key.match(events)
            min_stop = min(min_stop, match.stop_position)
            max_stop = max(max_stop, match.stop_position)

            if not match:
                return Result(stop_position=len(events))

            press_event = match.value.start
            release_event = press_event.get_key_next_event(events)

            intervals.append((press_event, release_event))

        intersection = get_common_intersection(intervals)

        if intersection:
            return Result(value=Match(
                start=intersection[0],
                end=intersection[1]
            ), stop_position=max_stop)

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
        return f"AndKeys({', '.join([k.debug() for k in self._keys])})"


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
                stop_position=stop,
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
        return _AndKeys(*a._keys, b)

    if isinstance(a, Key) and isinstance(b, _AndKeys):
        return _AndKeys(*b._keys, a)

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
