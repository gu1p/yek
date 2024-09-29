import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Deque, Generic, TypeVar, List


class LimitedMap(ABC):
    def __init__(self):
        self.data = OrderedDict()
        self.lock = threading.Lock()

    @abstractmethod
    def _evict(self):
        pass

    def __setitem__(self, key, value):
        with self.lock:
            if key in self.data:
                del self.data[key]
            self.data[key] = (value, time.time())
            self._evict()

    def __getitem__(self, key):
        with self.lock:
            if key in self.data:
                value, _ = self.data[key]
                return value
            raise KeyError(key)

    def __delitem__(self, key):
        with self.lock:
            del self.data[key]

    def __contains__(self, key):
        with self.lock:
            return key in self.data

    def __len__(self):
        with self.lock:
            return len(self.data)

    def keys(self):
        with self.lock:
            return list(self.data.keys())

    def values(self):
        with self.lock:
            return [value for value, _ in self.data.values()]

    def items(self):
        with self.lock:
            return [(key, value) for key, (value, _) in self.data.items()]

    def get(self, key, default=None):
        with self.lock:
            try:
                return self[key]
            except KeyError:
                return default

    def clear(self):
        with self.lock:
            self.data.clear()


class CountLimitedMap(LimitedMap):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def _evict(self):
        while len(self.data) > self.limit:
            self.data.popitem(last=False)


class TimeLimitedMap(LimitedMap):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def _evict(self):
        current_time = time.time()
        keys_to_remove = [
            key
            for key, (_, timestamp) in self.data.items()
            if current_time - timestamp > self.limit
        ]
        for key in keys_to_remove:
            del self.data[key]


T = TypeVar("T")
class ThreadSafeBuffer(Generic[T]):
    def __init__(self, max_len: int):
        self._buffer: Deque[T] = deque(maxlen=max_len)
        self.lock = threading.Lock()
        self.last_added = 0

    def append(self, key: T):
        with self.lock:
            self._buffer.append(key)

    def get(self) -> List[T]:
        with self.lock:
            return list(self._buffer)

    def get_since(self, since: float) -> List[T]:
        added = []
        with self.lock:
            for item in self._buffer:
                if item.pressed_at >= since:
                    added.append(item)
        return added
