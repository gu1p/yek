
class Wait:
    def __init__(self, milli: int = 0, seconds: int = 0):
        if milli < 0 or seconds < 0:
            raise ValueError("Time cannot be negative")

        if milli == 0 and seconds == 0:
            raise ValueError("Time cannot be zero")

        self._milliseconds = milli
        self._seconds = seconds

    @property
    def milliseconds(self):
        return self._milliseconds + self._seconds * 1000