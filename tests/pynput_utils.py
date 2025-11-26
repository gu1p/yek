"""Shared helpers for optional pynput dependency in tests."""

import unittest

try:
    import pynput  # type: ignore
except ImportError:
    pynput = None


def require_pynput():
    """Return pynput module or skip tests if it is unavailable."""
    if pynput is None:
        raise unittest.SkipTest("pynput not installed")
    return pynput
