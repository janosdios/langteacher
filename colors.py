"""ANSI terminal colors, llama.cpp-style: dark blue for the user, yellow for the tutor."""
import os
import sys


def _supports_color():
    if os.environ.get("NO_COLOR") is not None:
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_ENABLED = _supports_color()

RESET = "\033[0m" if _ENABLED else ""
_USER = "\033[1;34m" if _ENABLED else ""   # dark blue
_TUTOR = "\033[33m" if _ENABLED else ""    # yellow


def user(text):
    return f"{_USER}{text}{RESET}"


def tutor(text):
    return f"{_TUTOR}{text}{RESET}"
