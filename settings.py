"""Persisted user picks -- language to learn, CEFR level, native language,
mic/playback device targets -- stored in a per-machine config.<hostname>.ini
next to this file. One file per machine because mic/playback are sounddevice
indices that only make sense on the machine that picked them; this project
directory is synced across several machines (see README), so a shared
config.ini would leak one machine's device indices into another's.

main() reads this at startup and offers to reuse the previous session's
picks instead of running the interactive pickers again; it writes back
here whenever the pickers are used instead, so the "keep these settings?"
prompt has something to offer on the following run.
"""

import configparser
import platform
import re
from pathlib import Path

HOSTNAME = re.sub(r"[^A-Za-z0-9._-]", "_", platform.node() or "default")
SETTINGS_PATH = Path(__file__).resolve().parent / f"config.{HOSTNAME}.ini"

_SECTION = "langteacher"

_KEYS = ("language", "cefr_level", "native_language", "mic_target", "playback_target")


def load():
    """Return the saved settings as a dict of strings, or None if config.ini
    doesn't exist yet or is missing anything main() needs."""
    if not SETTINGS_PATH.is_file():
        return None
    parser = configparser.ConfigParser()
    parser.read(SETTINGS_PATH)
    if _SECTION not in parser:
        return None
    section = parser[_SECTION]
    if not all(key in section for key in _KEYS):
        return None
    return {key: section[key] for key in _KEYS}


def save(language, cefr_level, native_language, mic_target, playback_target):
    """Write the given settings to config.ini's [langteacher] section,
    creating the file if it doesn't exist yet."""
    parser = configparser.ConfigParser()
    parser[_SECTION] = {
        "language": language,
        "cefr_level": cefr_level,
        "native_language": native_language,
        "mic_target": str(mic_target),
        "playback_target": str(playback_target),
    }
    with SETTINGS_PATH.open("w", encoding="utf-8") as f:
        parser.write(f)
