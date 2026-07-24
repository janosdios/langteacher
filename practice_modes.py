"""Practice-mode profiles for the tutor's system prompt.

Each entry contributes zero or more extra instruction lines to
llm_engine.build_system_prompt() on top of the always-on core prompt (persona,
CEFR level, correction style, native-language fallback) -- e.g. quizzing
vocabulary or running translation drills. Picked interactively at the start of
every session (see main.py's _choose_settings()); never persisted to
config.<hostname>.ini, so it's asked fresh every run.

The "custom" mode doesn't take typed free text -- its goal comes from a .md
file the student drops in custom_goals/ (see CUSTOM_GOALS_DIR below), kept out
of git (see .gitignore) so pulling an app update never touches a student's own
goal files.
"""

import os
import gettext
from pathlib import Path

# Translates this module's interactive-picker strings (prompts, error
# messages), same pattern as languages.py. Selected via UI_LANGUAGE; falls
# back to the system locale, then to the original English string if no
# catalog matches (fallback=True).
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

def _load_translation(language=None):
    return gettext.translation(
        "practice_modes",
        localedir=_LOCALE_DIR,
        languages=[language] if language else None,
        fallback=True,
    )

def set_ui_language(language):
    """Switch the catalog used by `_()` (e.g. from main.py's --ui-lang)."""
    global _translation, _
    _translation = _load_translation(language)
    _ = _translation.gettext

_translation = _load_translation(os.environ.get("UI_LANGUAGE"))
_ = _translation.gettext

# prompt_lines are plain {placeholder} strings, not f-strings: they're
# .format()-ed by llm_engine.build_system_prompt() at call time, since
# NATIVE_LANGUAGE/TARGET_LANGUAGE/the custom goal text can all change after
# this module is imported.
PRACTICE_MODES = {
    "conversation": {
        "prompt_lines": [
            "At the very start of this session (your first message), ask "
            "the student what topic they would like to talk about today, "
            "then follow their lead once they answer.",
        ],
    },
    "roleplay": {
        "prompt_lines": [
            "Your goal in this session is a role-play between two people "
            "(e.g. a customer and a shopkeeper, a waiter and a guest, two "
            "strangers meeting). At the very start of this session (your "
            "first message), ask the student what scenario they'd like to "
            "role-play and which of the two roles they want to play; you "
            "take the other role.",
            "Once the roles are set, stay fully in character as your role "
            "for the rest of the session -- speak and react the way that "
            "person would, and don't break character to comment on the "
            "exercise itself. Corrections still apply: weave them "
            "naturally into your in-character reply instead of stepping "
            "outside the scene to give them.",
            "When you first take on your role, introduce yourself using "
            "both your name and that role together, so the roleplay stays "
            "connected to your tutor persona instead of dropping it "
            "entirely.",
        ],
    },
    "vocab": {
        "prompt_lines": [
            "Your goal in this session is to quiz the student on vocabulary. "
            "At the very start of this session, your first message must "
            "already be the first vocabulary question -- do not greet the "
            "student or ask what they want to do first. "
            "The word or phrase being tested must be given in {native_language}, "
            "because the student's job is to produce the {target_language} "
            "translation. For example, ask the equivalent of \"How do you say "
            "'X' in {target_language}?\" with X in {native_language} -- do NOT "
            "quote the {target_language} word and ask what it is 'in "
            "{target_language}', since that already states the answer.",
            "Keep quizzing one word after another without pausing to ask "
            "whether the student wants to continue. Only after quizzing "
            "somewhere between 15 and 20 words should you check whether "
            "they'd like to keep practicing vocabulary or move on to "
            "something else.",
        ],
    },
    "translation": {
        "prompt_lines": [
            "Your goal in this session is translation practice: give the "
            "student a short sentence or phrase in {native_language} and ask "
            "them to translate it into {target_language}. Wait for their "
            "attempt, correct it following the correction rules above, then "
            "give the next sentence. At the very start of this session, "
            "your first message must already contain the first sentence to "
            "translate -- do not greet the student or ask what they want to "
            "do first.",
            "Keep giving one sentence after another without pausing to ask "
            "whether the student wants to continue. Only after somewhere "
            "between 8 and 10 sentences should you check whether they'd "
            "like to keep practicing translation or move on to something "
            "else.",
        ],
    },
    "custom": {
        "prompt_lines": [
            # Block-style rather than inline-quoted: custom goal files range
            # from a one-line sentence to a multi-paragraph drill with a
            # numbered instruction list (see custom_goals/), and wrapping the
            # latter in "..." after a colon reads awkwardly. A blank line
            # before {custom_goal} works cleanly for both.
            "The student has set this specific goal for this session -- "
            "prioritize it while still following the rules above:\n\n"
            "{custom_goal}",
        ],
    },
}

AVAILABLE_PRACTICE_MODES = list(PRACTICE_MODES.keys())

DEFAULT_PRACTICE_MODE = "conversation"


def get_practice_mode(key):
    """Return the mode dict for `key` (e.g. "vocab"), raising a clear error if
    it isn't one of the configured AVAILABLE_PRACTICE_MODES."""
    try:
        return PRACTICE_MODES[key]
    except KeyError:
        raise ValueError(
            f"Unknown practice mode '{key}'. Available: {', '.join(AVAILABLE_PRACTICE_MODES)}"
        )


def mode_label(key):
    """Translated display label for `key`. Resolved per-call (not baked into
    PRACTICE_MODES at module-load time) so it reflects whatever UI language is
    active when it's actually printed, including after a --ui-lang switch."""
    labels = {
        "conversation": _("Free conversation"),
        "roleplay": _("Role-play"),
        "vocab": _("Vocabulary quiz"),
        "translation": _("Translation practice"),
        "custom": _("Custom goal"),
    }
    return labels[key]


def select_practice_mode():
    """List AVAILABLE_PRACTICE_MODES and let the user pick one interactively,
    returning its key (e.g. "vocab"). Mirrors languages.select_language."""
    keys = AVAILABLE_PRACTICE_MODES
    for i, key in enumerate(keys):
        print(f"{i}: {mode_label(key)}")

    while True:
        try:
            selected = int(input(_('Please select a practice mode: ')))
        except ValueError:
            print(_("Error! Please enter a number!"))
            continue
        if 0 <= selected < len(keys):
            return keys[selected]
        print(_("Error! Please select a valid practice mode!"))


# ===== Custom goal files =====

# Student-authored .md files describing a custom session goal. Kept as a
# separate directory (gitignored, see .gitignore) rather than typed free text
# saved into config.ini, so a student's own goal files never conflict with or
# get overwritten by an app update.
CUSTOM_GOALS_DIR = Path(__file__).resolve().parent / "custom_goals"

# How many of the most recently modified .md files to offer in the picker --
# keeps the menu on a single screen and selectable with one keypress (0-9).
MAX_CUSTOM_GOAL_LISTING = 5


def list_custom_goal_files():
    """Return up to MAX_CUSTOM_GOAL_LISTING .md files from CUSTOM_GOALS_DIR,
    most recently modified first. Empty list if the directory doesn't exist
    or has no .md files yet."""
    if not CUSTOM_GOALS_DIR.is_dir():
        return []
    files = sorted(
        CUSTOM_GOALS_DIR.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[:MAX_CUSTOM_GOAL_LISTING]


def select_custom_goal_file():
    """List the most recent custom goal .md files and let the user pick one
    by number, mirroring select_practice_mode(). Returns the chosen Path, or
    None if no .md files are available yet."""
    files = list_custom_goal_files()
    if not files:
        print(_(
            "No custom goal files found in custom_goals/. Add a .md file "
            "there (see custom_goals/put_custom_goal_files_here.txt) and "
            "restart."
        ))
        return None

    for i, path in enumerate(files):
        print(f"{i}: {path.stem.replace('_', ' ')}")

    while True:
        try:
            selected = int(input(_('Please select a custom goal file: ')))
        except ValueError:
            print(_("Error! Please enter a number!"))
            continue
        if 0 <= selected < len(files):
            return files[selected]
        print(_("Error! Please select a valid file!"))


def read_custom_goal(path):
    """Return the full text of a custom goal file, for substitution into
    PRACTICE_MODES["custom"]'s {custom_goal} placeholder."""
    return path.read_text(encoding="utf-8").strip()


def resolve_custom_goal_file(name):
    """Resolve a --custom-goal CLI value (bare filename or with a .md
    extension already) to a Path inside CUSTOM_GOALS_DIR. Returns None if
    `name` is falsy or no such file exists, so callers can fail fast with a
    clear message instead of guessing why a file wasn't found."""
    if not name:
        return None
    filename = name if name.endswith(".md") else f"{name}.md"
    path = CUSTOM_GOALS_DIR / filename
    return path if path.is_file() else None
