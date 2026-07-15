"""Per-language tutor profiles.

Each entry bundles everything that changes when the student switches which
language they're learning: the STT/RAG language code, Tesseract's OCR
language pack, and the tutor persona's name + voice-cloning reference audio
and its transcript (see samples/, tts_engine.set_ref_audio_target /
set_ref_text_target), plus a Piper voice id (see piper_voices/,
tts_engine.set_piper_voice_target) for when the Piper backend is active
instead. ref_text_target is declared here rather than left to tts_engine's
same-named-.txt-sidecar auto-detection, so the transcript that gets used is
explicit and doesn't silently depend on samples/ file naming. Keeping all of
this grouped also avoids the mismatches you get from setting LANGUAGE,
OCR_LANG, REF_AUDIO_TARGET etc. as separate, independently overridable env
vars.
"""

import os
import gettext
from functools import lru_cache
from pathlib import Path
from iso639 import Lang, iter_langs

# Translates this module's interactive-picker strings (prompts, error
# messages). Selected via UI_LANGUAGE; falls back to the system locale, then
# to the original English string if no catalog matches (fallback=True), so a
# missing/incomplete translation never crashes the picker. Left untranslated:
# the printed menu entries themselves (language/level names), since those are
# data, not fixed UI text.
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

def _load_translation(language=None):
    return gettext.translation(
        "languages",
        localedir=_LOCALE_DIR,
        languages=[language] if language else None,
        fallback=True,
    )

def set_ui_language(language):
    """Switch the catalog used by `_()` (e.g. from a caller's --ui-lang)."""
    global _translation, _
    _translation = _load_translation(language)
    _ = _translation.gettext

_translation = _load_translation(os.environ.get("UI_LANGUAGE"))
_ = _translation.gettext

LANGUAGES = {
    "german": {
        "code": "de",
        "ocr_code": "deu",
        "tutor_name": "Anna",
        "ref_audio_target": "anna_de",
        "ref_text_target": (
            "Ich heiße Anna und ich wohne in der Nähe von Hamburg. Ich bin in "
            "einer kleinen Stadt nahe Hannover aufgewachsen und dann immer mehr "
            "in den Norden gewandert."
        ),
        "piper_voice": "de_DE-kerstin-low",
    },
    "french": {
        "code": "fr",
        "ocr_code": "fra",
        "tutor_name": "Alice",
        "ref_audio_target": "alice_fr",
        "ref_text_target": (
            "Je m'appelle Alice et j'habite près de Paris. J'ai grandi dans une "
            "petite ville près de Marseille, puis je me suis installée de plus "
            "en plus au nord."
        ),
        "piper_voice": "fr_FR-siwis-medium",
    },
    "spanish": {
        "code": "es",
        "ocr_code": "spa",
        "tutor_name": "Maria",
        "ref_audio_target": "maria_es",
        "ref_text_target": (
            "Me llamo María y vivo cerca de Madrid. Crecí en una pequeña ciudad "
            "cerca de Sevilla y luego me fui mudando cada vez más al norte."
        ),
        "piper_voice": "es_ES-davefx-medium",
    },
    "hungarian": {
        "code": "hu",
        "ocr_code": "hun",
        "tutor_name": "Katalin",
        "ref_audio_target": "katalin_hu",
        "ref_text_target": (
            "A nevem Katalin, és Budapest közelében lakom. Egy Szegedhez közeli "
            "kisvárosban nőttem fel, majd egyre északabbra költöztem."
        ),
        "piper_voice": "hu_HU-anna-medium",
    },
    "english": {
        "code": "en",
        "ocr_code": "eng",
        "tutor_name": "Lucy",
        "ref_audio_target": "lucy_en",
        "ref_text_target": (
            "My name is Lucy, and I live close to London. I grew up in a small "
            "city close to Portsmouth and then moved further and further to "
            "the north."
        ),
        "piper_voice": "en_US-amy-medium",
    },
}

AVAILABLE_LANGUAGES = list(LANGUAGES.keys())

DEFAULT_LANGUAGE = "german"

# Student-level defaults: independent of which language is being learned, so
# they live here as flat constants rather than inside each LANGUAGES entry.
NATIVE_LANGUAGE = "English"

# Shortlist for select_native_language()'s numbered menu -- not the full set
# of possible native languages (see _iso639_name_lookup() below, backed by
# the iso639-lang package, for validating a free-typed "Other" language),
# just the ones common enough to deserve a one-keypress option before "Other".
COMMON_NATIVE_LANGUAGES = [
    "English", "Spanish", "German", "French", "Portuguese",
    "Chinese", "Japanese", "Hindi",
]

CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

TEACHER_LEVEL = "B1"  # CEFR scale


def native_language_code(name):
    """Return the ISO 639-1 code (e.g. "en") Whisper expects for a native
    language name as returned by select_native_language(), or None if that
    language has no two-letter code (rare -- some iso639 entries only have
    a 3-letter code, e.g. some COMMON_NATIVE_LANGUAGES's "Other" picks)."""
    try:
        return Lang(name).pt1 or None
    except Exception:
        return None


def get_language(name):
    """Return the profile dict for `name` (e.g. "german"), raising a clear
    error if it isn't one of the configured AVAILABLE_LANGUAGES."""
    try:
        return LANGUAGES[name]
    except KeyError:
        raise ValueError(
            f"Unknown language '{name}'. Available: {', '.join(AVAILABLE_LANGUAGES)}"
        )


def select_language():
    """List AVAILABLE_LANGUAGES and let the user pick one interactively,
    returning its key (e.g. "german"). Mirrors stt_engine.select_mic_target."""
    for i, name in enumerate(AVAILABLE_LANGUAGES):
        print(f"{i}: {name.capitalize()}")

    while True:
        try:
            selected = int(input(_('Please select a language: ')))
        except ValueError:
            print(_("Error! Please enter a number!"))
            continue
        if 0 <= selected < len(AVAILABLE_LANGUAGES):
            return AVAILABLE_LANGUAGES[selected]
        print(_("Error! Please select a valid language!"))


def select_teacher_level():
    """List CEFR_LEVELS and let the user pick one interactively, returning
    it (e.g. "B1"). Mirrors select_language / stt_engine.select_mic_target."""
    for i, level in enumerate(CEFR_LEVELS):
        print(f"{i}: {level}")

    while True:
        try:
            selected = int(input(_('Please select a CEFR level: ')))
        except ValueError:
            print(_("Error! Please enter a number!"))
            continue
        if 0 <= selected < len(CEFR_LEVELS):
            return CEFR_LEVELS[selected]
        print(_("Error! Please select a valid level!"))


def select_native_language():
    """List COMMON_NATIVE_LANGUAGES plus a final "Other" option. Picking
    "Other" prompts for free text, validated against real ISO 639 language
    names (case-insensitive) so a typo doesn't silently become the student's
    native language. Returns the canonical language name either way."""
    options = COMMON_NATIVE_LANGUAGES + ["Other"]
    for i, name in enumerate(options):
        print(f"{i}: {name}")

    while True:
        try:
            selected = int(input(_('Please select your native language: ')))
        except ValueError:
            print(_("Error! Please enter a number!"))
            continue
        if 0 <= selected < len(options) - 1:
            return options[selected]
        if selected == len(options) - 1:
            return _prompt_other_native_language()
        print(_("Error! Please select a valid option!"))


@lru_cache(maxsize=1)
def _iso639_name_lookup():
    """Lowercase name -> canonical name, for every language iso639-lang
    knows about. is_language()/Lang() are case-sensitive and str.title()
    doesn't reliably reconstruct ISO 639 names (e.g. "Afro-Asiatic languages"),
    so this is built once and matched against directly instead."""
    return {lang.name.lower(): lang.name for lang in iter_langs()}


def _prompt_other_native_language():
    lookup = _iso639_name_lookup()
    while True:
        typed = input(_('Please type your native language: ')).strip()
        match = lookup.get(typed.lower())
        if match:
            return match
        print(_("Error! '{typed}' isn't a recognized language name. Please try again.").format(typed=typed))
