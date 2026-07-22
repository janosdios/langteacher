import os
import sys
import signal
import logging
import gettext
from datetime import datetime
from pathlib import Path
from typing import Optional
import colors
import languages
import settings
import stt_engine
import tts_engine
import llm_engine
import rag_engine
from stt_engine import record_speech

# Translates this module's own CLI-facing strings (--help banner, startup/
# setup messages, session prompts). Selected via UI_LANGUAGE; falls back to
# the system locale, then to the original English string if no catalog
# matches (fallback=True), so a missing/incomplete translation never crashes
# the CLI. Log messages are left in English, consistent with the other
# modules.
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

def _load_translation(language=None):
    return gettext.translation(
        "main",
        localedir=_LOCALE_DIR,
        languages=[language] if language else None,
        fallback=True,
    )

def set_ui_language(language):
    """Switch the catalog used by `_()`, and propagate to every module whose
    own CLI text main.py's flow can surface (their own `_()` was already
    resolved at import time from UI_LANGUAGE, so a --ui-lang override here
    needs to be pushed to them explicitly)."""
    global _translation, _
    _translation = _load_translation(language)
    _ = _translation.gettext
    languages.set_ui_language(language)
    stt_engine.set_ui_language(language)
    tts_engine.set_ui_language(language)
    llm_engine.set_ui_language(language)

_translation = _load_translation(os.environ.get("UI_LANGUAGE"))
_ = _translation.gettext

# RAG is optional: on by default, but the tutor works fine without a
# knowledge base or embedding server -- it just falls back to no context.
RAG_ENABLED = os.environ.get("RAG_ENABLED", "true").lower() not in ("0", "false", "no")

# How converse() listens for the student: "key" (push-to-talk, see
# stt_engine.PTT_KEY) or "vad" (voice activity detection, starts recording as
# soon as it hears speech). Kept as one constant so the startup message in
# main() can describe the right way to talk instead of a generic one.
RECORD_METHOD = "key"

# Which language profile to teach (see languages.py for the available names
# and what each bundles: STT/RAG code, OCR pack, tutor name, reference voice).
# Picked interactively at startup (see main()) via languages.select_language();
# these start unset and are filled in before anything else needs them.
LANGUAGE_NAME = None
LANG_PROFILE = None

# Language code the student is learning, shared between STT and RAG so
# retrieval only pulls chunks from books in the language actually being taught.
LANGUAGE = None

# Tesseract's OCR language pack, only relevant if a PDF gets (re-)ingested from
# this process. Uses tesseract's 3-letter codes (e.g. "deu"), not LANGUAGE's
# 2-letter one, so it's kept as its own setting in languages.py rather than derived.
OCR_LANG = None

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("langteacher")
if not logger.handlers:
    _handler = logging.FileHandler(LOGS_DIR / f"langteacher.{settings.HOSTNAME}.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s - line: %(lineno)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)

# Readable per-session transcripts (for the student to review, and to give
# the tutor a short recap of the previous session on the next run).
TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "transcripts"
RECAP_TURNS = 3  # prior exchanges carried into the next session's system prompt

SUMMARY_MARKER = "\nSummary:\n"

_transcript_path: Optional[Path] = None


def _shutdown_handler(sig, frame):
    logger.debug(f"Signal received: {sig}, {frame} ")
    print(_("\n\nEnding the session..."))
    sys.exit(0)


def _start_transcript():
    """Create this session's transcript file and record its path for
    subsequent _append_transcript calls."""
    global _transcript_path
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    _transcript_path = TRANSCRIPTS_DIR / f"{now.strftime('%Y-%m-%d_%H%M%S')}_{settings.HOSTNAME}_{LANGUAGE}_{llm_engine.TEACHER_LEVEL}.txt"
    header = (
        f"Session: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Machine: {settings.HOSTNAME}\n"
        f"Language: {LANGUAGE}\n"
        f"Level: {llm_engine.TEACHER_LEVEL}\n\n"
    )
    _transcript_path.write_text(header, encoding="utf-8")


def _append_transcript(user_text, tutor_text):
    """Append one exchange to the current session's transcript file."""
    if _transcript_path is None:
        return
    with _transcript_path.open("a", encoding="utf-8") as f:
        f.write(f"You: {user_text}\nTutor: {tutor_text}\n\n")


def _load_last_recap(turns=RECAP_TURNS):
    """Return a short recap of the most recently completed transcript: its
    LLM-written summary if one was generated at shutdown, otherwise its last
    few exchanges. Returns None if there's no previous transcript. Must be
    called before _start_transcript() creates the current session's file."""
    if not TRANSCRIPTS_DIR.exists():
        return None
    files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    if not files:
        return None

    header, _, body = files[-1].read_text(encoding="utf-8").partition("\n\n")
    body = body.strip()
    if not body:
        return None

    _, marker, summary = body.partition(SUMMARY_MARKER)
    if marker:
        return f"{header}\nSummary: {summary.strip()}"

    exchanges = [e.strip() for e in body.split("\n\n") if e.strip()]
    if not exchanges:
        return None
    recap_body = "\n".join(exchanges[-turns:])
    return f"{header}\n{recap_body}"


def _finalize_transcript():
    """On shutdown, ask the tutor LLM to summarize the finished session and
    append that summary to its transcript file, for the next session's
    recap. Silently does nothing if there's no transcript, nothing was said,
    or the LLM server is unreachable."""
    if _transcript_path is None:
        return
    _, _, body = _transcript_path.read_text(encoding="utf-8").partition("\n\n")
    if not body.strip():
        return

    summary = llm_engine.summarize_session(body.strip())
    if not summary:
        return
    # The file already ends in a blank line after the last exchange (see
    # _append_transcript), so this needs no leading newline of its own --
    # SUMMARY_MARKER's leading "\n" matches against that existing blank line.
    with _transcript_path.open("a", encoding="utf-8") as f:
        f.write(f"Summary:\n{summary}\n")


def _prompt_use_saved_settings(saved):
    """Show the previous session's saved picks and ask whether to keep them."""
    print(_("Found settings from your last session:"))
    print(_("  Language: {language}").format(language=saved['language'].capitalize()))
    print(_("  CEFR level: {level}").format(level=saved['cefr_level']))
    print(_("  Native language: {language}").format(language=saved['native_language']))
    print(_("  Mic device: {name}").format(name=stt_engine.get_mic_target_name(saved['mic_target'])))
    print(_("  Playback device: {name}").format(name=tts_engine.get_playback_target_name(saved['playback_target'])))
    answer = input(_("Keep these settings? [Y/n]: ")).strip().lower()
    return answer in ("", "y", "yes")


def _choose_settings():
    """Return this session's language/level/native-language/mic/playback
    picks, either reused from config.ini (if the user opts in) or gathered
    via the interactive pickers. saving the outcome back to config.ini
    whenever the pickers are used, so next run has something to offer."""
    saved = settings.load()
    if saved and saved["language"] not in languages.AVAILABLE_LANGUAGES:
        saved = None  # stale/invalid entry, e.g. from a removed language profile

    if saved and _prompt_use_saved_settings(saved):
        return saved

    print(_("Which language would you like to learn?"))
    language = languages.select_language()

    print(_("What's your CEFR level?"))
    cefr_level = languages.select_teacher_level()

    print(_("What's your native language?"))
    native_language = languages.select_native_language()

    mic_target = stt_engine.select_mic_target()
    playback_target = tts_engine.select_playback_target()

    settings.save(language, cefr_level, native_language, mic_target, playback_target)
    return {
        "language": language,
        "cefr_level": cefr_level,
        "native_language": native_language,
        "mic_target": mic_target,
        "playback_target": playback_target,
    }


def converse():
    """One listen -> think -> speak turn. Returns False if nothing was heard."""
    user_text = record_speech(method=RECORD_METHOD)
    if not user_text:
        print(_("Didn't catch any speech in that recording.\n"))
        return False

    print(colors.user(_("Heard: \"{text}\"").format(text=user_text)))
    logger.info(f"User: {user_text}")

    context_chunks = []
    if RAG_ENABLED:
        context_chunks = rag_engine.retrieve_context(user_text, language=LANGUAGE, level=llm_engine.TEACHER_LEVEL)

    reply_sentences = []
    for sentence in llm_engine.generate_reply_stream(user_text, context_chunks=context_chunks):
        print(colors.tutor(_("Tutor: {sentence}").format(sentence=sentence)))
        logger.info(f"Tutor: {sentence}")
        tts_engine.speak_text(sentence)
        reply_sentences.append(sentence)

    if not reply_sentences:
        print(_("(Tutor is unreachable, no reply generated)\n"))
        return True

    _append_transcript(user_text, " ".join(reply_sentences))
    return True


# noinspection PyBroadException
def main():
    global RAG_ENABLED, LANGUAGE_NAME, LANG_PROFILE, LANGUAGE, OCR_LANG
    args = sys.argv[1:]

    if "--ui-lang" in args:
        try:
            set_ui_language(args[args.index("--ui-lang") + 1])
        except IndexError:
            print(_("Usage: --ui-lang <language-code>"))

    if "--whisper-model" in args:
        try:
            stt_engine.set_whisper_model(args[args.index("--whisper-model") + 1])
        except IndexError:
            print(_("Usage: --whisper-model <model-name-or-path>"))

    if "--tts-engine" in args:
        try:
            tts_engine.set_tts_engine(args[args.index("--tts-engine") + 1])
        except IndexError:
            print(_("Usage: --tts-engine <omnivoice|piper|auto>"))

    if "--tutor-host" in args or "--tutor-port" in args:
        try:
            tutor_host = args[args.index("--tutor-host") + 1] if "--tutor-host" in args else llm_engine.LLAMACPP_HOST
            tutor_port = args[args.index("--tutor-port") + 1] if "--tutor-port" in args else None
        except IndexError:
            print(_("Usage: --tutor-host <ip-or-hostname> / --tutor-port <port>"))
        else:
            llm_engine.set_server_target(tutor_host, tutor_port)

    if "--rag-host" in args or "--rag-port" in args:
        try:
            rag_host = args[args.index("--rag-host") + 1] if "--rag-host" in args else rag_engine.EMBED_HOST
            rag_port = args[args.index("--rag-port") + 1] if "--rag-port" in args else None
        except IndexError:
            print(_("Usage: --rag-host <ip-or-hostname> / --rag-port <port>"))
        else:
            rag_engine.set_embed_server_target(rag_host, rag_port)

    if "--help" in args:
        print(_("LangTeacher - voice-based language tutor"))
        print(_("\nUsage: python3 main.py [--no-recap] [--whisper-model <model-name-or-path>] [--tts-engine <omnivoice|piper|auto>] [--tutor-host <ip-or-hostname>] [--tutor-port <port>] [--rag-host <ip-or-hostname>] [--rag-port <port>] [--ui-lang <language-code>]"))
        print(_("  --no-recap       Start with a clean slate, ignoring any previous session's recap"))
        print(_("  --whisper-model  Whisper model name or path to use for transcription (default: env WHISPER_MODEL or 'small'); pick according to platform/language, e.g. a smaller model on a Raspberry Pi or a language-specific fine-tune"))
        print(_("  --tts-engine     TTS backend to use: omnivoice (voice cloning/design), piper (lightweight, e.g. for a Raspberry Pi), or auto (default: env TTS_ENGINE or 'auto')"))
        print(_("  --tutor-host     llama.cpp server host for the tutor LLM (default: env LLAMACPP_HOST or 127.0.0.1)"))
        print(_("  --tutor-port     llama.cpp server port for the tutor LLM (default: env LLAMACPP_PORT or 8080)"))
        print(_("  --rag-host       llama.cpp server host for the RAG embedding model (default: env EMBED_HOST or 127.0.0.1)"))
        print(_("  --rag-port       llama.cpp server port for the RAG embedding model (default: env EMBED_PORT or 8081)"))
        print(_("  --ui-lang        Language for this CLI's own text, e.g. en/hu (default: env UI_LANGUAGE or system locale)"))
        return

    logger.info('LangTeacher started.')
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    picks = _choose_settings()

    LANGUAGE_NAME = picks["language"]
    LANG_PROFILE = languages.get_language(LANGUAGE_NAME)
    LANGUAGE = LANG_PROFILE["code"]
    OCR_LANG = LANG_PROFILE["ocr_code"]

    llm_engine.set_teacher_level(picks["cefr_level"])
    llm_engine.set_native_language(picks["native_language"])

    stt_engine.set_mic_target(int(picks["mic_target"]))
    stt_engine.set_lang_target(LANGUAGE)
    native_code = languages.native_language_code(picks["native_language"])
    stt_engine.set_native_lang_target(native_code or "")
    tts_engine.set_playback_target(int(picks["playback_target"]))
    tts_engine.set_ref_audio_target(LANG_PROFILE["ref_audio_target"])
    tts_engine.set_ref_text_target(LANG_PROFILE["ref_text_target"])
    tts_engine.set_piper_voice_target(LANG_PROFILE.get("piper_voice", ""))
    llm_engine.set_target_language(LANGUAGE_NAME.capitalize())
    llm_engine.set_tutor_name(LANG_PROFILE["tutor_name"])

    if not llm_engine.init_engine():
        print(_("Could not reach the tutor's language model at {host}:{port}.").format(
            host=llm_engine.LLAMACPP_HOST, port=llm_engine.LLAMACPP_PORT
        ))
        print(_("Check LLAMACPP_HOST/LLAMACPP_PORT and that llama-server is running."))
        return

    if "--no-recap" in args:
        logger.info("Starting without a recap (--no-recap).")
    else:
        recap = _load_last_recap()
        if recap:
            llm_engine.set_last_session_recap(recap)
    _start_transcript()

    rag_engine.set_ocr_lang(OCR_LANG)
    if RAG_ENABLED and not rag_engine.init_engine():
        print(_("RAG knowledge base unavailable, continuing without it (set RAG_ENABLED=false to silence this)."))
        RAG_ENABLED = False

    print(_("Loading speech recognition model..."))
    stt_engine.whisper_engine()
    print(_("Loading voice model..."))
    tts_engine.get_tts_model()

    if RECORD_METHOD == "key":
        print(_("\nLangTeacher ready. Hold {key} to talk (Ctrl+C to quit).\n").format(key=stt_engine.get_ptt_key_name()))
    else:
        print(_("\nLangTeacher ready. Speak whenever you like (Ctrl+C to quit).\n"))
    try:
        while True:
            converse()
    except Exception:
        logger.exception("Unexpected error in the conversation loop, ending the session.")
        print(_("\nUnexpected error, ending the session."))
    finally:
        # Runs on every exit path -- normal Ctrl+C (via _shutdown_handler's
        # sys.exit, which SystemExit skips right past `except Exception` to
        # reach here) as well as a crash from the loop above. so a turn
        # going wrong never silently drops the session summary/recap.
        _finalize_transcript()

if __name__ == "__main__":
    main()