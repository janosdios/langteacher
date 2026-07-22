import os, sys, json
import logging
import gettext
import requests
import colors
import languages
import settings
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Translates the CLI-facing strings in main() (--help banner, usage errors,
# startup/goodbye messages). Selected via UI_LANGUAGE; falls back to the
# system locale, then to the original English string if no catalog matches
# (fallback=True), so a missing/incomplete translation never crashes the CLI.
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

def _load_translation(language=None):
    return gettext.translation(
        "llm_engine",
        localedir=_LOCALE_DIR,
        languages=[language] if language else None,
        fallback=True,
    )

def set_ui_language(language):
    """Switch the catalog used by `_()` (e.g. from --ui-lang or main.py)."""
    global _translation, _
    _translation = _load_translation(language)
    _ = _translation.gettext

_translation = _load_translation(os.environ.get("UI_LANGUAGE"))
_ = _translation.gettext

# Named logger so consuming projects can see what's happening on import
# without any setup, but can also reconfigure/silence it independently of
# their own logging (e.g. logging.getLogger("llm_engine").setLevel(...)),
# since it doesn't propagate to the root logger.
logger = logging.getLogger("llm_engine")
if not logger.handlers:
    _handler = logging.FileHandler(LOGS_DIR / f"llm_engine.{settings.HOSTNAME}.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [llm_engine] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)


# ===== Server connection =====

# llama.cpp server (llama-server) reachable somewhere on the local network
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "127.0.0.1")
LLAMACPP_PORT = os.environ.get("LLAMACPP_PORT", "8080")

def set_server_target(host, port=None):
    """Override LLAMACPP_HOST/PORT at runtime (e.g. from an interactive picker in another module)."""
    global LLAMACPP_HOST, LLAMACPP_PORT
    LLAMACPP_HOST = host
    if port is not None:
        LLAMACPP_PORT = str(port)

def _base_url():
    # noinspection HttpUrlsUsage
    return f"http://{LLAMACPP_HOST}:{LLAMACPP_PORT}"

REQUEST_TIMEOUT = float(os.environ.get("LLAMACPP_TIMEOUT", 60))
CONNECT_TIMEOUT = 5

# Sampling / generation parameters. Kept modest so the tutor stays consistent
# and on-topic rather than wandering (this is a teacher persona, not a
# creative-writing one).
TEMPERATURE = 0.6
TOP_P = 0.95
REPEAT_PENALTY = 1.05
MAX_TOKENS = 200

# Reuse the KV cache for the shared prefix (system prompt + prior turns)
# between requests. Matched with --parallel 1 on the server, this makes
# each turn only pay for the new tokens instead of reprocessing the whole
# conversation, which is most of what "smooth" multi-turn latency comes down to.
CACHE_PROMPT = True

# Session-summary generation (see summarize_session). Kept short and on a
# tight timeout since this runs synchronously on app shutdown.
SUMMARY_TEMPERATURE = 0.1
SUMMARY_MAX_TOKENS = 150
SUMMARY_TIMEOUT = float(os.environ.get("LLAMACPP_SUMMARY_TIMEOUT", 20))


# ===== Teacher persona =====

# Defaults come from languages.py's DEFAULT_LANGUAGE profile, so a standalone
# run of this module (no main.py) still gets a language/tutor-name pairing
# that matches the voice tts_engine defaults to, instead of two independently
# hardcoded values that can drift apart.
_DEFAULT_PROFILE = languages.get_language(languages.DEFAULT_LANGUAGE)

TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE", languages.DEFAULT_LANGUAGE.capitalize())
NATIVE_LANGUAGE = os.environ.get("NATIVE_LANGUAGE", languages.NATIVE_LANGUAGE)
TEACHER_LEVEL = os.environ.get("TEACHER_LEVEL", languages.TEACHER_LEVEL)  # CEFR scale

# Fixed so the tutor answers "what's your name?" the same way every time --
# left unset, the model invents a different name each time it's asked, which
# breaks the illusion of a consistent tutor across sessions.
TUTOR_NAME = os.environ.get("TUTOR_NAME", _DEFAULT_PROFILE["tutor_name"])

# Short recap of the previous session's transcript, set once at startup (see
# main.py) so the tutor has continuity across restarts without carrying the
# full prior conversation in every request.
LAST_SESSION_RECAP: Optional[str] = None

def set_target_language(language):
    global TARGET_LANGUAGE
    TARGET_LANGUAGE = language

def set_native_language(language):
    global NATIVE_LANGUAGE
    NATIVE_LANGUAGE = language

def set_teacher_level(level):
    global TEACHER_LEVEL
    TEACHER_LEVEL = level

def set_tutor_name(name):
    global TUTOR_NAME
    TUTOR_NAME = name

def set_last_session_recap(recap: Optional[str]):
    global LAST_SESSION_RECAP
    LAST_SESSION_RECAP = recap

def build_system_prompt(context_chunks=None):
    """Assemble the system prompt from the current persona settings and, once
    RAG is wired in, any retrieved lesson excerpts passed via `context_chunks`."""
    lines = [
        f"You are a friendly, patient {TARGET_LANGUAGE} language tutor having a spoken conversation with a student.",
        f"Your name is {TUTOR_NAME}. If asked your name, always answer {TUTOR_NAME}.",
        f"The student's approximate level is {TEACHER_LEVEL} on the CEFR scale.",
        f"Use {TARGET_LANGUAGE} as the default language of the conversation.",
        f"Use vocabulary, sentence length, and grammar appropriate for "
        f"CEFR level {TEACHER_LEVEL}.",
        f"Correct at most one mistake per turn.",
        f"When correcting a mistake, naturally restate the student's phrase "
        f"in its corrected form, then continue the conversation.",
        f"Keep replies short and conversational (1-3 sentences) since they will be read aloud.",
        f"Ask only one question at a time.",
    ]
    if NATIVE_LANGUAGE:
        lines.append(
            f"IMPORTANT: If the student writes or speaks in {NATIVE_LANGUAGE}, "
            f"asks 'what does X mean', or says they do not understand, "
            f"you MUST first explain briefly in {NATIVE_LANGUAGE} (1-2 sentences), "
            f"then continue in {TARGET_LANGUAGE}."
        )
    if LAST_SESSION_RECAP:
        lines.append(
            "For continuity, here is a short recap of your last session with "
            "this student. Use it to remember context, don't repeat it verbatim:"
        )
        lines.append(LAST_SESSION_RECAP)
    if context_chunks:
        lines.append("Reference material from the course, use it only if relevant:")
        lines.extend(f"- {chunk}" for chunk in context_chunks)
    return "\n".join(lines)

# ===== Conversation state =====

_history = []  # list of {"role": ..., "content": ...}, excluding the system prompt
HISTORY_TURN_LIMIT = 12  # user+assistant messages kept; older ones are dropped

def reset_conversation():
    global _history
    _history = []
    logger.info("Conversation reset.")

def _trim_history():
    global _history
    if len(_history) > HISTORY_TURN_LIMIT:
        _history = _history[-HISTORY_TURN_LIMIT:]


# ===== Server communication =====

def get_session():
    """Lazily create a requests.Session so the TCP connection to the server
    is kept alive across turns instead of reconnecting every request."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

_session = None

def check_server():
    """Return True if the llama.cpp server is reachable and has a model loaded."""
    try:
        resp = get_session().get(f"{_base_url()}/health", timeout=CONNECT_TIMEOUT)
        return resp.ok
    except requests.RequestException as e:
        logger.error(f"llama.cpp server unreachable at {_base_url()}: {e}")
        return False

def get_server_props():
    """Fetch /props from the server (context size, model path, etc.) for diagnostics."""
    try:
        resp = get_session().get(f"{_base_url()}/props", timeout=CONNECT_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning(f"Could not fetch server properties: {e}")
        return {}

def init_engine():
    """Verify the remote llama.cpp server is up and log what it's serving."""
    logger.info(f"Connecting to llama.cpp server at {_base_url()}...")
    if not check_server():
        return False
    props = get_server_props()
    if props:
        logger.info(f"Server ready. n_ctx={props.get('default_generation_settings', {}).get('n_ctx', '?')}")
    else:
        logger.info("Server ready.")
    return True


# ===== Generation =====

def _build_messages(user_text, context_chunks=None):
    messages = [{"role": "system", "content": build_system_prompt(context_chunks)}]
    messages.extend(_history)
    messages.append({"role": "user", "content": user_text})
    return messages

def _payload(messages, stream=False):
    return {
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repeat_penalty": REPEAT_PENALTY,
        "max_tokens": MAX_TOKENS,
        "cache_prompt": CACHE_PROMPT,
        "stream": stream,
    }

def generate_reply(user_text, context_chunks=None):
    """Send `user_text` plus conversation history to the server and return the
    tutor's reply as a string. `context_chunks`, if given, are retrieved RAG
    snippets to ground the reply in (see rag_engine.retrieve_context).
    Returns None on error."""
    messages = _build_messages(user_text, context_chunks)
    try:
        resp = get_session().post(
            f"{_base_url()}/v1/chat/completions",
            json=_payload(messages),
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError) as e:
        logger.error(f"Generation error: {e}")
        return None

    _history.append({"role": "user", "content": user_text})
    _history.append({"role": "assistant", "content": reply})
    _trim_history()
    return reply

def generate_reply_stream(user_text, context_chunks=None):
    """Like generate_reply, but yields the reply sentence-by-sentence as it
    streams in, so a caller (e.g. TTS) can start speaking before the full
    reply has finished generating. Yields text chunks; appends the full
    reply to history once the stream ends."""
    messages = _build_messages(user_text, context_chunks)
    buffer = ""
    full_reply = ""
    try:
        resp = get_session().post(
            f"{_base_url()}/v1/chat/completions",
            json=_payload(messages, stream=True),
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
            stream=True,
        )
        resp.raise_for_status()
        # Decode raw bytes ourselves instead of iter_lines(decode_unicode=True):
        # that flag decodes using resp.encoding, which requests defaults to
        # Latin-1 for text/event-stream responses lacking an explicit charset,
        # mangling any multibyte UTF-8 text (e.g. German umlauts/eszett).
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            if not delta:
                continue
            buffer += delta
            full_reply += delta
            if buffer[-1:] in ".!?\n":
                yield buffer.strip()
                buffer = ""
    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Streaming generation error: {e}")
        return
    if buffer.strip():
        yield buffer.strip()

    _history.append({"role": "user", "content": user_text})
    _history.append({"role": "assistant", "content": full_reply.strip()})
    _trim_history()


def summarize_session(transcript_text):
    """Ask the LLM for a compact summary of a finished session's transcript
    (topics covered, mistakes corrected, vocabulary used), for the next
    session's recap (see main.py's _finalize_transcript). This is a one-off
    request with its own summarization prompt -- not the tutor persona, and
    not part of `_history`. Returns None on error."""
    vocab_instruction = (
        "notable vocabulary used. For vocabulary, list only "
        "ordinary words worth reviewing (nouns, verbs, adjectives, "
        "idioms) and include each noun's article or grammatical "
        f"gender marker if {TARGET_LANGUAGE} has one"
    )
    if NATIVE_LANGUAGE:
        vocab_instruction += f", followed by its {NATIVE_LANGUAGE} translation"
    vocab_instruction += ". Never include "

    messages = [
        {
            "role": "system",
            "content": (
                f"You summarize a {TARGET_LANGUAGE}-learning conversation "
                "transcript for the tutor's own notes on the student. Be "
                "concise and factual: list the topics discussed, any "
                "mistakes that were corrected (with the correction), and "
                f"{vocab_instruction}"
                "names of specific places, streets, squares, landmarks, or "
                "people, even though they are capitalized nouns; a proper "
                "noun naming a specific place or person is not reusable "
                "vocabulary, no matter how it's formed in "
                f"{TARGET_LANGUAGE}. No greetings, no commentary, 5 lines "
                "maximum."
            ),
        },
        {"role": "user", "content": transcript_text},
    ]
    try:
        resp = get_session().post(
            f"{_base_url()}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": SUMMARY_TEMPERATURE,
                "max_tokens": SUMMARY_MAX_TOKENS,
                "cache_prompt": False,
            },
            timeout=(CONNECT_TIMEOUT, SUMMARY_TIMEOUT),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError) as e:
        logger.error(f"Session summary error: {e}")
        return None


# ===== Main =====


def main():
    global LLAMACPP_HOST, LLAMACPP_PORT, TARGET_LANGUAGE, TEACHER_LEVEL, TUTOR_NAME
    args = sys.argv[1:]

    if "--ui-lang" in args:
        try:
            set_ui_language(args[args.index("--ui-lang") + 1])
        except IndexError:
            print(_("Usage: --ui-lang <language-code>"))

    if "--help" in args:
        print(_("LLM Engine - llama.cpp tutor client"))
        print(_("\nUsage: python3 llm_engine.py [--host <ip-or-hostname>] [--port <port>] [--lang-target <language>] [--level <cefr-level>] [--ui-lang <language-code>]"))
        print(_("  --host          llama.cpp server host (default: env LLAMACPP_HOST or 127.0.0.1)"))
        print(_("  --port          llama.cpp server port (default: env LLAMACPP_PORT or 8080)"))
        print(_("  --lang-target   Target language the tutor teaches (default: env TARGET_LANGUAGE or German). Also"))
        print(_("                  switches the tutor name to match, per languages.py, unless env TUTOR_NAME is set."))
        print(_("  --level         Student's CEFR level, e.g. A1/A2/B1/B2 (default: env TEACHER_LEVEL or B1)"))
        print(_("  --ui-lang       Language for this CLI's own text, e.g. en/hu (default: env UI_LANGUAGE or system locale)"))
        sys.exit(0)

    if "--host" in args:
        try:
            LLAMACPP_HOST = args[args.index("--host") + 1]
        except IndexError:
            print(_("Usage: --host <ip-or-hostname>"))
    if "--port" in args:
        try:
            LLAMACPP_PORT = args[args.index("--port") + 1]
        except IndexError:
            print(_("Usage: --port <port>"))
    if "--lang-target" in args:
        try:
            TARGET_LANGUAGE = args[args.index("--lang-target") + 1]
        except IndexError:
            print(_("Usage: --lang-target <language>"))
        else:
            # Keep the tutor's name in sync with the language switch, unless
            # the user pinned it explicitly via TUTOR_NAME.
            if "TUTOR_NAME" not in os.environ:
                try:
                    TUTOR_NAME = languages.get_language(TARGET_LANGUAGE.lower())["tutor_name"]
                except ValueError:
                    logger.warning(
                        f"No languages.py profile for '{TARGET_LANGUAGE}', keeping tutor name '{TUTOR_NAME}'."
                    )
    if "--level" in args:
        try:
            TEACHER_LEVEL = args[args.index("--level") + 1]
        except IndexError:
            print(_("Usage: --level <cefr-level>"))

    if not init_engine():
        print(_("Could not reach llama.cpp server at {url}").format(url=_base_url()))
        return

    print(_("Chatting with your {lang} tutor ({level}). Ctrl+C to quit.\n").format(
        lang=TARGET_LANGUAGE, level=TEACHER_LEVEL
    ))
    try:
        while True:
            user_text = input(colors.user(_("You: "))).strip()
            if not user_text:
                continue
            print(colors.tutor(_("Tutor: ")), end="", flush=True)
            for chunk in generate_reply_stream(user_text):
                print(colors.tutor(chunk), end=" ", flush=True)
            print()
    except (KeyboardInterrupt, EOFError):
        print(_("\nBye!"))

if __name__ == "__main__":
    main()
