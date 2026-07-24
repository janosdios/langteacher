# LangTeacher

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)
[![Platforms: macOS | Windows | Linux | Raspberry Pi 5+](https://img.shields.io/badge/Platforms-macOS%20%7C%20Windows%20%7C%20Linux%20%7C%20Raspberry%20Pi%205%2B-lightgrey.svg)](#requirements)
[![Release](https://img.shields.io/github/v/release/janosdios/langteacher)](https://github.com/janosdios/langteacher/releases)

A voice-based language tutor that runs entirely on your own hardware. Talk
to it out loud in the language you're learning, it listens, thinks, and
talks back in a cloned or generated voice — corrections, native-language
help, and lesson material included.

Everything runs locally: speech recognition ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)),
the tutor LLM ([llama.cpp](https://github.com/ggml-org/llama.cpp)), and
text-to-speech via [OmniVoice](https://github.com/k2-fsa/OmniVoice) (voice
cloning/design) or, on lighter hardware like a Raspberry Pi, [Piper](https://github.com/OHF-Voice/piper1-gpl)
(fixed pretrained voices, CPU-only, no language switch). No cloud API
keys, no per-message cost.

## Features

- **Spoken conversation** — push-to-talk or hands-free (voice-activity
  detection) input, streamed spoken replies.
- **Optional text mode** — pick text instead of voice for input, output, or
  both, at the start of any session. Text input skips the microphone
  entirely (type your turn instead); text output just prints the tutor's
  replies instead of speaking them. Handy for a quiet room or a machine
  with no mic/speaker.
- **Multiple target languages** — German, French, Spanish, Hungarian,
  English out of the box, each with its own tutor persona and cloned voice
  (see [`languages.py`](languages.py)).
- **Add any target language** - If Whisper, Omnivoice/PiperTTS and your Tutor 
  LLM knows a language, you can use it with LandTeacher.
- **Native-language code-switching** — speak in your native language to ask
  "what does X mean?" or say you're lost, and the tutor answers briefly in
  your native language before returning to the target language. Speech
  recognition is constrained to just your two languages (target + native),
  so it doesn't mistake native-language speech for noise or a third
  language.
- **CEFR-aware tutoring** — set your level (A1–C2); the tutor adjusts
  vocabulary and sentence complexity, and corrects at most one mistake per
  turn.
- **Practice modes** — pick free conversation, role-play, a vocabulary quiz,
  translation practice, or your own custom goal at the start of every
  session (asked fresh every run, never saved between sessions). A custom
  goal can be a whole drill you write yourself as a `.md` file in
  `custom_goals/` (see [Practice modes](#practice-modes)).
- **Voice cloning** — reference audio + transcript per tutor persona
  (`samples/`), or fall back to a generated (`--instruct`) voice (OmniVoice
  backend).
- **Two TTS backends** — OmniVoice (voice cloning/design, heavier) or Piper
  (fixed pretrained voices, CPU-only, much lighter), selectable 
  via `TTS_ENGINE`/`--tts-engine`, or auto-picked based on whether you're 
  running on a Raspberry Pi.
- **Optional RAG knowledge base** — ingest your own course PDFs
  (`rag_engine.py`) so the tutor can pull in real lesson material relevant
  to what you're discussing.
- **Session recaps** — each session's transcript is summarized (topics,
  corrections, vocabulary with native-language translations) and fed back
  into the next session's opening context.
- **Raspberry Pi friendly** — optional GPIO stop button, CPU-only inference
  by default, TTS auto-switches to the lighter Piper backend when a
  Raspberry Pi is detected, option to use remote LLM as Tutor.
- **Standalone modules** - STT engine, TTS engine, LLM engine are usable
  as standalone applications.
- **Supported platforms** - macOS, Windows, Linux, Raspberry Pi 5+

## Architecture

| Module | Responsibility |
|---|---|
| [`main.py`](main.py) | Orchestrates a session: settings, the listen → think → speak loop, transcripts/recaps. |
| [`stt_engine.py`](stt_engine.py) | Microphone capture (VAD or push-to-talk) and transcription via faster-whisper. |
| [`llm_engine.py`](llm_engine.py) | Tutor persona, system prompt, streaming chat against a llama.cpp server, session summaries. |
| [`practice_modes.py`](practice_modes.py) | Built-in practice-mode prompts (conversation, role-play, vocabulary quiz, translation practice) and the custom-goal-file picker (`custom_goals/`). |
| [`tts_engine.py`](tts_engine.py) | Speech synthesis and playback, via OmniVoice (voice cloning/design) or Piper (lightweight, pretrained voices). |
| [`rag_engine.py`](rag_engine.py) | PDF ingestion and retrieval for the optional course-material knowledge base. |
| [`languages.py`](languages.py) | Per-language tutor profiles (STT code, OCR pack, persona, reference voice, Piper voice) and CEFR/native-language pickers. |
| [`settings.py`](settings.py) | Persists your language/level/native-language/device/input-output-method picks between runs (`config.<hostname>.ini`, one file per machine). |

Each engine module also works standalone for testing, e.g.
`python3 stt_engine.py --test` or `python3 tts_engine.py --list-voices`
— run any of them with `--help` for its own flags.

## Requirements

- Python 3.10+
- A running [llama.cpp](https://github.com/ggml-org/llama.cpp) server
  (`llama-server`) with a chat-capable GGUF model, for the tutor LLM.
- A microphone and speaker/output device.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) if you plan to
  ingest scanned-image PDFs into the RAG knowledge base.
- Optionally, a second llama.cpp server loaded with an embedding model
  (e.g. bge-m3, bge-small, nomic-embed-text) for RAG.
- A CUDA GPU is used automatically if available; otherwise everything runs
  on CPU.
- If you plan to use the Piper TTS backend, the `espeak-ng` system package
  (not pip-installable): `apt install espeak-ng` on Debian/Raspberry Pi OS,
  `brew install espeak-ng` on macOS.

## Setup

Create an isolated Python environment for LangTeacher via
[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
(requires Miniconda/Anaconda already installed), then activate it:

```bash
conda create -n langteacher python=3.11
conda activate langteacher
```

Install the dependencies into that environment:

```bash
pip install -r requirements.txt
```

On a Raspberry Pi, use `requirements-rpi.txt` instead — it skips `torch` and
`omnivoice` (unneeded there since the [Piper TTS backend](#optional-piper-tts-backend-recommended-for-a-raspberry-pi)
is CPU-only and torch-free):

```bash
pip install -r requirements-rpi.txt
```

Start your llama.cpp chat server for the tutor LLM. Any chat-capable GGUF model
works; `Qwen3.5-9B-Q5_K_M.gguf` is a good default:

```bash
llama-server -m /path/to/Qwen3.5-9B-Q5_K_M.gguf --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --parallel 1 --cache-reuse 256 -ngl 999 --mlock --jinja \
  --temp 0.6 --top-p 0.95 --top-k 20 --min_p 0 --repeat-penalty 1.05 \
  --chat-template-kwargs '{"enable_thinking":false}'
```

If you're also using the [optional RAG knowledge base](#optional-rag-knowledge-base),
start a second llama.cpp server on a different port for the embedding model;
`bge-m3-Q8_0.gguf` is a good default:

```bash
llama-server -m /path/to/bge-m3-Q8_0.gguf --embedding -c 2048 \
  --n-gpu-layers 99 --parallel 1 --host 0.0.0.0 --port 8081
```

Then run the tutor:

```bash
python3 main.py
```

To point at llama.cpp servers running on other machines, pass `--tutor-host`/`--tutor-port`
(the tutor chat model) and/or `--rag-host`/`--rag-port` (the RAG embedding model) — these
are equivalent to setting `LLAMACPP_HOST`/`PORT` and `EMBED_HOST`/`PORT`, just as one-off
flags instead of env vars:

```bash
python3 main.py --tutor-host 192.168.1.10 --tutor-port 8080 --rag-host 192.168.1.11 --rag-port 8081
```

On first run you'll be prompted to pick your target language, CEFR level,
native language, input method (voice or text), and output method (voice or
text) — these are saved to `config.<hostname>.ini` (one per machine, since
mic/playback picks don't carry over between machines) and offered again
next time. Picking voice for input or output also asks for a microphone or
playback device, respectively; picking text skips that device prompt
entirely, and startup skips loading the corresponding model (Whisper for
text input, the TTS voice model for text output). With voice input, hold
Right Shift to talk (push-to-talk is the default; see `RECORD_METHOD` in
`main.py` to switch to hands-free VAD). With text input, just type your
turn and press Enter.

### Practice modes

You're also asked what to practice at the start of every session — free
conversation, role-play, a vocabulary quiz, translation practice, or a
custom goal. This choice is asked fresh every single run and is never
saved to `config.<hostname>.ini`.

- **Free conversation** — the tutor opens by asking what topic you'd like
  to talk about, then follows your lead.
- **Role-play** — the tutor asks what scenario you'd like to act out (e.g.
  buying something in a shop) and which of the two roles you want to play,
  introduces herself by name and role, then stays in character for the
  rest of the session.
- **Vocabulary quiz** — the tutor quizzes you word by word, asking in your
  native language and expecting the target-language translation, checking
  in only after 15–20 words rather than after every single one.
- **Translation practice** — the tutor gives you a sentence in your native
  language to translate, corrects your attempt, and moves on to the next
  one, checking in only after 8–10 sentences.
- **Custom goal** — write your own goal as a `.md` file in `custom_goals/`
  (see `custom_goals/put_custom_goal_files_here.txt`); at session start
  you'll get a numbered list of the 5 most recently modified files to pick
  from. A goal file can be as short as one sentence or a full multi-
  paragraph drill with its own instructions, and can reference
  `{NATIVE_LANGUAGE}`/`{TARGET_LANGUAGE}`, substituted with your actual
  session languages. This directory is gitignored, so pulling app updates
  never touches your own goal files.

For scripted or non-interactive runs, skip the picker with `--practice-mode
<mode>` (one of `conversation`, `roleplay`, `vocab`, `translation`,
`custom`; default `conversation`) and, for `custom`, `--custom-goal
<filename>` naming a file in `custom_goals/` (with or without the `.md`
extension). Passing `--custom-goal` alone, with no `--practice-mode`,
implies `custom` mode. An invalid mode name, or `custom` mode without a
goal file that actually exists, prints a clear error and exits instead of
silently falling back to something else.

### Voice samples

A default reference voice per language ships in `samples/`. To use your own,
drop a short audio clip (+ optional same-named `.txt` transcript) in
`samples/` and point a language's `ref_audio_target` at it in
`languages.py`, or override at runtime with
`tts_engine.py --ref-audio <name>`.

### Optional: Piper TTS backend (recommended for a Raspberry Pi)

OmniVoice is the default backend, but it's heavy for a Raspberry Pi. Piper
is CPU-only, has no torch dependency, and covers all 5 taught languages
(with fixed pretrained voices — no cloning). To use it, download the voice
files it needs (one `.onnx` + `.onnx.json` pair per language, into
`piper_voices/` by default):

```bash
python3 -m piper.download_voices de_DE-thorsten-medium fr_FR-siwis-medium es_ES-davefx-medium hu_HU-imre-medium en_GB-alba-medium
```

Then either set `TTS_ENGINE=piper` (or leave it at the default `auto`,
which picks Piper automatically when a Raspberry Pi is detected), or pass
`--tts-engine piper` to `main.py`. Voice ids are set per language in
`languages.py`'s `piper_voice` field; double-check current names against
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) if a
download fails, since the catalog can change.

### Raspberry Pi: push-to-talk requires X11

Push-to-talk (holding a key to record) uses `pynput` for a global keyboard
hook, which needs an X11 session — under Raspberry Pi OS's default Wayland
desktop (Bookworm and later), `pynput` starts without error but silently
never receives the key press. If holding the push-to-talk key does nothing,
switch the Pi to X11 via `raspi-config` → *Advanced Options* → *Wayland* →
*X11*, then reboot. Alternatively, use hands-free VAD mode instead (`0` in
`stt_engine.py`'s interactive picker, or `RECORD_METHOD` in `main.py`), which
doesn't depend on `pynput` at all.

### Optional: RAG knowledge base

```bash
python3 rag_engine.py ingest path/to/book.pdf --book "Book Name" --language de --level A2 --lesson "Kapitel 3"
python3 rag_engine.py query "some question" --language de --level A2
```

Requires an embedding-model llama.cpp server (`EMBED_HOST`/`EMBED_PORT`,
default `127.0.0.1:8081`). Set `RAG_ENABLED=false` to run without it — the
tutor works fine with no knowledge base, it just skips retrieved context.
Both subcommands also take `--debug` (debug-level logging to
`logs/rag_engine.log`) and `--embed-host`/`--embed-port` to override the
embedding server target for that one invocation.

## Configuration

`main.py` is the app you actually run; `stt_engine.py`, `tts_engine.py`, and
`llm_engine.py` also work standalone (each has its own, more granular set of flags, handy
for testing one piece in isolation — e.g. `python3 tts_engine.py --test`). Every flag
below also has an environment-variable equivalent (shown in parentheses) if you'd rather
set it once instead of passing it every run; run any module with `--help` to see this
same list from the CLI.

### `main.py`

| Flag | Purpose |
|---|---|
| `--version` | Print the installed version and exit |
| `--debug` | Enable debug-level logging across all engines — including the full assembled tutor system prompt on every turn — to `logs/*.log` |
| `--no-recap` | Start with a clean slate, ignoring any previous session's recap |
| `--whisper-model <name-or-path>` | Whisper model for transcription (env `WHISPER_MODEL`, default `small`) |
| `--tts-engine <omnivoice\|piper\|auto>` | TTS backend to use (env `TTS_ENGINE`, default `auto`, which picks `piper` on a detected Raspberry Pi) |
| `--tutor-host <host>` | llama.cpp server host for the tutor LLM (env `LLAMACPP_HOST`, default `127.0.0.1`) |
| `--tutor-port <port>` | llama.cpp server port for the tutor LLM (env `LLAMACPP_PORT`, default `8080`) |
| `--rag-host <host>` | llama.cpp server host for the RAG embedding model (env `EMBED_HOST`, default `127.0.0.1`) |
| `--rag-port <port>` | llama.cpp server port for the RAG embedding model (env `EMBED_PORT`, default `8081`) |
| `--practice-mode <mode>` | Practice mode, skipping the interactive picker: `conversation`/`roleplay`/`vocab`/`translation`/`custom` (default `conversation`) — see [Practice modes](#practice-modes) |
| `--custom-goal <filename>` | Filename of a `.md` file in `custom_goals/` to use as the session's goal; required when `--practice-mode` is `custom` (and implies it if `--practice-mode` is omitted) |
| `--ui-lang <language-code>` | Language for this CLI's own text, e.g. `en`/`hu` (env `UI_LANGUAGE`, default: system locale) |

Two settings are env-only, with no CLI flag on `main.py`: `RAG_ENABLED` (default `true`)
turns the RAG knowledge base on/off, and target language/CEFR level/native
language/input method/output method/mic/playback device are picked interactively on
first run (or reused from `config.<hostname>.ini`) rather than passed as flags.

### `stt_engine.py`

| Flag | Purpose |
|---|---|
| `--mic-device <id-or-name>` | Force a specific input device (env `MIC_DEVICE_TARGET`, default: auto) |
| `--playback-target <id-or-name>` | Force a specific output device for `--test` playback (env `PLAYBACK_TARGET`, default: auto) |
| `--lang-target <language-code>` | Force a specific transcription language (default: auto-detect) |
| `--native-lang-target <language-code>` | Also allow this language, for code-switching to ask for help |
| `--whisper-model <name-or-path>` | Whisper model for transcription (env `WHISPER_MODEL`, default `small`) |
| `--list-devices` | Show all available audio input/output devices |
| `--test` | Record ~3s and play it back (quick audio sanity check) |
| `--ui-lang <language-code>` | Language for this CLI's own text (env `UI_LANGUAGE`, default: system locale) |
| `--debug` | Enable debug-level logging to `logs/stt_engine.log` |

### `tts_engine.py`

| Flag | Purpose |
|---|---|
| `--engine <omnivoice\|piper\|auto>` | TTS backend to use (env `TTS_ENGINE`, default `auto`) |
| `--playback-target <id-or-name>` | Force a specific output device (env `PLAYBACK_TARGET`, default: auto) |
| `--lang-target <language-code>` | Force a specific synthesis language (default: auto) |
| `--instruct <voice-style>` | Voice design instruction, e.g. `"male, British accent"` (env `TTS_INSTRUCT`; OmniVoice only) |
| `--ref-audio <path-or-name>` | Reference audio for voice cloning, path or name in `samples/` (OmniVoice only) |
| `--ref-text <transcript>` | Transcript of the reference audio (default: auto-transcribed; OmniVoice only) |
| `--list-devices` | Show all available audio input/output devices |
| `--list-voices` | Show reference audio files available in `samples/` (OmniVoice only) |
| `--check-updates` | Check the HF Hub for a newer model revision instead of using the local cache offline (OmniVoice only) |
| `--test` | Synthesize and play a short built-in phrase (quick sanity check) |
| `--ui-lang <language-code>` | Language for this CLI's own text (env `UI_LANGUAGE`, default: system locale) |
| `--debug` | Enable debug-level logging to `logs/tts_engine.log` |

Also relevant, env-only: `OMNIVOICE_MODEL` (default `k2-fsa/OmniVoice`, an HF repo id or
local path), `PIPER_VOICES_DIR` (default `piper_voices/`), and `PIPER_VOICE_TARGET`
(default: from `languages.py`, e.g. `en_GB-alba-medium`).

### `llm_engine.py`

| Flag | Purpose |
|---|---|
| `--host <ip-or-hostname>` | llama.cpp server host (env `LLAMACPP_HOST`, default `127.0.0.1`) |
| `--port <port>` | llama.cpp server port (env `LLAMACPP_PORT`, default `8080`) |
| `--lang-target <language>` | Target language the tutor teaches (env `TARGET_LANGUAGE`, default `German`); also switches the tutor name to match, per `languages.py`, unless env `TUTOR_NAME` is set |
| `--native-lang <language>` | Student's native language, for code-switch fallback and quiz/translation modes (env `NATIVE_LANGUAGE`, default `English`) |
| `--level <cefr-level>` | Student's CEFR level, e.g. `A1`/`A2`/`B1`/`B2` (env `TEACHER_LEVEL`, default `B1`) |
| `--practice-mode <mode>` | Practice mode: `conversation`/`roleplay`/`vocab`/`translation`/`custom` (default `conversation`) — see [Practice modes](#practice-modes) |
| `--custom-goal <filename>` | Filename of a `.md` file in `custom_goals/` to use as the goal; required when `--practice-mode` is `custom` (and implies it if `--practice-mode` is omitted) |
| `--debug` | Enable debug-level logging (e.g. the full tutor system prompt) to `logs/llm_engine.log` |
| `--ui-lang <language-code>` | Language for this CLI's own text (env `UI_LANGUAGE`, default: system locale) |

## Project layout

```
books/           Your own course PDFs for RAG ingestion
knowledge_base/  RAG's extracted chunks database
samples/         Voice-cloning reference audio
piper_voices/    Downloaded Piper .onnx/.onnx.json voice files
custom_goals/    Your own custom practice-goal .md files, see Practice modes
transcripts/     Your session transcripts + recap summaries
logs/            Per-module log files
config.<hostname>.ini   Your saved settings for this machine (one file per machine)
locale/          CLI translation catalogs, see locale/LOCALIZATION.md to edit/add one
```

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for noncommercial
use; see the license file for the full terms.
</content>
