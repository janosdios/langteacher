# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0] - 2026-07-24

### Added
- Input method and output method choice at the start of every session,
  asked right after native language and saved to `config.<hostname>.ini`
  alongside your other picks. Input can be voice (microphone + speech
  recognition, the previous default) or text (typed at a prompt); output
  can be voice (spoken aloud) or text (printed only). Choosing text for
  either side skips its device-selection prompt and, at startup, its model
  preload (Whisper for text input, the TTS voice model for text output),
  so a text-only session starts faster and needs no microphone or speaker.

### Changed
- In role-play mode, the tutor now introduces herself by name and role
  together once the scenario starts, so her fixed tutor persona and the
  in-scene character (e.g. a waiter, a shopkeeper) stay linked from the
  first line instead of only being connected implicitly.

## [1.1.0] - 2026-07-23

### Added
- Practice-mode selection at the start of every session: free conversation,
  role-play, vocabulary quiz, translation practice, or a custom goal. In
  role-play mode the tutor asks the student for a scenario and which of the
  two roles they want to play, then stays in character for the rest of the
  session. Built-in modes live in the new `practice_modes.py`; a custom goal
  is read from a `.md` file the student drops in `custom_goals/` (see
  `custom_goals/put_custom_goal_files_here.txt`), listing the 5 most
  recently modified files to choose from. The choice is asked fresh on
  every startup and is never saved to `config.<hostname>.ini`. Custom goal
  files can reference `{NATIVE_LANGUAGE}`/`{TARGET_LANGUAGE}`, substituted
  with the session's actual languages, and can be as short as one line or a
  full multi-paragraph drill with its own instructions -- the goal's text is
  given to the tutor on its own line rather than wrapped inline in quotes, so
  longer, structured goals stay readable in the assembled prompt.
- `--practice-mode <mode>` and `--custom-goal <filename>` flags for both
  `main.py` and `llm_engine.py`'s standalone CLI, for scripted/non-interactive
  runs. Passing `--practice-mode` skips `main.py`'s interactive picker
  entirely; an invalid mode, or `--practice-mode custom` without a
  `--custom-goal` naming an existing file in `custom_goals/`, prints a clear
  error and exits immediately instead of falling back silently. Passing
  `--custom-goal` alone, with no `--practice-mode`, implies `custom` mode.
- `--native-lang <language>` flag for `llm_engine.py`'s standalone CLI,
  mirroring `--lang-target`/`--level`.
- `--debug` flag for `main.py`, enabling debug-level logging across all
  engines (including the full assembled tutor system prompt on every turn),
  and for `llm_engine.py`'s, `tts_engine.py`'s, and `rag_engine.py`'s own
  standalone CLIs.

## [1.0.0] - 2026-07-22

First stable release. LangTeacher is a voice-based language tutor that runs
entirely on your own hardware, verified on macOS, Windows, Linux, and
Raspberry Pi 5+.

### Added
- Split requirements for a lighter Raspberry Pi install.
- `--debug` flag and device selection for STT test playback.

### Changed
- Suffix log and transcript filenames with the machine's hostname.
- Resample audio to the output device's native rate before playback.

### Fixed
- Missing vocabulary translations for genderless target languages.
- Nonsensical vocabulary quiz prompts that asked the student to translate a
  word already given in the target language.
- Log encoding by specifying UTF-8 for log files and settings.
- Noisy but non-fatal console output on the Raspberry Pi.

## [0.1.0] - 2026-07-15

Initial pre-release.
