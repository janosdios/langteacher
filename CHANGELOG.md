# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
