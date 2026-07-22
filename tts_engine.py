import os, sys, platform

# huggingface_hub reads HF_HUB_OFFLINE into a module-level constant at import
# time, so this must be set before `omnivoice` (which pulls in huggingface_hub)
# is imported below. Without it, every run does a HEAD request per cached file
# just to confirm nothing changed upstream ("Fetching N files"), even though
# the model is already fully cached locally. Pass --check-updates to skip
# this default and let the Hub check happen.
if "--check-updates" not in sys.argv[1:] and "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

import signal
import logging
import gettext
import io
import time
import wave
from pathlib import Path
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import languages

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Translates this module's CLI-facing strings (--help banner, usage errors,
# device/voice pickers, startup/shutdown messages). Selected via UI_LANGUAGE;
# falls back to the system locale, then to the original English string if no
# catalog matches (fallback=True), so a missing/incomplete translation never
# crashes the CLI. Log messages and the built-in --test synthesis phrase are
# left in English -- the former are developer-facing, the latter is content
# fed to the TTS model rather than text shown to a person.
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

def _load_translation(language=None):
    return gettext.translation(
        "tts_engine",
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
# their own logging (e.g. logging.getLogger("tts_engine").setLevel(...)),
# since it doesn't propagate to the root logger.
logger = logging.getLogger("tts_engine")
if not logger.handlers:
    _handler = logging.FileHandler(LOGS_DIR / "tts_engine.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [tts_engine] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)

logger.info(f"Platform: {platform.system()}, {platform.platform()}")

# piper logs "Missing phoneme from id map: <char>" (via Python logging, not a
# raised error) for every diacritic/combining mark its phoneme set doesn't
# cover -- it just skips that phoneme and carries on. Harmless per occurrence
# but noisy on text with several such marks, so keep it out of the console.
logging.getLogger("piper").setLevel(logging.ERROR)

# TTS model (HuggingFace repo id or local checkpoint path)
OMNIVOICE_MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

# Which TTS backend to use: "omnivoice" (voice cloning/design, heavier, needs
# torch), "piper" (fixed pretrained voices, CPU-only, much lighter -- good for
# a Raspberry Pi), or "auto" (picks piper on a detected Raspberry Pi, omnivoice
# otherwise). See _resolve_engine().
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto")

def set_tts_engine(engine):
    """Override TTS_ENGINE at runtime (e.g. from --tts-engine or main.py).
    Invalidates the cached model since the resolved backend may change."""
    global TTS_ENGINE, _tts_model, _tts_model_engine
    TTS_ENGINE = engine
    _tts_model = None
    _tts_model_engine = None

# noinspection PyBroadException
def _resolve_engine(requested):
    """Resolve TTS_ENGINE ("omnivoice"/"piper"/"auto") to a concrete backend name."""
    if requested and requested.lower() != "auto":
        return requested.lower()
    try:
        # Local import: only needed for the Raspberry Pi auto-detect, so a
        # standalone `--engine piper` run doesn't pay for faster-whisper/pynput.
        import stt_engine
        return "piper" if stt_engine.is_raspberrypi else "omnivoice"
    except Exception:
        return "omnivoice"

# Generation parameters
# NUM_STEP = 32
NUM_STEP = 16
GUIDANCE_SCALE = 2.0
SPEED = 1.0
DENOISE = True

# Temp file
TEMP_DIR = Path(tempfile.gettempdir())
TEMP_WAV_SPEAK = TEMP_DIR / "speak.wav"

# Reference audio samples for voice cloning
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# Optional: force a specific playback device (id or name)
PLAYBACK_TARGET = os.environ.get("PLAYBACK_TARGET")

def set_playback_target(target):
    """Override PLAYBACK_TARGET at runtime (e.g. from an interactive picker in another module)."""
    global PLAYBACK_TARGET
    PLAYBACK_TARGET = target

# Optional: force a specific synthesis language (empty/"auto" = language-agnostic)
LANG_TARGET = os.environ.get("LANG_TARGET", "")

def set_lang_target(target):
    """Override LANG_TARGET at runtime (e.g. from another module)."""
    global LANG_TARGET
    LANG_TARGET = target

# Voice design instruction, e.g. "male, British accent" (empty = auto voice).
# Defaulted to a fixed gender/style so the tutor doesn't switch voices when
# the target language changes (OmniVoice's auto mode picks a voice per call,
# which isn't guaranteed to stay the same gender across languages).
INSTRUCT_TARGET = os.environ.get("TTS_INSTRUCT", "female, young adult, high pitch")

def set_instruct_target(target):
    """Override INSTRUCT_TARGET at runtime (e.g. from another module)."""
    global INSTRUCT_TARGET
    INSTRUCT_TARGET = target

# Optional: reference audio for voice cloning (path, or name/substring of a file in SAMPLES_DIR).
# Defaults to languages.py's DEFAULT_LANGUAGE profile so a standalone run of this
# module lines up with the voice main.py would pick for that same language.
REF_AUDIO_TARGET = os.environ.get(
    "REF_AUDIO_TARGET", languages.get_language(languages.DEFAULT_LANGUAGE)["ref_audio_target"]
)

def set_ref_audio_target(target):
    """Override REF_AUDIO_TARGET at runtime (e.g. from an interactive picker in another module)."""
    global REF_AUDIO_TARGET
    REF_AUDIO_TARGET = target

# Optional: transcript of REF_AUDIO_TARGET. If empty, _resolve_ref_text() falls
# back to a same-named .txt sidecar next to whichever ref audio is resolved
# (see below), or ASR auto-transcription if there's no sidecar either. Left
# empty by default rather than defaulting to languages.py's DEFAULT_LANGUAGE
# profile -- unlike REF_AUDIO_TARGET, that would pair the wrong transcript
# with the reference audio whenever --ref-audio is overridden without also
# passing a matching --ref-text.
REF_TEXT_TARGET = os.environ.get("REF_TEXT_TARGET", "")

def set_ref_text_target(target):
    """Override REF_TEXT_TARGET at runtime (e.g. from another module)."""
    global REF_TEXT_TARGET
    REF_TEXT_TARGET = target

# Directory holding downloaded Piper voice files (<name>.onnx + <name>.onnx.json
# pairs), e.g. via `python3 -m piper.download_voices <name>`.
PIPER_VOICES_DIR = Path(os.environ.get(
    "PIPER_VOICES_DIR", str(Path(__file__).resolve().parent / "piper_voices")
))

# Which Piper voice to load (filename without extension, e.g. "en_GB-alba-medium").
# Defaults to languages.py's DEFAULT_LANGUAGE profile, mirroring REF_AUDIO_TARGET.
PIPER_VOICE_TARGET = os.environ.get(
    "PIPER_VOICE_TARGET",
    languages.get_language(languages.DEFAULT_LANGUAGE).get("piper_voice", "")
)

def set_piper_voice_target(target):
    """Override PIPER_VOICE_TARGET at runtime (e.g. from main.py). Invalidates
    the cached model since a loaded PiperVoice is locked to one voice file."""
    global PIPER_VOICE_TARGET, _tts_model, _tts_model_engine
    PIPER_VOICE_TARGET = target
    _tts_model = None
    _tts_model_engine = None

def _detect_device():
    """Return the best available torch device: cuda > mps > cpu, with a matching dtype."""
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

def _init_omnivoice():
    from omnivoice import OmniVoice
    logger.info(f"Loading OmniVoice model '{OMNIVOICE_MODEL}' (this may take a moment the first time)...")

    device, dtype = _detect_device()
    logger.info(f"Using device: {device} ({dtype})")

    model = OmniVoice.from_pretrained(
        OMNIVOICE_MODEL,
        device_map=device,
        dtype=dtype,
    )

    logger.info("TTS Engine loaded successfully!")
    return model

def _init_piper():
    from piper import PiperVoice
    voice_path = PIPER_VOICES_DIR / f"{PIPER_VOICE_TARGET}.onnx"
    if not voice_path.is_file():
        raise FileNotFoundError(
            f"Piper voice '{PIPER_VOICE_TARGET}' not found in {PIPER_VOICES_DIR}. "
            f"Run: python3 -m piper.download_voices {PIPER_VOICE_TARGET}"
        )
    logger.info(f"Loading Piper voice '{PIPER_VOICE_TARGET}' from {voice_path}...")
    voice = PiperVoice.load(str(voice_path), use_cuda=False)
    logger.info("TTS Engine loaded successfully!")
    return voice

def init_model(engine):
    logger.info("Starting TTS Engine...")
    logger.info(f"Resolved TTS engine: {engine}")
    return _init_piper() if engine == "piper" else _init_omnivoice()

_tts_model = None
_tts_model_engine = None

def get_tts_model():
    """Lazily load the TTS model for the currently resolved backend and cache
    it, reloading if the resolved engine (or, for Piper, the selected voice)
    has changed since the last call."""
    global _tts_model, _tts_model_engine
    engine = _resolve_engine(TTS_ENGINE)
    if _tts_model is None or _tts_model_engine != engine:
        _tts_model = init_model(engine)
        _tts_model_engine = engine
    return _tts_model

def _get_sd_device(target):
    """Resolve PLAYBACK_TARGET to a sounddevice output device index, or None for default."""
    if target is None:
        return None
    try:
        return int(target)
    except (ValueError, TypeError):
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if target.lower() in d['name'].lower() and d['max_output_channels'] > 0:
                return i
    logger.warning(f"Could not find playback device '{target}', using default.")
    return None

def select_playback_target():
    devices_available = sd.query_devices()
    if len(devices_available) == 0:
        logger.error("No available output device found!")
        print(_("No available output device found!"))
        exit(_("Application terminated!"))

    for i, dev_available in enumerate(devices_available):
        if dev_available.get('max_output_channels') != 0:
            print(f"{i}: {dev_available.get('name')}")

    while True:
        selected_playback = int(input(_('Please select an output device: ')).lower())
        if sd.query_devices(device=selected_playback)['max_output_channels'] != 0:
            break
        print(_("Error! Please select a valid output device!"))
    return selected_playback

# noinspection PyBroadException
def get_playback_target_name(target):
    """Return the device name for a playback device index (e.g. one loaded
    from settings.py), or a placeholder if that index no longer exists --
    devices can disappear between runs (unplugged, driver change, etc.)."""
    try:
        return sd.query_devices(device=int(target))['name']
    except Exception:
        return f"device {target} (not found)"

def _resolve_ref_audio(target):
    """Resolve REF_AUDIO_TARGET to a file path: direct path, filename in SAMPLES_DIR, or a name substring."""
    if not target:
        return None
    p = Path(target)
    if p.is_file():
        return str(p)
    candidate = SAMPLES_DIR / target
    if candidate.is_file():
        return str(candidate)
    if SAMPLES_DIR.is_dir():
        for f in sorted(SAMPLES_DIR.iterdir()):
            if f.is_file() and target.lower() in f.name.lower():
                return str(f)
    logger.warning(f"Could not find reference audio '{target}', using auto voice.")
    return None

def _resolve_ref_text(ref_audio, explicit_ref_text):
    """Return the transcript to use for `ref_audio`: REF_TEXT_TARGET if set,
    otherwise a same-named .txt sidecar next to the audio file if one exists
    (e.g. trixie_de.mp3 -> trixie_de.txt), otherwise None (auto-transcribe)."""
    if explicit_ref_text:
        return explicit_ref_text
    if not ref_audio:
        return None
    sidecar = Path(ref_audio).with_suffix(".txt")
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip()
    return None

def select_ref_audio():
    """List reference audio files in SAMPLES_DIR and let the user pick one for voice cloning (or none)."""
    if not SAMPLES_DIR.is_dir():
        logger.warning(f"Samples directory not found: {SAMPLES_DIR}")
        return None

    samples = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_file())
    if not samples:
        logger.warning(f"No reference audio files found in {SAMPLES_DIR}")
        return None

    print(_("0: (none - auto voice)"))
    for i, p in enumerate(samples, start=1):
        print(f"{i}: {p.name}")

    while True:
        try:
            selected = int(input(_('Please select a reference voice: ')))
        except ValueError:
            print(_("Error! Please enter a number!"))
            continue
        if selected == 0:
            return None
        if 1 <= selected <= len(samples):
            return str(samples[selected - 1])
        print(_("Error! Please select a valid option!"))

_voice_clone_prompt = None
_voice_clone_prompt_key = None

def get_voice_clone_prompt(model, ref_audio, ref_text):
    """Build a VoiceClonePrompt from `ref_audio`/`ref_text` once and cache it.

    model.generate() rebuilds this from scratch on every call when given raw
    ref_audio/ref_text: reload + resample the clip, trim silence, re-encode it
    through the audio tokenizer, and (if ref_text is empty) run a full Whisper
    ASR pass over it. None of that changes between turns, so we do it once per
    reference and reuse the resulting prompt instead of paying that cost on
    every sentence.
    """
    global _voice_clone_prompt, _voice_clone_prompt_key
    key = (ref_audio, ref_text)
    if _voice_clone_prompt is None or _voice_clone_prompt_key != key:
        logger.info(f"Building voice clone prompt from '{ref_audio}'...")
        _voice_clone_prompt = model.create_voice_clone_prompt(ref_audio, ref_text)
        _voice_clone_prompt_key = key
    return _voice_clone_prompt

def synthesize_speech(model, text):
    """Generate speech audio for `text`. Returns (audio, sample_rate) or (None, None)."""
    logger.debug("Synthesizing...")
    try:
        if _tts_model_engine == "piper":
            return _synthesize_piper(model, text)
        return _synthesize_omnivoice(model, text)
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return None, None

def _synthesize_omnivoice(model, text):
    from omnivoice import OmniVoiceGenerationConfig
    language = LANG_TARGET if LANG_TARGET and LANG_TARGET.lower() != "auto" else None
    ref_audio = _resolve_ref_audio(REF_AUDIO_TARGET)
    ref_text = _resolve_ref_text(ref_audio, REF_TEXT_TARGET)
    # Voice clone and voice design are separate modes (see OmniVoice.generate
    # docs) - only fall back to the instruct-based design voice when there's
    # no reference sample to clone from.
    voice_clone_prompt = get_voice_clone_prompt(model, ref_audio, ref_text) if ref_audio else None
    instruct = INSTRUCT_TARGET if (INSTRUCT_TARGET and not voice_clone_prompt) else None
    gen_config = OmniVoiceGenerationConfig(
        num_step=NUM_STEP,
        guidance_scale=GUIDANCE_SCALE,
        denoise=DENOISE,
    )
    audios = model.generate(
        text=text,
        language=language,
        instruct=instruct,
        voice_clone_prompt=voice_clone_prompt,
        speed=SPEED,
        generation_config=gen_config,
    )
    return audios[0], model.sampling_rate

def _synthesize_piper(model, text):
    """Synthesize via Piper into an in-memory WAV, then read it back into the
    same (audio, sample_rate) shape OmniVoice returns, so save_wav()/play_audio()
    need no backend-specific handling."""
    from piper import SynthesisConfig
    syn_config = SynthesisConfig(length_scale=(1.0 / SPEED if SPEED else 1.0))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        model.synthesize_wav(text, wf, syn_config=syn_config)
    buf.seek(0)
    return sf.read(buf, dtype="float32")

def save_wav(audio, filepath, sample_rate):
    sf.write(str(filepath), audio, sample_rate)
    logger.debug(f"Audio saved to {filepath.resolve()}")

def _resample_for_device(audio, sample_rate, device):
    """Resample `audio` to the output device's native rate if they differ.

    CoreAudio/WASAPI/PulseAudio transparently resample on playback, but some
    ALSA output devices (e.g. a Raspberry Pi's default hw device) don't --
    they just play the samples at the wrong rate, changing speed and pitch.
    Doing the conversion ourselves fixes that and is a no-op everywhere the
    rates already match.
    """
    try:
        info = sd.query_devices(device=device) if device is not None else sd.query_devices(kind="output")
        device_rate = int(round(info["default_samplerate"]))
    except Exception as e:
        logger.warning(f"Could not query output device sample rate, skipping resample check: {e}")
        return audio, sample_rate
    if device_rate == sample_rate:
        return audio, sample_rate

    from pydub import AudioSegment
    start = time.perf_counter()
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    segment = AudioSegment(pcm16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=channels)
    segment = segment.set_frame_rate(device_rate)
    resampled = np.frombuffer(segment.raw_data, dtype=np.int16).astype(np.float32) / 32767
    if channels > 1:
        resampled = resampled.reshape(-1, channels)
    elapsed_ms = (time.perf_counter() - start) * 1000
    audio_duration_ms = len(audio) / sample_rate * 1000
    logger.debug(
        f"Resampled audio {sample_rate} Hz -> {device_rate} Hz "
        f"({audio_duration_ms:.0f} ms of audio) in {elapsed_ms:.1f} ms"
    )
    return resampled, device_rate

def play_audio(audio, sample_rate, device=None):
    try:
        audio, sample_rate = _resample_for_device(audio, sample_rate, device)
        sd.play(audio, sample_rate, device=device)
        sd.wait()
    except Exception as e:
        logger.error(f"Playback error: {e}")
        raise

def speak_text(text):
    """Synthesize `text` and play it back on PLAYBACK_TARGET (or the default output device)."""
    try:
        model = get_tts_model()
        audio, sample_rate = synthesize_speech(model, text)
        if audio is None:
            return False
        save_wav(audio, TEMP_WAV_SPEAK, sample_rate)
        device = _get_sd_device(PLAYBACK_TARGET)
        play_audio(audio, sample_rate, device=device)
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        print(_("\nError: {error}").format(error=e))
        return False


# ===== Main =====

# noinspection PyBroadException
def main():

    global PLAYBACK_TARGET, LANG_TARGET, INSTRUCT_TARGET, REF_AUDIO_TARGET, REF_TEXT_TARGET, PIPER_VOICE_TARGET
    args = sys.argv[1:]

    if "--ui-lang" in args:
        try:
            set_ui_language(args[args.index("--ui-lang") + 1])
        except IndexError:
            print(_("Usage: --ui-lang <language-code>"))

    if "--engine" in args:
        try:
            set_tts_engine(args[args.index("--engine") + 1])
        except IndexError:
            print(_("Usage: --engine <omnivoice|piper|auto>"))

    if "--playback-target" in args:
        try:
            PLAYBACK_TARGET = args[args.index("--playback-target") + 1]
        except Exception:
            print(_("Usage: --playback-target <output-id-or-name>"))

    if "--lang-target" in args:
        try:
            LANG_TARGET = args[args.index("--lang-target") + 1]
        except Exception:
            print(_("Usage: --lang-target <language-code>"))

    if "--instruct" in args:
        try:
            INSTRUCT_TARGET = args[args.index("--instruct") + 1]
        except Exception:
            print(_("Usage: --instruct <voice-style-description>"))

    if "--ref-audio" in args:
        try:
            REF_AUDIO_TARGET = args[args.index("--ref-audio") + 1]
        except Exception:
            print(_("Usage: --ref-audio <path-or-name-in-samples-dir>"))

    if "--ref-text" in args:
        try:
            REF_TEXT_TARGET = args[args.index("--ref-text") + 1]
        except Exception:
            print(_("Usage: --ref-text <transcript-of-reference-audio>"))

    # Keep the cloned voice in sync with --lang-target, the same way main.py
    # pairs LANG_PROFILE["ref_audio_target"]/["ref_text_target"] with the
    # chosen language -- otherwise REF_AUDIO_TARGET/REF_TEXT_TARGET are left
    # at their env-var/DEFAULT_LANGUAGE defaults regardless of --lang-target,
    # so e.g. --lang-target hungarian would silently still clone the German
    # sample voice. Only applies when the user didn't already pin the voice
    # explicitly via --ref-audio/--ref-text.
    if "--lang-target" in args:
        try:
            lang_profile = languages.get_language(LANG_TARGET.lower())
        except ValueError as e:
            logger.warning(f"No languages.py profile for '{LANG_TARGET}', keeping current voice.")
            print(_("Warning: {error} -- keeping current voice.").format(error=e))
        else:
            if "--ref-audio" not in args:
                REF_AUDIO_TARGET = lang_profile["ref_audio_target"]
            if "--ref-text" not in args:
                REF_TEXT_TARGET = lang_profile["ref_text_target"]
            PIPER_VOICE_TARGET = lang_profile.get("piper_voice", "")

    # --instruct without an explicit --ref-audio means the user wants a
    # generated (designed) voice, not a cloned one. but REF_AUDIO_TARGET
    # always has a non-empty default (env var, or DEFAULT_LANGUAGE's profile,
    # or the --lang-target sync above), and synthesize_speech() only falls
    # back to INSTRUCT_TARGET when there's no reference audio to clone. see
    # its "Voice clone and voice design are separate modes" comment. So
    # clear it here, otherwise --instruct is silently ignored in favor of
    # whatever reference voice happened to be set.
    if "--instruct" in args and "--ref-audio" not in args:
        REF_AUDIO_TARGET = ""

    def shutdown_handler(sig, frame):
        logger.debug(f"Signal received: {sig}, {frame} ")
        print(_("\n\nShutting down..."))
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if len(args) > 0:
        if "--help" in args:
            print(_("TTS Engine - OmniVoice / Piper"))
            print(_("\nUsage: python3 tts_engine.py [--engine <omnivoice|piper|auto>] [--playback-target <id-or-name>] [--lang-target <language-code>] [--instruct <voice-style>] [--ref-audio <path-or-name>] [--ref-text <transcript>] [--test] [--list-devices] [--list-voices] [--ui-lang <language-code>]"))
            print(_("  --engine           TTS backend to use: omnivoice (voice cloning/design), piper (lightweight, e.g. for a Raspberry Pi), or auto (default: env TTS_ENGINE or 'auto')"))
            print(_("  --playback-target  Force a specific output device (index or name substring)"))
            print(_("  --lang-target      Force a specific synthesis language (default: auto)"))
            print(_("  --instruct         Voice design instruction, e.g. \"male, British accent\" (omnivoice only)"))
            print(_("  --ref-audio        Reference audio for voice cloning (path, or name/substring of a file in samples/) (omnivoice only)"))
            print(_("  --ref-text         Transcript of the reference audio (default: auto-transcribed) (omnivoice only)"))
            print(_("  --list-devices     Show all available audio input/output devices"))
            print(_("  --list-voices      Show reference audio files available in samples/ (omnivoice only)"))
            print(_("  --check-updates    Check the HF Hub for a newer model revision instead of using the local cache offline (omnivoice only)"))
            print(_("  --test             Synthesize and play a short built-in phrase (quick sanity check)"))
            print(_("  --ui-lang          Language for this CLI's own text, e.g. en/hu (default: env UI_LANGUAGE or system locale)"))
            sys.exit(0)
        elif "--list-devices" in args:
            hostapis = sd.query_hostapis()
            output_devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]
            print(_("Output devices (usable with --playback-target <index>):"))
            if output_devices:
                for i, d in output_devices:
                    print(f"  {i}: {d['name']} [{hostapis[d['hostapi']]['name']}] ({d['max_output_channels']} out)")
            else:
                print(_("  (none found)"))
            print(_("\nDefault output: {name}").format(name=sd.query_devices(kind='output')['name']))
            sys.exit(0)
        elif "--list-voices" in args:
            if SAMPLES_DIR.is_dir() and any(SAMPLES_DIR.iterdir()):
                print(_("Reference voices in {dir}:").format(dir=SAMPLES_DIR))
                for p in sorted(SAMPLES_DIR.iterdir()):
                    if p.is_file():
                        print(f"  {p.name}")
            else:
                print(_("No reference audio files found in {dir}").format(dir=SAMPLES_DIR))
            sys.exit(0)
        elif args[0] == "--test" or "--test" in args:
            print(_("Loading voice model..."))
            model = get_tts_model()
            print(_("Synthesizing test phrase..."))
            audio, sample_rate = synthesize_speech(model, f"This is a test of the {_tts_model_engine} text to speech engine.")
            if audio is None:
                print(_("Synthesis failed during test."))
                sys.exit(1)
            save_wav(audio, TEMP_WAV_SPEAK, sample_rate)
            print(_("Playing back test synthesis..."))
            play_audio(audio, sample_rate, device=_get_sd_device(PLAYBACK_TARGET))
            print(_("Audio test complete!"))
            sys.exit(0)

    print(_("Loading voice model..."))
    get_tts_model()

    print(_("TTS engine ready!"))
    print(_("Setup:"))
    print(_("- Speaker: Sounddevice default output"))
    print(_("- Stop: Press Ctrl+C"))
    if PLAYBACK_TARGET:
        print(_("- Playback target override: {target} - {name}").format(
            target=PLAYBACK_TARGET, name=sd.query_devices(device=int(PLAYBACK_TARGET))['name']
        ))
    print(_("- Engine: {engine}").format(engine=_tts_model_engine))
    print(_("- Language: {lang}").format(lang=LANG_TARGET if LANG_TARGET else _("auto")))
    if _tts_model_engine == "piper":
        print(_("- Voice: {voice}").format(voice=PIPER_VOICE_TARGET or _("(none set)")))
    else:
        resolved_ref_audio = _resolve_ref_audio(REF_AUDIO_TARGET)
        if resolved_ref_audio:
            print(_("- Voice: cloned from {ref_audio}").format(ref_audio=resolved_ref_audio))
        else:
            print(_("- Voice: {instruct}").format(instruct=INSTRUCT_TARGET if INSTRUCT_TARGET else _("auto")))

    while True:
        try:
            text = input(_("Enter text to speak: ")).strip()
            if not text:
                continue
            speak_text(text)
        except KeyboardInterrupt:
            print(_("\n\nInterrupted by user"))
            break
        except Exception as e:
            print(_("\nError: {error}").format(error=e))

    print(_("\nBye!"))

if __name__ == "__main__":
    main()
