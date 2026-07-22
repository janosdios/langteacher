import os, sys, platform
import time
import signal
import wave
import tempfile
import threading
import logging
import gettext
import warnings
import contextlib
import colors
import languages
import settings
from pathlib import Path
from faster_whisper import WhisperModel, decode_audio
try:
    #noinspection PyProtectedMember
    from faster_whisper.tokenizer import _LANGUAGE_CODES as _FASTER_WHISPER_LANGUAGE_CODES
except ImportError:
    _FASTER_WHISPER_LANGUAGE_CODES = None
import sounddevice as sd
import numpy as np
from pynput import keyboard

# Named logger so consuming projects can see what's happening on import
# without any setup, but can also reconfigure/silence it independently of
# their own logging (e.g. logging.getLogger("stt_engine").setLevel(...)),
# since it doesn't propagate to the root logger.
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Translates this module's CLI-facing strings (--help banner, usage errors,
# status/setup messages). Selected via UI_LANGUAGE; falls back to the system
# locale, then to the original English string if no catalog matches
# (fallback=True), so a missing/incomplete translation never crashes the CLI.
# Left untranslated: log messages, and prints with no fixed text of their own
# (the spinner, the level meter, and raw device-name listings).
_LOCALE_DIR = Path(__file__).resolve().parent / "locale"

def _load_translation(language=None):
    return gettext.translation(
        "stt_engine",
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

logger = logging.getLogger("stt_engine")
if not logger.handlers:
    _handler = logging.FileHandler(LOGS_DIR / f"stt_engine.{settings.HOSTNAME}.log", encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [stt_engine] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)

logger.info(f"Platform: {platform.system()}, {platform.platform()}")

is_raspberrypi = False
raspberry_identifier = "Raspberry"
try:
    with open('/proc/device-tree/model') as f:
        model = f.read()
    logger.debug(f"Model: {model}")
    if model.find(raspberry_identifier) != -1:
        is_raspberrypi = True
except FileNotFoundError:
    logger.debug("RPI Model: not available, or not RPI")

# Optional GPIO stop button for Raspberry Pi
if is_raspberrypi:
    try:
        from gpiozero import Button
        GPIO_AVAILABLE = True
    except ImportError:
        GPIO_AVAILABLE = False
        Button = None
        logger.warning("GPIO not available - running without button support")
else:
    GPIO_AVAILABLE = False
    Button = None

# Raspberry Pi Configuration
STOP_BUTTON_GPIO_PIN = 22

# Capture format we ask the audio backend for first; if the device rejects
# it, _open_capture_stream() works down a list of fallback combinations.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Voice-activity-detection tuning. These are plain RMS-over-a-frame checks
# (no external VAD library), tuned by ear against a typical USB mic + room.
AUDIO_FRAME_MS = 30
SILENCE_RMS_FLOOR = 120.0   # frames quieter than this count as silence
TRAILING_SILENCE_MS = 800   # silence needed after speech to end the utterance
MIN_UTTERANCE_MS = 300      # ignore blips shorter than this
MAX_UTTERANCE_MS = 15000    # hard cap so a stuck open mic can't run forever

# Push-to-talk settings
# A standalone modifier key is used (rather than e.g. space) because pressing
# it alone doesn't produce a printable character or escape sequence, so it
# never leaks into the focused terminal/app and no event suppression is needed.
PTT_KEY = keyboard.Key.shift_r

_KEY_DISPLAY_NAMES = {
    "shift_r": "Right Shift", "shift_l": "Left Shift",
    "ctrl_r": "Right Ctrl", "ctrl_l": "Left Ctrl",
    "alt_r": "Right Alt", "alt_l": "Left Alt",
    "cmd_r": "Right Cmd", "cmd_l": "Left Cmd",
}

def get_ptt_key_name():
    """Human-readable name for PTT_KEY, so prompts (e.g. main.py's "ready"
    message) can tell the user which key to hold -- pynput's own Key.name
    ("shift_r") isn't user-facing on its own."""
    name = getattr(PTT_KEY, "name", None) or getattr(PTT_KEY, "char", None) or str(PTT_KEY)
    return _KEY_DISPLAY_NAMES.get(name, name.replace("_", " ").title())

# STT model. Which model is worth using depends on the platform (e.g. a
# Raspberry Pi wants something smaller than a desktop with a GPU) and the
# target language (e.g. a language-specific fine-tune), so this is left
# overridable rather than hardcoded -- see set_whisper_model()/--whisper-model.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

def set_whisper_model(model_name):
    """Override WHISPER_MODEL at runtime (e.g. from --whisper-model or main.py)."""
    global WHISPER_MODEL, _whisper_model
    WHISPER_MODEL = model_name
    _whisper_model = None

# Temp file
TEMP_DIR = Path(tempfile.gettempdir())
TEMP_WAV = TEMP_DIR / "recording.wav"
TEST_WAV = TEMP_DIR / "test.wav"

# Optional: force a specific audio input device (sounddevice index or name substring)
MIC_DEVICE_TARGET = os.environ.get("MIC_DEVICE_TARGET")

# Optional: force a specific audio output device for --test playback (sounddevice index or name substring)
PLAYBACK_TARGET = os.environ.get("PLAYBACK_TARGET")

def set_mic_target(target):
    """Override MIC_DEVICE_TARGET at runtime (e.g. from an interactive picker in another module)."""
    global MIC_DEVICE_TARGET
    MIC_DEVICE_TARGET = target

# Optional: force a specific transcription language (empty/"multi" = auto-detect)
LANG_TARGET = os.environ.get("LANG_TARGET", "")

def set_lang_target(target):
    """Override LANG_TARGET at runtime (e.g. from another module)."""
    global LANG_TARGET
    LANG_TARGET = target

# Optional: student's native language code, so they can code-switch into it
# (e.g. to ask for help) without transcription drifting to whatever language
# Whisper's unconstrained detector happens to guess. Empty = don't constrain,
# just use LANG_TARGET as before.
NATIVE_LANG_TARGET = os.environ.get("NATIVE_LANG_TARGET", "")

def set_native_lang_target(target):
    """Override NATIVE_LANG_TARGET at runtime (e.g. from another module)."""
    global NATIVE_LANG_TARGET
    NATIVE_LANG_TARGET = target

# If neither the target nor the native language clears this probability on
# Whisper's own language detection, assume the audio is target-language
# speech that was just too quiet/short/noisy for confident detection.
LANG_DETECT_MIN_PROB = 0.3

def _detect_constrained_language(whisper_model, audio_path, allowed, fallback):
    """Ask Whisper for full per-language probabilities and return whichever
    of `allowed` scores highest, instead of Whisper's own top pick (which
    could be neither language the tutor session actually uses). Falls back
    to `fallback` when even the best allowed candidate is too weak to trust."""
    audio = decode_audio(str(audio_path))
    _, _, all_probs = whisper_model.detect_language(audio)
    probs = dict(all_probs)
    best = max(allowed, key=lambda l: probs.get(l, 0.0))
    if probs.get(best, 0.0) < LANG_DETECT_MIN_PROB:
        return fallback
    return best

# Raspberry Pi GPIO button handling, if exists
# noinspection PyBroadException
def _create_stop_button():
    if not GPIO_AVAILABLE:
        return None
    try:
        # gpiozero picks its pin factory lazily, on this first Device
        # instantiation rather than on import -- it probes lgpio/RPi.GPIO/
        # pigpio/native in turn and warns on each rejected one before
        # settling on whatever works. That's expected here, not an error
        # (none of the real backends are usable without root/the native
        # kernel driver on this Pi), so it shouldn't spam the console.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="gpiozero")
            button = Button(STOP_BUTTON_GPIO_PIN, pull_up=True, bounce_time=0.1)
        logger.info(f"Stop button ready on GPIO {STOP_BUTTON_GPIO_PIN}")
        return button
    except Exception:
        logger.warning("GPIO pins not accessible")
        return None

_stop_button = None
_stop_button_initialized = False

def stop_button_handle():
    """Lazily create the GPIO stop button once and cache it (including the no-GPIO None case)."""
    global _stop_button, _stop_button_initialized
    if not _stop_button_initialized:
        _stop_button = _create_stop_button()
        _stop_button_initialized = True
    return _stop_button

# noinspection PyBroadException
def _select_compute_backend():
    """Return ("cuda", "float16") if a CUDA GPU is available via ctranslate2, else ("cpu", "int8")."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"

def _load_whisper_model():
    logger.info("Starting STT Engine...")
    logger.info("Loading Whisper (this may take a moment the first time)...")

    cpu_threads = max(1, os.cpu_count() // 2)
    if cpu_threads != 1 and cpu_threads % 2 != 0:
        cpu_threads -= 1
    logger.info(f"Detected {os.cpu_count()} CPUs, using {cpu_threads} threads")

    device, compute_type = _select_compute_backend()
    logger.info(f"Using device: {device} ({compute_type})")

    whisper = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=str(Path.home() / ".cache" / "whisper")
    )

    logger.info("STT Engine loaded successfully!")
    return whisper

_whisper_model = None

def whisper_engine():
    """Lazily load the Whisper model once and cache it, since loading takes several seconds."""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = _load_whisper_model()
    return _whisper_model

def _stop_requested(stop_button):
    if stop_button is None:
        return False
    return bool(stop_button.is_pressed)

def _resolve_device_index(target):
    """Turn MIC_DEVICE_TARGET (an index, a name substring, or None) into a
    sounddevice device index, or None to let sounddevice pick the default."""
    if target is None:
        return None
    try:
        return int(target)
    except (ValueError, TypeError):
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if target.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                return i
    logger.warning(f"Could not find mic device '{target}', using default.")
    return None

def _resolve_output_device_index(target):
    """Turn PLAYBACK_TARGET (an index, a name substring, or None) into a
    sounddevice output device index, or None to let sounddevice pick the default."""
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

@contextlib.contextmanager
def _suppress_alsa_errors():
    """PortAudio's ALSA backend writes rejected-format errors (e.g.
    'paInvalidSampleRate') straight to the process's stderr file descriptor
    from C, bypassing Python's logging/warnings entirely -- so it can't be
    caught or filtered from the Python side. Since the fallback loop below
    is expected to hit rejections on the way to a working format, redirect
    fd 2 to /dev/null for the duration rather than let PortAudio spam a
    message the user can't act on."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)

def _open_capture_stream(target):
    """Open a sounddevice RawInputStream for `target`, working down a list of
    (rate, channels) fallbacks until one is accepted -- some USB mics reject
    16kHz mono outright. Returns (stream, rate, channels, first_chunk, err_text);
    on total failure stream is None and err_text explains why."""
    device = _resolve_device_index(target)
    fallback_formats = [
        (TARGET_SAMPLE_RATE, TARGET_CHANNELS),
        (TARGET_SAMPLE_RATE, 2),
        (48000, TARGET_CHANNELS),
        (48000, 2),
    ]
    for rate, channels in fallback_formats:
        frame_samples = int(rate * AUDIO_FRAME_MS / 1000)
        try:
            with _suppress_alsa_errors():
                stream = sd.RawInputStream(
                    samplerate=rate,
                    channels=channels,
                    dtype='int16',
                    device=device,
                    blocksize=frame_samples
                )
                stream.start()
            raw_chunk, _overflowed = stream.read(frame_samples)
            chunk = bytes(raw_chunk)
            if chunk:
                return stream, rate, channels, chunk, ""
            stream.stop(); stream.close()
        except Exception as e:
            logger.debug(f"Input device refused {rate}Hz/{channels}ch: {e}")
    return None, None, None, None, "No working audio input configuration found"

def _begin_capture(status_message):
    """Print and log `status_message` (the logger alone doesn't reach the
    console -- see logs/stt_engine.log -- but the user needs to see this to
    know it's their turn to speak), open the input stream for
    MIC_DEVICE_TARGET, and return the pieces every capture function needs:
    (stream, rate, channels, first_chunk, frame_samples, bytes_per_sample).
    Returns None if no input device could be opened (already logged)."""
    print(status_message)
    logger.info(status_message)
    if MIC_DEVICE_TARGET:
        logger.info(f"Using source target: {MIC_DEVICE_TARGET} - {sd.query_devices(device=int(MIC_DEVICE_TARGET))['name']}")

    stream, rate, channels, first_chunk, err = _open_capture_stream(MIC_DEVICE_TARGET)
    if not stream:
        logger.error(err)
        return None

    frame_samples = int(rate * AUDIO_FRAME_MS / 1000)
    return stream, rate, channels, first_chunk, frame_samples, 2

# noinspection PyBroadException
def capture_until_silence(timeout_seconds=30, stop_button=None):
    """Record audio until trailing silence is detected. Returns (bytes, rate, channels) or (None, None, None)."""
    opened = _begin_capture(_("Listening... (speak now)"))
    if opened is None:
        return None, None, None
    stream, rate, channels, first_chunk, frame_samples, bytes_per_sample = opened
    audio_buffer = bytearray()

    try:
        # Quick calibration (~300ms) to learn the room's noise floor before
        # deciding what counts as "silence" for this utterance.
        noise_samples = []
        if first_chunk:
            s = np.frombuffer(first_chunk, dtype=np.int16).astype(np.float32)
            noise_samples.append(float(np.sqrt(np.mean(s * s))))
        for _i in range(9):
            raw_chunk, _overflowed = stream.read(frame_samples)
            chunk = bytes(raw_chunk)
            if chunk:
                s = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                noise_samples.append(float(np.sqrt(np.mean(s * s))))
        noise_floor = float(np.median(noise_samples)) if noise_samples else 50.0
        threshold = max(SILENCE_RMS_FLOOR, noise_floor * 1.8)
        logger.debug(f"Noise floor: {noise_floor:.1f}  |  Threshold: {threshold:.1f}")

        is_speaking = False
        silence_ms = 0
        speech_ms = 0
        total_ms = 0
        start = time.time()

        if first_chunk is not None:
            samples = np.frombuffer(first_chunk, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)))
            level = int(rms / 100)
            print(f"\r  Level: {'▁'*min(level,20):<20} ", end="", flush=True)
            if rms > threshold:
                is_speaking = True
                speech_ms = AUDIO_FRAME_MS
                audio_buffer.extend(first_chunk)

        while True:
            if _stop_requested(stop_button):
                raise KeyboardInterrupt

            if (time.time() - start) > timeout_seconds:
                if not is_speaking:
                    return None, None, None
                break

            raw_chunk, _overflowed = stream.read(frame_samples)
            chunk = bytes(raw_chunk)
            if not chunk:
                break

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
            rms = float(np.sqrt(np.mean(samples * samples)))
            level = int(rms / 100)
            print(f"\r  Level: {'▁'*min(level,20):<20} ", end="", flush=True)

            if is_speaking:
                audio_buffer.extend(chunk)
                if rms < threshold:
                    silence_ms += AUDIO_FRAME_MS
                else:
                    silence_ms = 0
                    speech_ms += AUDIO_FRAME_MS

                if silence_ms >= TRAILING_SILENCE_MS and speech_ms >= MIN_UTTERANCE_MS:
                    dur_s = len(audio_buffer) / (rate * bytes_per_sample * channels)
                    print(_("\n Recorded {duration}s").format(duration=f"{dur_s:.1f}"))
                    break
                elif total_ms >= MAX_UTTERANCE_MS:
                    print(_("\n Max recording length"))
                    break
            else:
                if rms > threshold:
                    is_speaking = True
                    speech_ms = AUDIO_FRAME_MS
                    silence_ms = 0
                    audio_buffer.extend(chunk)
            total_ms += AUDIO_FRAME_MS

    except KeyboardInterrupt:
        print(_("\nRecording stopped"))
        audio_buffer = None
    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

    if audio_buffer and len(audio_buffer) > 1000:
        return bytes(audio_buffer), rate, channels
    return None, None, None

SPINNER_CHARS = "-\\|/"

# noinspection PyBroadException
def capture_while_key_held(key=PTT_KEY, timeout_seconds=30, stop_button=None):
    """Record audio while `key` is held down (push-to-talk). Returns (bytes, rate, channels) or (None, None, None)."""
    status_message = _("Hold {key} down to record, release to stop...").format(key=get_ptt_key_name())
    opened = _begin_capture(status_message)
    if opened is None:
        return None, None, None
    stream, rate, channels, _first_chunk, frame_samples, bytes_per_sample = opened
    audio_buffer = bytearray()

    key_pressed = threading.Event()
    key_released = threading.Event()

    def on_press(k):
        if k == key and not key_pressed.is_set():
            key_pressed.set()

    def on_release(k):
        if k == key and key_pressed.is_set():
            key_released.set()
            return False  # stop the listener
        return None

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    is_recording = False
    total_ms = 0
    spinner_i = 0
    start = time.time()

    try:
        logger.debug("Waiting for key press...")
        while True:
            if _stop_requested(stop_button):
                raise KeyboardInterrupt

            if not is_recording:
                if key_pressed.is_set():
                    is_recording = True
                    print(_("Recording... (release key to stop)"))
                    logger.info("Recording... (release key to stop)")
                elif (time.time() - start) > timeout_seconds:
                    return None, None, None

            raw_chunk, _overflowed = stream.read(frame_samples)
            chunk = bytes(raw_chunk)
            if not chunk:
                break

            if is_recording:
                audio_buffer.extend(chunk)
                total_ms += AUDIO_FRAME_MS

                print(f"\r{SPINNER_CHARS[spinner_i % len(SPINNER_CHARS)]}", end="", flush=True)
                spinner_i += 1

                if key_released.is_set():
                    dur_s = len(audio_buffer) / (rate * bytes_per_sample * channels)
                    print(_("\r Recorded {duration}s        ").format(duration=f"{dur_s:.1f}"))
                    break
                if total_ms >= MAX_UTTERANCE_MS:
                    print(_("\r Max recording length        "))
                    break

    except KeyboardInterrupt:
        print(_("\nRecording stopped"))
        audio_buffer = None
    finally:
        listener.stop()
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

    if audio_buffer and len(audio_buffer) > 1000:
        return bytes(audio_buffer), rate, channels
    return None, None, None

def _resample_for_device(frames, rate, channels, device):
    """Resample raw 16-bit PCM `frames` to the output device's native rate if
    they differ. Some ALSA output devices (e.g. a Raspberry Pi's default hw
    device) don't resample on their own, unlike CoreAudio/WASAPI/PulseAudio --
    playing audio at the wrong rate either crashes with paInvalidSampleRate or
    changes speed/pitch."""
    try:
        info = sd.query_devices(device=device) if device is not None else sd.query_devices(kind="output")
        device_rate = int(round(info["default_samplerate"]))
    except Exception as e:
        logger.warning(f"Could not query output device sample rate, skipping resample check: {e}")
        return frames, rate
    if device_rate == rate:
        return frames, rate

    from pydub import AudioSegment
    start = time.perf_counter()
    segment = AudioSegment(frames, frame_rate=rate, sample_width=2, channels=channels)
    segment = segment.set_frame_rate(device_rate)
    elapsed_ms = (time.perf_counter() - start) * 1000
    audio_duration_ms = len(frames) / 2 / channels / rate * 1000
    logger.debug(
        f"Resampled audio {rate} Hz -> {device_rate} Hz "
        f"({audio_duration_ms:.0f} ms of audio) in {elapsed_ms:.1f} ms"
    )
    return segment.raw_data, device_rate

def playback_wav(path, device=None):
    """Play a WAV file through sounddevice, honoring `device` -- unlike shelling
    out to aplay/afplay/winsound, this way we can target a specific output
    device (e.g. a USB speaker) instead of whatever the OS default happens to be."""
    with wave.open(str(path), 'rb') as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    frames, rate = _resample_for_device(frames, rate, channels, device)
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    try:
        sd.play(audio, rate, device=device)
        sd.wait()
    except Exception as e:
        logger.error(f"Playback error: {e}")

NORMALIZE_TARGET_PEAK = 0.9
NORMALIZE_MAX_GAIN = 10.0

def _normalize_pcm16(audio_data):
    """Scale int16 PCM samples so the loudest peak hits NORMALIZE_TARGET_PEAK of full scale."""
    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(samples))
    if peak == 0:
        return audio_data
    gain = min((NORMALIZE_TARGET_PEAK * 32767) / peak, NORMALIZE_MAX_GAIN)
    samples = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    return samples.tobytes()

def write_wav(audio_data, filepath, sample_rate, channels):
    audio_data = _normalize_pcm16(audio_data)
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    logger.debug(f"Audio saved to {filepath.resolve()}")

def run_transcription(whisper_model, audio_path):
    logger.debug("Transcribing...")
    language = LANG_TARGET if LANG_TARGET and LANG_TARGET.lower() != "multi" else None

    if language and NATIVE_LANG_TARGET and NATIVE_LANG_TARGET != language:
        try:
            language = _detect_constrained_language(
                whisper_model, audio_path,
                allowed={language, NATIVE_LANG_TARGET},
                fallback=LANG_TARGET
            )
            logger.debug(f"Constrained detection picked '{language}'")
        except Exception as e:
            logger.warning(f"Constrained language detection failed, using target language: {e}")
            language = LANG_TARGET

    try:
        segments, info = whisper_model.transcribe(
            str(audio_path),
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            )
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip() if text else None
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        print(_("\nTranscription error: {error}").format(error=e))
        return None

# noinspection PyBroadException
def capture_fixed_duration(seconds=3, stop_button=None):
    opened = _begin_capture(_("Recording ~{seconds}s for test...").format(seconds=seconds))
    if opened is None:
        return None, None, None
    stream, rate, channels, first_chunk, frame_samples, _bytes_per_sample = opened

    total_frames = int((seconds * 1000) / AUDIO_FRAME_MS)
    buf = bytearray()
    if first_chunk:
        buf.extend(first_chunk)

    try:
        for _i in range(total_frames - (1 if first_chunk else 0)):
            if _stop_requested(stop_button):
                break
            raw_chunk, _overflowed = stream.read(frame_samples)
            chunk = bytes(raw_chunk)
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        try:
            stream.stop(); stream.close()
        except Exception:
            pass

    return (bytes(buf), rate, channels) if buf else (None, None, None)


def select_mic_target():
    devices_available = sd.query_devices()
    if len(devices_available) == 0:
        logger.error(f"No available input device found!")
        print(_("No available input device found!"))
        exit(_("Application terminated!"))

    for i, dev_available in enumerate(devices_available):
        if dev_available.get('max_input_channels') != 0:
            print(f"{i}: {dev_available.get('name')}")

    while True:
        selected_mic = int(input(_('Please select an input device: ')).lower())
        if sd.query_devices(device=selected_mic)['max_input_channels'] != 0:
           break
        print(_("Error! Please select a valid input device!"))
    return selected_mic

# noinspection PyBroadException
def get_mic_target_name(target):
    """Return the device name for a mic device index (e.g. one loaded from
    settings.py), or a placeholder if that index no longer exists -- devices
    can disappear between runs (unplugged, driver change, etc.)."""
    try:
        return sd.query_devices(device=int(target))['name']
    except Exception:
        return f"device {target} (not found)"

def record_speech(method="vad"):
    """Record and transcribe speech using either "vad" (voice activity detection) or "key" (push-to-talk)."""
    try:
        whisper_model = whisper_engine()
        stop_button = stop_button_handle()
        if method == "vad":
            audio_data, rate, ch = capture_until_silence(timeout_seconds=30, stop_button=stop_button)
        elif method == "key":
            audio_data, rate, ch = capture_while_key_held(PTT_KEY, timeout_seconds=30, stop_button=stop_button)
        else:
            raise ValueError(f"Unknown method '{method}', expected 'vad' or 'key'")

        if audio_data:
            write_wav(audio_data, TEMP_WAV, sample_rate=rate, channels=ch)
            user_text = run_transcription(whisper_model, TEMP_WAV)
            return user_text
    except Exception as e:
        logger.error(f"Error: {e}")
        print(_("\nError: {error}").format(error=e))


# ===== Main =====

def _resolve_lang_code(value):
    """Accept either a full language name (matched against languages.py's
    LANGUAGES, e.g. "hungarian") or an ISO 639-1 code Whisper already
    understands (e.g. "hu"), and return the short code Whisper needs."""
    try:
        return languages.get_language(value.lower())["code"]
    except ValueError:
        return value

# noinspection PyBroadException
def main():

    global MIC_DEVICE_TARGET, PLAYBACK_TARGET, LANG_TARGET, NATIVE_LANG_TARGET
    args = sys.argv[1:]

    if "--debug" in args:
        logger.setLevel(logging.DEBUG)

    if "--ui-lang" in args:
        try:
            set_ui_language(args[args.index("--ui-lang") + 1])
        except IndexError:
            print(_("Usage: --ui-lang <language-code>"))

    if "--whisper-model" in args:
        try:
            set_whisper_model(args[args.index("--whisper-model") + 1])
        except IndexError:
            print(_("Usage: --whisper-model <model-name-or-path>"))

    if "--mic-device" in args:
        try:
            MIC_DEVICE_TARGET = args[args.index("--mic-device") + 1]
        except Exception:
            print(_("Usage: --mic-device <source-id-or-name>"))

    if "--playback-target" in args:
        try:
            PLAYBACK_TARGET = args[args.index("--playback-target") + 1]
        except Exception:
            print(_("Usage: --playback-target <output-id-or-name>"))

    if "--lang-target" in args:
        try:
            LANG_TARGET = _resolve_lang_code(args[args.index("--lang-target") + 1])
        except Exception:
            print(_("Usage: --lang-target <language-code>"))

    if "--native-lang-target" in args:
        try:
            NATIVE_LANG_TARGET = _resolve_lang_code(args[args.index("--native-lang-target") + 1])
        except Exception:
            print(_("Usage: --native-lang-target <language-code>"))

    if _FASTER_WHISPER_LANGUAGE_CODES is not None:
        for flag, value in (("--lang-target", LANG_TARGET), ("--native-lang-target", NATIVE_LANG_TARGET)):
            if value and value.lower() != "multi" and value not in _FASTER_WHISPER_LANGUAGE_CODES:
                print(_("'{value}' ({flag}) is not a language Whisper understands.").format(value=value, flag=flag))
                print(_("Valid codes: {codes}").format(codes=", ".join(sorted(_FASTER_WHISPER_LANGUAGE_CODES))))
                sys.exit(1)

    def shutdown_handler(sig, frame):
        logger.debug(f"Signal received: {sig}, {frame} ")
        print(_("\n\nShutting down..."))
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    if len(args) > 0:
        if "--help" in args:
            print(_("STT Engine - USB Mic"))
            print(_("\nUsage: python3 stt_engine.py [--mic-device <id-or-name>] [--playback-target <id-or-name>] [--lang-target <language-code>] [--native-lang-target <language-code>] [--whisper-model <model-name-or-path>] [--test] [--list-devices] [--ui-lang <language-code>] [--debug]"))
            print(_("  --mic-device        Force a specific input device (index or name substring)"))
            print(_("  --playback-target   Force a specific output device for --test playback (index or name substring)"))
            print(_("  --lang-target       Force a specific transcription language, e.g. en/hu (default: auto-detect)"))
            print(_("  --native-lang-target  Also allow this language (e.g. to code-switch and ask for help), picked between it and --lang-target per utterance"))
            print(_("  --whisper-model     Whisper model name or path to use for transcription (default: env WHISPER_MODEL or 'small')"))
            print(_("  --list-devices      Show all available audio input/output devices"))
            print(_("  --test           Record ~3s and play back (quick audio sanity check)"))
            print(_("  --ui-lang        Language for this CLI's own text, e.g. en/hu (default: env UI_LANGUAGE or system locale)"))
            print(_("  --debug          Enable debug-level logging to logs/stt_engine.log"))
            sys.exit(0)
        elif "--list-devices" in args:
            hostapis = sd.query_hostapis()
            input_devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
            print(_("Input devices (usable with --mic-device <index>):"))
            if input_devices:
                for i, d in input_devices:
                    print(f"  {i}: {d['name']} [{hostapis[d['hostapi']]['name']}] ({d['max_input_channels']} in)")
            else:
                print(_("  (none found)"))
            print(_("\nDefault input:  {name}").format(name=sd.query_devices(kind='input')['name']))
            output_devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]
            print(_("\nOutput devices (usable with --playback-target <index>):"))
            if output_devices:
                for i, d in output_devices:
                    print(f"  {i}: {d['name']} [{hostapis[d['hostapi']]['name']}] ({d['max_output_channels']} out)")
            else:
                print(_("  (none found)"))
            print(_("\nDefault output: {name}").format(name=sd.query_devices(kind='output')['name']))
            sys.exit(0)
        elif args[0] == "--test" or "--test" in args:
            stop_button = stop_button_handle()
            data, rate, ch = capture_fixed_duration(seconds=3, stop_button=stop_button)
            if not data:
                print(_("No audio captured during test."))
                sys.exit(1)
            out = TEST_WAV
            write_wav(data, out, sample_rate=rate, channels=ch)
            print(_("Playing back test recording..."))
            playback_wav(out, device=_resolve_output_device_index(PLAYBACK_TARGET))
            print(_("Audio test complete!"))
            sys.exit(0)

    print(_("Loading speech recognition model..."))
    whisper_model = whisper_engine()
    stop_button = stop_button_handle()

    print(_("STT engine ready!"))
    print(_("Setup:"))
    print(_("- Microphone: Sounddevice default input"))
    print(_("- Stop: {stop_option}").format(
        stop_option=_("GPIO 22 button or Ctrl+C") if stop_button else _("Press Ctrl+C")
    ))
    if MIC_DEVICE_TARGET:
        print(_("- Mic target override: {target} - {name}").format(
            target=MIC_DEVICE_TARGET, name=sd.query_devices(device=int(MIC_DEVICE_TARGET))['name']
        ))
    print(_("- Language: {lang}").format(lang=LANG_TARGET if LANG_TARGET else _("auto-detect")))
    if LANG_TARGET and NATIVE_LANG_TARGET and NATIVE_LANG_TARGET != LANG_TARGET:
        print(_("- Also allowing: {lang} (native language code-switch)").format(lang=NATIVE_LANG_TARGET))

    # Ask user which variant should be used, when stt_engine is directly called

    print("")
    print(_("0 - Voice activity detection"))
    print(_("1 - Press and hold recording"))
    while True:
        input_method = int(input(_('Please select an input method: ')).lower())
        if input_method in [0,1]:
           break
        print(_("Error! Please select a valid input method!"))

    print(_("\nListening for speech...\n"))

    if input_method == 1:
        # STT with Press-And-Hold recording
        while True:
            try:
                if _stop_requested(stop_button):
                    print(_("\nStop button pressed"))
                    break

                audio_data, rate, ch = capture_while_key_held(PTT_KEY, timeout_seconds=30, stop_button=stop_button)
                if audio_data:
                    write_wav(audio_data, TEMP_WAV, sample_rate=rate, channels=ch)
                    user_text = run_transcription(whisper_model, TEMP_WAV)

                    if user_text:
                        print(colors.user(_("You said: \"{text}\"").format(text=user_text)))
                    else:
                        print(_("No speech detected in the captured audio\n"))
                else:
                    print(_("No speech detected.\n"))

            except KeyboardInterrupt:
                print(_("\n\nInterrupted by user"))
                break
            except Exception as e:
                print(_("\nError: {error}").format(error=e))

        print(_("\nBye!"))

    if input_method == 0:
        # STT with VAD recording
        while True:
            try:
                if _stop_requested(stop_button):
                    print(_("\nStop button pressed"))
                    break

                audio_data, rate, ch = capture_until_silence(timeout_seconds=30, stop_button=stop_button)

                if audio_data:
                    write_wav(audio_data, TEMP_WAV, sample_rate=rate, channels=ch)
                    user_text = run_transcription(whisper_model, TEMP_WAV)

                    if user_text:
                        print(colors.user(_("You said: \"{text}\"").format(text=user_text)))
                        print(_("Listening...\n"))
                    else:
                        print(_("No speech detected in the captured audio\n"))
                else:
                    print(_("No speech detected, still listening...\n"))
            except KeyboardInterrupt:
                print(_("\n\nInterrupted by user"))
                break
            except Exception as e:
                print(_("\nError: {error}").format(error=e))
        print(_("\nBye!"))

if __name__ == "__main__":
    main()
