import importlib
import os
import shutil
import tempfile

import whisper

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model


def _ensure_ffmpeg_available():
    """Ensure ffmpeg is discoverable for Whisper audio loading."""
    if shutil.which("ffmpeg"):
        return

    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        ffmpeg_shim = os.path.join(ffmpeg_dir, "ffmpeg.exe")

        if not os.path.exists(ffmpeg_shim):
            shutil.copyfile(ffmpeg_exe, ffmpeg_shim)

        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg is required for speech-to-text but was not found. "
            "Install ffmpeg or install 'imageio-ffmpeg' in this environment."
        ) from exc

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is still not available. Please install ffmpeg and ensure it is on PATH."
        )


def _normalize_audio_source(audio):
    """
    Accepts:
    - file path (str / os.PathLike)
    - bytes / bytearray
    - file-like object (e.g. Streamlit UploadedFile)
    Returns a tuple: (path_to_audio_file, should_cleanup)
    """
    if isinstance(audio, (str, os.PathLike)):
        return str(audio), False

    data = None

    if isinstance(audio, (bytes, bytearray)):
        data = bytes(audio)
    elif hasattr(audio, "getvalue"):
        data = audio.getvalue()
    elif hasattr(audio, "read"):
        # Fallback for generic file-like objects
        current_pos = None
        if hasattr(audio, "tell"):
            try:
                current_pos = audio.tell()
            except Exception:
                current_pos = None

        try:
            if hasattr(audio, "seek"):
                audio.seek(0)
            data = audio.read()
        finally:
            if current_pos is not None and hasattr(audio, "seek"):
                try:
                    audio.seek(current_pos)
                except Exception:
                    pass

    if not data:
        raise ValueError(
            "Unsupported audio input. Provide a file path, bytes, or an uploaded file."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        return tmp.name, True


def transcribe(audio):
    """Transcribe audio into text using OpenAI Whisper."""
    _ensure_ffmpeg_available()
    model = _get_model()
    audio_path, should_cleanup = _normalize_audio_source(audio)

    try:
        result = model.transcribe(audio_path, fp16=False)
        text = result.get("text", "") if isinstance(result, dict) else ""
        if not isinstance(text, str):
            text = str(text)
        return text.strip()
    finally:
        if should_cleanup and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass