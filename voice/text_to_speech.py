from io import BytesIO

from gtts import gTTS


def speak_to_bytes(text, lang="en"):
    audio_buffer = BytesIO()
    tts = gTTS(text=text, lang=lang)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()


def speak(text):
    tts = gTTS(text)
    tts.save("response.mp3")