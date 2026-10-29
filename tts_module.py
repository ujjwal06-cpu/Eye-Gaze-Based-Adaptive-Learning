"""
Week 8 -- Text-to-Speech
==========================
Roadmap: Selected Request -> Intent -> Message -> Text-to-Speech -> Voice Output

Uses pyttsx3 because it's fully OFFLINE at runtime -- it drives your
OS's own speech engine (espeak on Linux, SAPI5 on Windows, NSSpeech on
Mac). No internet needed once installed (unlike gTTS, which calls a
Google API for every sentence -- a bad fit for a device someone
depends on to communicate).

This module runs unmodified in two environments:
  - On your own machine (with pyttsx3 + a system TTS engine installed):
    actually speaks the message out loud.
  - Anywhere that engine isn't available (like this sandbox): prints a
    clearly-labeled "[VOICE OUTPUT]" line instead of crashing.

To get REAL audio on your own machine:
    pip install pyttsx3
    # Linux also needs the system engine:
    sudo apt install espeak
"""

_engine = None
_engine_available = None  # cached after first check


def _get_engine():
    global _engine, _engine_available
    if _engine_available is not None:
        return _engine if _engine_available else None
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine_available = True
    except Exception:
        _engine = None
        _engine_available = False
    return _engine


def speak(text: str) -> bool:
    """Speak `text` aloud if a TTS engine is available.

    Returns:
        True if real audio was produced, False if it fell back to the
        printed placeholder (still fine for demos/logging -- just not
        actual sound).
    """
    engine = _get_engine()
    if engine is not None:
        engine.say(text)
        engine.runAndWait()
        return True
    print(f"[VOICE OUTPUT -- TTS engine unavailable here, printing instead]: {text}")
    return False


if __name__ == "__main__":
    spoke_for_real = speak("I need some water, please.")
    print(f"\n(Produced real audio: {spoke_for_real})")
