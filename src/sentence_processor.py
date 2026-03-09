"""
Sentence Processor — Grammar correction via Gemini + Text-to-Speech.
Converts raw sign language word sequences into proper English sentences
and speaks them aloud.
"""

import os
import threading
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv

# ─── Configuration ───────────────────────────────────────────────────
load_dotenv()  # loads from .env file in project root
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found! "
        "Create a .env file in the project root with: GEMINI_API_KEY=your_key_here"
    )

GRAMMAR_PROMPT = (
    "You are a grammar assistant for a sign language translator. "
    "I will give you a sequence of detected sign language words. "
    "Convert them into a single grammatically correct, natural English sentence. "
    "Only output the corrected sentence, nothing else. "
    "Do not add extra meaning — just fix the grammar.\n\n"
    "Words: {words}"
)


class SentenceProcessor:
    """Handles grammar correction (Gemini) and text-to-speech (pyttsx3)."""

    def __init__(self):
        # ─── Gemini Setup ────────────────────────────────
        genai.configure(api_key=GEMINI_API_KEY)
        self._model = genai.GenerativeModel("gemini-2.0-flash")
        print("[ISL] Gemini grammar correction ready.")

        # ─── TTS Setup ───────────────────────────────────
        self._tts_engine = pyttsx3.init()
        self._tts_engine.setProperty('rate', 150)    # speech speed
        self._tts_engine.setProperty('volume', 1.0)   # max volume
        # Use a clear voice if available
        voices = self._tts_engine.getProperty('voices')
        if len(voices) > 1:
            self._tts_engine.setProperty('voice', voices[1].id)  # usually female/clearer
        print("[ISL] Text-to-speech ready.")

    def correct_grammar(self, words):
        """
        Send word list to Gemini for grammar correction.

        Args:
            words: list of detected sign language words, e.g. ["I", "HUNGRY", "FOOD"]

        Returns:
            str: grammatically corrected sentence, or fallback raw sentence on error.
        """
        raw_sentence = " ".join(words)

        try:
            prompt = GRAMMAR_PROMPT.format(words=raw_sentence)
            response = self._model.generate_content(prompt)
            corrected = response.text.strip()
            return corrected
        except Exception as e:
            print(f"[ISL] Grammar correction failed: {e}")
            return raw_sentence  # fallback to raw words

    def speak(self, text):
        """Speak the text aloud in a background thread (non-blocking)."""
        def _speak():
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                print(f"[ISL] TTS error: {e}")

        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def process(self, words):
        """
        Full pipeline: correct grammar → print → speak.

        Args:
            words: list of detected sign language words.

        Returns:
            str: the corrected sentence.
        """
        corrected = self.correct_grammar(words)
        raw = " ".join(words)

        print(f"[ISL] Detected:  {raw}")
        print(f"[ISL] Sentence:  {corrected}")

        self.speak(corrected)
        return corrected
