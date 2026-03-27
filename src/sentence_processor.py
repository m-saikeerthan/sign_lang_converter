"""
Sentence Processor — Grammar correction, Hindi translation via Gemini,
Text-to-Speech, and AI sign description generation.
"""

import os
import base64
import threading
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv

# ─── Configuration ───────────────────────────────────────────────────
load_dotenv()
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

HINDI_PROMPT = (
    "Translate the following English sentence to Hindi. "
    "Only output the Hindi translation in Devanagari script, nothing else.\n\n"
    "Sentence: {sentence}"
)

SIGN_DESCRIPTION_PROMPT = (
    "You are an Indian Sign Language (ISL) expert. "
    "For each of the following words, describe how to perform the ISL gesture. "
    "Give a clear, concise 2-3 bullet point description of the hand shape and movement for each. "
    "If you're unsure about the exact ISL sign, describe the most common sign language gesture. "
    "Format your response EXACTLY as:\n"
    "WORD1:\n- step1\n- step2\n\n"
    "WORD2:\n- step1\n- step2\n\n"
    "Words: {words}"
)


class SentenceProcessor:
    """Handles grammar correction, Hindi translation, TTS, and sign descriptions."""

    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self._text_model = genai.GenerativeModel("gemini-2.0-flash")
        print("[ISL] Gemini text model ready.")

        # TTS Setup
        self._tts_engine = pyttsx3.init()
        self._tts_engine.setProperty('rate', 150)
        self._tts_engine.setProperty('volume', 1.0)
        voices = self._tts_engine.getProperty('voices')
        if len(voices) > 1:
            self._tts_engine.setProperty('voice', voices[1].id)
        print("[ISL] Text-to-speech ready.")

        # Cache for sign descriptions
        self._sign_cache = {}

    def correct_grammar(self, words):
        """Send word list to Gemini for grammar correction."""
        raw_sentence = " ".join(words)
        try:
            prompt = GRAMMAR_PROMPT.format(words=raw_sentence)
            response = self._text_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[ISL] Grammar correction failed: {e}")
            return raw_sentence

    def translate_to_hindi(self, english_sentence):
        """Translate English sentence to Hindi via Gemini."""
        try:
            prompt = HINDI_PROMPT.format(sentence=english_sentence)
            response = self._text_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[ISL] Hindi translation failed: {e}")
            return ""

    def correct_and_translate(self, words):
        """Full pipeline: correct grammar → translate to Hindi."""
        corrected = self.correct_grammar(words)
        hindi = self.translate_to_hindi(corrected)
        return {
            'english': corrected,
            'hindi': hindi,
            'raw': " ".join(words)
        }

    def generate_sign_descriptions_batch(self, words):
        """
        Generate sign descriptions for a list of words in a SINGLE API call.
        Returns list of {word, description} dicts.
        """
        words_upper = [w.upper().strip() for w in words]

        # Check cache for all words
        all_cached = all(w in self._sign_cache for w in words_upper)
        if all_cached:
            return [{'word': w, 'description': self._sign_cache[w]} for w in words_upper]

        # Single API call for all uncached words
        uncached = [w for w in words_upper if w not in self._sign_cache]
        try:
            import time
            prompt = SIGN_DESCRIPTION_PROMPT.format(words=', '.join(uncached))
            response = self._text_model.generate_content(prompt)
            text = response.text.strip()

            # Parse response into per-word descriptions
            current_word = None
            current_lines = []
            parsed = {}

            for line in text.split('\n'):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Check if this is a word header (e.g., "HELLO:" or "**HELLO:**")
                clean = line_stripped.replace('*', '').replace(':', '').strip()
                if clean.upper() in uncached:
                    if current_word and current_lines:
                        parsed[current_word] = '\n'.join(current_lines)
                    current_word = clean.upper()
                    current_lines = []
                elif current_word and (line_stripped.startswith('-') or line_stripped.startswith('•')):
                    current_lines.append(line_stripped)

            if current_word and current_lines:
                parsed[current_word] = '\n'.join(current_lines)

            # Cache results
            for w in uncached:
                if w in parsed:
                    self._sign_cache[w] = parsed[w]
                else:
                    self._sign_cache[w] = f"- Open palm facing forward and make the sign for '{w}'"

            print(f"[ISL] Sign descriptions generated for: {list(parsed.keys())}")

        except Exception as e:
            print(f"[ISL] Batch sign description failed: {e}")
            # Provide a graceful fallback
            for w in uncached:
                self._sign_cache[w] = f"- Gesture description could not be generated (API limit). Try again shortly."

        return [{'word': w, 'description': self._sign_cache.get(w, '')} for w in words_upper]

    def speak(self, text):
        """Speak text aloud in a background thread."""
        def _speak():
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                print(f"[ISL] TTS error: {e}")
        thread = threading.Thread(target=_speak, daemon=True)
        thread.start()

    def process(self, words):
        """Full pipeline: correct → translate → print → speak."""
        result = self.correct_and_translate(words)
        print(f"[ISL] Detected:  {result['raw']}")
        print(f"[ISL] English:   {result['english']}")
        print(f"[ISL] Hindi:     {result['hindi']}")
        self.speak(result['english'])
        return result
