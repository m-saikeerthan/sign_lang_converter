"""
Sentence Processor — Grammar correction, Hindi translation via Gemini,
Text-to-Speech, and AI sign description generation.
"""

import os
import base64
import threading
import pyttsx3
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ─── Configuration ───────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found! "
        "Create a .env file in the project root with: GEMINI_API_KEY=your_key_here"
    )

GRAMMAR_PROMPT = "Words: {words}\nOutput:"

GRAMMAR_SYSTEM = (
    "You are an Indian Sign Language (ISL) to English sentence converter. "
    "ISL follows Subject-Object-Verb order, which differs from English Subject-Verb-Object order. "
    "Reorder and convert ISL gloss words into a single natural, grammatically correct English sentence. "
    "Output ONLY the English sentence. No explanations. No extra content.\n\n"
    "Examples:\n"
    "Words: I YOU LIKE\nOutput: I like you.\n"
    "Words: ME HAPPY MEET YOU\nOutput: I am happy to meet you.\n"
    "Words: YOU GO WHERE\nOutput: Where are you going?\n"
    "Words: I HUNGRY FOOD WANT\nOutput: I am hungry and I want food.\n"
    "Words: HE SCHOOL GO\nOutput: He goes to school.\n"
    "Words: PLEASE HELP DEAF COMMUNICATE\nOutput: Please help the deaf communicate.\n"
)

# ─── Pronoun & Verb Normalization Maps ──────────────────────────────
_PRONOUN_MAP = {
    'ME': 'I', 'MY': 'my', 'MINE': 'mine',
    'YOU': 'you', 'YOUR': 'your',
    'HE': 'he', 'HIM': 'him', 'HIS': 'his',
    'SHE': 'her', 'HER': 'her',
    'WE': 'we', 'OUR': 'our', 'US': 'us',
    'THEY': 'they', 'THEM': 'them', 'THEIR': 'their',
    'IT': 'it', 'ITS': 'its',
}

# Common action verbs that are typically placed after Subject-Object in ISL
_VERB_HINTS = {
    'LIKE', 'LOVE', 'HATE', 'WANT', 'NEED', 'HELP', 'GO', 'COME',
    'SEE', 'KNOW', 'THINK', 'FEEL', 'GIVE', 'TAKE', 'MAKE', 'EAT',
    'DRINK', 'BUY', 'LEARN', 'TEACH', 'PLAY', 'WORK', 'TALK', 'MEET',
}


def _local_isl_to_english(words):
    """
    Smart local ISL→English converter (no API needed).
    Implements basic SOV→SVO reordering + pronoun normalization.
    """
    if not words:
        return ""

    words_upper = [w.upper() for w in words]

    # Identify subject (first pronoun/noun), verb (last verb-hint), object (rest)
    subject = None
    verb = None
    verb_idx = None
    obj_words = []

    # Find subject at the start
    if words_upper[0] in _PRONOUN_MAP:
        subject = _PRONOUN_MAP[words_upper[0]]
        remaining = words_upper[1:]
    else:
        subject = words_upper[0].lower()
        remaining = words_upper[1:]

    # Find a verb (look for last occurrence among hints)
    for i in range(len(remaining) - 1, -1, -1):
        if remaining[i] in _VERB_HINTS:
            verb = remaining[i].lower()
            verb_idx = i
            break

    if verb_idx is not None:
        obj_words = [w for j, w in enumerate(remaining) if j != verb_idx]
    else:
        # No clear verb found — just normalize and join
        obj_words = remaining

    # Normalize object words
    normalized_obj = []
    for w in obj_words:
        normalized_obj.append(_PRONOUN_MAP.get(w, w.lower()))

    # Build sentence
    parts = [subject]
    if verb:
        parts.append(verb)
    if normalized_obj:
        parts.extend(normalized_obj)

    sentence = " ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


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
        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._model_name = "gemini-2.0-flash"
            print("[ISL] Gemini Client ready.")
        except Exception as e:
            print(f"[ISL] Warning: Failed to init Gemini client: {e}")
            self._client = None
            
        # TTS Setup
        self._tts_engine = pyttsx3.init()
        self._tts_engine.setProperty('rate', 150)
        self._tts_engine.setProperty('volume', 1.0)
        voices = self._tts_engine.getProperty('voices')
        if len(voices) > 1:
            self._tts_engine.setProperty('voice', voices[1].id)
        print("[ISL] Text-to-speech ready.")

        # API Caches to prevent quota limits
        self._sign_cache = {}
        self._sentence_cache = {}

    def correct_grammar(self, words):
        """Send word list to Gemini for grammar correction with caching/fallback."""
        if not words:
            return ""

        raw_sentence = " ".join(words)
        cache_key = " ".join(w.upper() for w in words)

        # 1. Return from memory if cached
        if cache_key in self._sentence_cache:
            print(f"[ISL] Cache hit for: {cache_key}")
            return self._sentence_cache[cache_key]

        # 2. Try LLM with system prompt
        if self._client:
            try:
                prompt = GRAMMAR_PROMPT.format(words=raw_sentence)
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=GRAMMAR_SYSTEM,
                        temperature=0.1,
                        max_output_tokens=100,
                    )
                )
                corrected = response.text.strip()
                # Strip prefixes like "Output:" if model echoes it
                if corrected.lower().startswith("output:"):
                    corrected = corrected[7:].strip()
                self._sentence_cache[cache_key] = corrected
                return corrected
            except Exception as e:
                print(f"[ISL] Grammar API failed (quota/network): {e}")

        # 3. Smart local ISL→English fallback (SOV→SVO reordering)
        print(f"[ISL] Using local ISL reorder fallback.")
        fallback = _local_isl_to_english(words)
        self._sentence_cache[cache_key] = fallback
        return fallback

    def translate_to_hindi(self, english_sentence):
        """Translate English sentence to Hindi via Gemini."""
        if not self._client or not english_sentence:
            return ""
        try:
            prompt = HINDI_PROMPT.format(sentence=english_sentence)
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            return response.text.strip()
        except Exception as e:
            print(f"[ISL] Hindi translation failed/limits reached: {e}")
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
        if not self._client:
            for w in uncached:
                self._sign_cache[w] = "- Setup Gemini API Key for detailed ISL description."
            return [{'word': w, 'description': self._sign_cache[w]} for w in words_upper]
            
        try:
            import time
            prompt = SIGN_DESCRIPTION_PROMPT.format(words=', '.join(uncached))
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt
            )
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
