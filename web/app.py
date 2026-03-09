"""
ISL Web Application — Flask-SocketIO backend.
Receives webcam frames via WebSocket, processes with MediaPipe + TFLite,
returns real-time predictions, and handles grammar correction via Gemini.

Usage: python web/app.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import time
import base64
import webbrowser
import threading

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Add src to path for ISL module imports
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

print("[ISL Web] Loading ISL modules...")
sys.stdout.flush()

from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor

# ─── App Setup ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-detector-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── ISL Pipeline (initialized once at startup) ─────────────────────
print("[ISL Web] Initializing hand tracker...")
sys.stdout.flush()
tracker = HandTracker(static_mode=True)

print("[ISL Web] Initializing gesture classifier...")
sys.stdout.flush()
classifier = GestureClassifier()

print("[ISL Web] Initializing sentence processor...")
sys.stdout.flush()
sentence_processor = SentenceProcessor()

# ─── Session State ───────────────────────────────────────────────────
sentence = []
last_confirmed = ""
no_hand_since = None
NO_HAND_TIMEOUT = 10.0


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def on_connect():
    print("[ISL Web] Client connected", flush=True)
    emit('status', {'message': 'Connected to ISL Detector'})


@socketio.on('disconnect')
def on_disconnect():
    print("[ISL Web] Client disconnected", flush=True)


@socketio.on('frame')
def handle_frame(data):
    global sentence, last_confirmed, no_hand_since

    try:
        # Decode base64 JPEG frame from browser
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # Process with hand tracker
        landmarks, raw_hand = tracker.process(frame)

        if landmarks is not None:
            no_hand_since = None
            result = classifier.classify(landmarks)

            response = {
                'hand_detected': True,
                'word': result['word'],
                'confidence': result['confidence'],
                'status': result['status'],
                'hold_progress': result['hold_progress'],
                'sentence': sentence.copy()
            }

            # Handle confirmed word (no consecutive duplicates)
            if result['status'] == 'confirmed':
                word = result['word']
                if not sentence or sentence[-1] != word:
                    sentence.append(word)
                    response['sentence'] = sentence.copy()

            emit('prediction', response)

        else:
            # No hand detected — track timeout
            if no_hand_since is None:
                no_hand_since = time.time()
            elif time.time() - no_hand_since >= NO_HAND_TIMEOUT:
                if sentence:
                    # Grammar correct via Gemini
                    corrected = sentence_processor.correct_grammar(sentence)
                    raw = " ".join(sentence)
                    print(f"[ISL Web] Detected:  {raw}", flush=True)
                    print(f"[ISL Web] Sentence:  {corrected}", flush=True)

                    emit('sentence_complete', {
                        'raw': raw,
                        'corrected': corrected,
                        'sentence': sentence.copy()
                    })

                    sentence = []
                    last_confirmed = ""
                    classifier.reset()

                no_hand_since = None

            emit('prediction', {
                'hand_detected': False,
                'word': '',
                'confidence': 0,
                'status': 'no hand',
                'hold_progress': 0,
                'sentence': sentence.copy(),
                'no_hand_seconds': time.time() - no_hand_since if no_hand_since else 0
            })

    except Exception as e:
        print(f"[ISL Web] Error processing frame: {e}", flush=True)


@socketio.on('clear_sentence')
def handle_clear():
    global sentence, last_confirmed
    sentence = []
    last_confirmed = ""
    classifier.reset()
    emit('prediction', {
        'hand_detected': False,
        'word': '',
        'confidence': 0,
        'status': 'detecting',
        'hold_progress': 0,
        'sentence': []
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  ISL Sign Language Detector — Web UI")
    print("  Opening http://localhost:5000 in your browser...")
    print("=" * 50)
    sys.stdout.flush()

    # Auto-open browser after a short delay (server needs to start first)
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()

    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
