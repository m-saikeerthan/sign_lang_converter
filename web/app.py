"""
ISL Web Application — Flask-SocketIO backend.
Receives webcam frames via WebSocket, processes with MediaPipe + TFLite,
returns real-time predictions with annotated frames, handles grammar
correction + Hindi translation via Gemini, and text-to-sign image generation.

Usage: python web/app.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys
import time
import base64
import webbrowser
import threading

os.environ['PYTHONUNBUFFERED'] = '1'

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import cv2
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

print("[ISL Web] Loading ISL modules...")
sys.stdout.flush()

from config import NO_HAND_TIMEOUT
from hand_tracker import HandTracker
from gesture_classifier import GestureClassifier
from sentence_processor import SentenceProcessor

# ─── App Setup ───────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-detector-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    max_http_buffer_size=10 * 1024 * 1024)

# ─── ISL Pipeline ────────────────────────────────────────────────────
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
last_snapshot = None  # stores annotated frame on confirmation


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
    global sentence, last_confirmed, no_hand_since, last_snapshot

    try:
        # Decode base64 JPEG
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Process
        landmarks, raw_hands = tracker.process(frame)

        # Draw landmarks on frame
        annotated = frame.copy()
        tracker.draw(annotated, raw_hands)

        # Encode annotated frame
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')

        if landmarks is not None:
            no_hand_since = None
            result = classifier.classify(landmarks)

            response = {
                'hand_detected': True,
                'word': result['word'],
                'confidence': result['confidence'],
                'status': result['status'],
                'hold_progress': result['hold_progress'],
                'sentence': sentence.copy(),
                'annotated_frame': annotated_b64,
                'snapshot': None
            }

            # On confirmation: add word + capture snapshot
            if result['status'] == 'confirmed':
                word = result['word']
                if not sentence or sentence[-1] != word:
                    sentence.append(word)
                    response['sentence'] = sentence.copy()

                    # Capture snapshot thumbnail
                    thumb = cv2.resize(annotated, (160, 120))
                    _, tbuf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    response['snapshot'] = 'data:image/jpeg;base64,' + base64.b64encode(tbuf).decode('utf-8')

            emit('prediction', response)

        else:
            # No hand detected
            if no_hand_since is None:
                no_hand_since = time.time()
            elif time.time() - no_hand_since >= NO_HAND_TIMEOUT:
                if sentence:
                    result = sentence_processor.correct_and_translate(sentence)
                    print(f"[ISL Web] English: {result['english']}", flush=True)
                    print(f"[ISL Web] Hindi:   {result['hindi']}", flush=True)

                    emit('sentence_complete', {
                        'raw': result['raw'],
                        'corrected': result['english'],
                        'hindi': result['hindi'],
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
                'annotated_frame': annotated_b64,
                'no_hand_seconds': time.time() - no_hand_since if no_hand_since else 0,
                'snapshot': None
            })

    except Exception as e:
        print(f"[ISL Web] Error: {e}", flush=True)


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
        'sentence': [],
        'annotated_frame': None,
        'snapshot': None
    })


@socketio.on('text_to_sign')
def handle_text_to_sign(data):
    """Serve ISL sign images: dataset first, Gemini API fallback."""
    text = data.get('text', '').strip()
    if not text:
        emit('sign_results', {'words': [], 'error': 'No text provided'})
        return

    words = text.upper().split()
    print(f"[ISL Web] Text→Sign: {words}", flush=True)

    import glob
    results = []
    api_fallback_words = []

    for word in words:
        word_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'word_frames', word)
        images = sorted(glob.glob(os.path.join(word_dir, '*.jpg'))) + sorted(glob.glob(os.path.join(word_dir, '*.png')))

        if images:
            # Use middle image from dataset as representative
            img_path = images[len(images) // 2]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (300, 225))
                _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
                results.append({'word': word, 'image': img_b64, 'description': '', 'source': 'dataset'})
                print(f"  {word}: from dataset", flush=True)
                continue

        # No dataset image — need API fallback
        api_fallback_words.append(word)
        results.append({'word': word, 'image': '', 'description': '', 'source': 'api'})

    # Batch API call for words not in dataset
    if api_fallback_words:
        descriptions = sentence_processor.generate_sign_descriptions_batch(api_fallback_words)
        desc_map = {d['word']: d['description'] for d in descriptions}
        for r in results:
            if r['source'] == 'api':
                r['description'] = desc_map.get(r['word'], 'Description not available')

    emit('sign_results', {'words': results})


if __name__ == '__main__':
    print("=" * 50)
    print("  ISL Sign Language Detector — Web UI")
    print("  Opening http://localhost:5000 ...")
    print("=" * 50)
    sys.stdout.flush()

    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:5000')).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
