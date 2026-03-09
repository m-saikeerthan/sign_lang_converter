/**
 * ISL Detector — Web Frontend Logic
 * Handles camera, WebSocket communication, UI updates, and Web Speech API.
 */

// ─── Elements ───────────────────────────────────────────────────────
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const btnCamera = document.getElementById('btn-camera');
const btnClear = document.getElementById('btn-clear');
const btnSpeak = document.getElementById('btn-speak');
const cameraOverlay = document.getElementById('camera-overlay');
const statusBadge = document.getElementById('status-badge');
const statusDot = document.getElementById('status-dot');
const detectionStatus = document.getElementById('detection-status');
const fpsDisplay = document.getElementById('fps-display');
const predictedWord = document.getElementById('predicted-word');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-text');
const holdContainer = document.getElementById('hold-container');
const holdRingFill = document.getElementById('hold-ring-fill');
const holdText = document.getElementById('hold-text');
const wordChips = document.getElementById('word-chips');
const timerCard = document.getElementById('timer-card');
const timerFill = document.getElementById('timer-fill');
const timerText = document.getElementById('timer-text');
const outputCard = document.getElementById('output-card');
const correctedSentence = document.getElementById('corrected-sentence');
const historyCard = document.getElementById('history-card');
const historyList = document.getElementById('history-list');

// ─── State ──────────────────────────────────────────────────────────
let cameraActive = false;
let streamRef = null;
let sendInterval = null;
let lastCorrectedSentence = '';
let frameCount = 0;
let fpsStartTime = Date.now();

// ─── Socket Connection ─────────────────────────────────────────────
const socket = io();

socket.on('connect', () => {
    statusBadge.textContent = 'Connected';
    statusBadge.classList.add('connected');
});

socket.on('disconnect', () => {
    statusBadge.textContent = 'Disconnected';
    statusBadge.classList.remove('connected');
});

socket.on('status', (data) => {
    console.log('[ISL]', data.message);
});

// ─── Prediction Handler ─────────────────────────────────────────────
socket.on('prediction', (data) => {
    // FPS tracking
    frameCount++;
    const elapsed = (Date.now() - fpsStartTime) / 1000;
    if (elapsed >= 1) {
        fpsDisplay.textContent = `${Math.round(frameCount / elapsed)} FPS`;
        frameCount = 0;
        fpsStartTime = Date.now();
    }

    // Status dot
    statusDot.className = 'status-dot';
    if (data.hand_detected) {
        statusDot.classList.add(data.status);
        detectionStatus.textContent = capitalize(data.status);
    } else {
        detectionStatus.textContent = 'No Hand';
    }

    // Predicted word
    if (data.word) {
        predictedWord.textContent = data.word;
        predictedWord.className = 'predicted-word';
        if (data.status === 'confirmed') {
            predictedWord.classList.add('active');
        } else if (data.status === 'holding') {
            predictedWord.classList.add('holding');
        }
    } else {
        predictedWord.textContent = '—';
        predictedWord.className = 'predicted-word';
    }

    // Confidence bar
    const conf = Math.round(data.confidence * 100);
    confidenceFill.style.width = conf + '%';
    confidenceText.textContent = conf + '%';

    // Hold progress
    if (data.status === 'holding' && data.hold_progress > 0) {
        holdContainer.style.display = 'flex';
        const circumference = 188.5;
        const offset = circumference * (1 - data.hold_progress);
        holdRingFill.style.strokeDashoffset = offset;
        holdText.textContent = Math.round(data.hold_progress * 100) + '%';
    } else {
        holdContainer.style.display = 'none';
    }

    // Word chips (sentence)
    updateWordChips(data.sentence);

    // No-hand timer
    if (!data.hand_detected && data.no_hand_seconds > 0 && data.sentence && data.sentence.length > 0) {
        timerCard.style.display = 'block';
        const progress = Math.min(data.no_hand_seconds / 10, 1) * 100;
        timerFill.style.width = progress + '%';
        const remaining = Math.max(10 - Math.floor(data.no_hand_seconds), 0);
        timerText.textContent = `Sentence finalizes in ${remaining}s...`;
    } else {
        timerCard.style.display = 'none';
    }
});

// ─── Sentence Complete Handler ──────────────────────────────────────
socket.on('sentence_complete', (data) => {
    lastCorrectedSentence = data.corrected;

    // Show output card
    outputCard.style.display = 'block';
    correctedSentence.textContent = data.corrected;

    // Clear word chips
    updateWordChips([]);

    // Hide timer
    timerCard.style.display = 'none';

    // Add to history
    addHistoryItem(data.raw, data.corrected);

    // Speak it!
    speak(data.corrected);
});

// ─── Camera Controls ────────────────────────────────────────────────
btnCamera.addEventListener('click', () => {
    if (cameraActive) {
        stopCamera();
    } else {
        startCamera();
    }
});

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });

        streamRef = stream;
        video.srcObject = stream;
        cameraActive = true;

        cameraOverlay.classList.add('hidden');
        btnCamera.textContent = '⏹ Stop Camera';
        btnCamera.classList.add('active');

        // Set canvas size
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        // Start sending frames
        sendInterval = setInterval(captureAndSend, 100); // ~10 fps

    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera access denied. Please allow camera access and try again.');
    }
}

function stopCamera() {
    if (streamRef) {
        streamRef.getTracks().forEach(track => track.stop());
    }
    if (sendInterval) {
        clearInterval(sendInterval);
    }

    cameraActive = false;
    video.srcObject = null;
    cameraOverlay.classList.remove('hidden');
    btnCamera.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="13" r="4"/><path d="M9.5 2h5L17 5h3a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3l2.5-3z"/></svg> Start Camera`;
    btnCamera.classList.remove('active');
}

function captureAndSend() {
    if (!cameraActive || !video.videoWidth) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
    socket.emit('frame', { image: dataUrl });
}

// ─── Clear Button ───────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
    socket.emit('clear_sentence');
    updateWordChips([]);
    outputCard.style.display = 'none';
    predictedWord.textContent = '—';
    predictedWord.className = 'predicted-word';
    confidenceFill.style.width = '0%';
    confidenceText.textContent = '0%';
});

// ─── Speak Button ───────────────────────────────────────────────────
btnSpeak.addEventListener('click', () => {
    if (lastCorrectedSentence) {
        speak(lastCorrectedSentence);
    }
});

// ─── Web Speech API ─────────────────────────────────────────────────
function speak(text) {
    if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported');
        return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    // Try to use a good English voice
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v =>
        v.lang.startsWith('en') && v.name.includes('Google')
    ) || voices.find(v => v.lang.startsWith('en'));
    if (preferred) utterance.voice = preferred;

    window.speechSynthesis.speak(utterance);
}

// Load voices (needed for some browsers)
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
    };
}

// ─── UI Helpers ─────────────────────────────────────────────────────
function updateWordChips(words) {
    if (!words || words.length === 0) {
        wordChips.innerHTML = '<span class="word-placeholder">Words will appear here...</span>';
        return;
    }
    wordChips.innerHTML = words
        .map(w => `<span class="word-chip">${w}</span>`)
        .join('');
}

function addHistoryItem(raw, corrected) {
    historyCard.style.display = 'block';

    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
        <div>
            <div class="history-corrected">${corrected}</div>
            <div class="history-raw">Raw: ${raw}</div>
        </div>
    `;

    historyList.prepend(item);

    // Keep max 10 items
    while (historyList.children.length > 10) {
        historyList.removeChild(historyList.lastChild);
    }
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
