/**
 * ISL Detector — Web Frontend Logic
 * Setup screen, Sign→Text detection, Text→Sign (dataset + AI fallback),
 * gesture gallery with snapshots, sentence export, English TTS.
 */

// ─── Elements ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Screens
const setupScreen      = $('setup-screen');
const detectorScreen   = $('detector-screen');
const textToSignScreen = $('text-to-sign-screen');

// Setup
const modeCards    = document.querySelectorAll('.mode-card');
const handBtns     = document.querySelectorAll('.hand-btn');
const handPrefSec  = $('hand-pref-section');
const btnStart     = $('btn-start');

// Detector
const video            = $('video');
const canvas           = $('canvas');
const ctx              = canvas.getContext('2d');
const annotatedOverlay = $('annotated-overlay');
const btnCamera        = $('btn-camera');
const btnClear         = $('btn-clear');
const btnSpeak         = $('btn-speak');
const btnCopy          = $('btn-copy');
const btnDownload      = $('btn-download');
const btnBack          = $('btn-back');
const cameraOverlay    = $('camera-overlay');
const statusBadge      = $('status-badge');
const statusDot        = $('status-dot');
const detectionStatus  = $('detection-status');
const fpsDisplay       = $('fps-display');
const predictedWord    = $('predicted-word');
const confidenceFill   = $('confidence-fill');
const confidenceText   = $('confidence-text');
const holdContainer    = $('hold-container');
const holdRingFill     = $('hold-ring-fill');
const holdText         = $('hold-text');
const wordChips        = $('word-chips');
const timerCard        = $('timer-card');
const timerFill        = $('timer-fill');
const timerText        = $('timer-text');
const outputCard       = $('output-card');
const correctedSentence = $('corrected-sentence');
const hindiSentence    = $('hindi-sentence');
const clickHint        = $('click-hint');
const historyCard      = $('history-card');
const historyList      = $('history-list');
const toast            = $('toast');

// Text to Sign
const btnBackTts       = $('btn-back-tts');
const ttsInput         = $('tts-input');
const btnTranslate     = $('btn-translate');
const ttsSlideshow     = $('tts-slideshow');
const ttsGrid          = $('tts-grid');
const slideshowWord    = $('slideshow-word');
const slideshowProgress = $('slideshow-progress');
const slideshowLoading = $('slideshow-loading');
const btnPrev          = $('btn-prev');
const btnPlay          = $('btn-play');
const btnNext          = $('btn-next');
const speedSelect      = $('speed-select');
const gestureGrid      = $('gesture-grid');

// Modal
const gestureModal = $('gesture-modal');
const modalClose   = $('modal-close');
const modalImage   = $('modal-image');
const modalWord    = $('modal-word');

// ─── State ──────────────────────────────────────────────────────────
let selectedMode = 'sign-to-text';
let selectedHand = 'right';
let cameraActive = false;
let streamRef    = null;
let sendInterval = null;
let lastCorrectedSentence = '';
let frameCount    = 0;
let fpsStartTime  = Date.now();

// Gesture gallery: word → snapshot
let wordSnapshots = {};
let lastSentenceKey = '';

// Slideshow
let slideshowData    = [];
let slideshowIndex   = 0;
let slideshowPlaying = false;
let slideshowTimer   = null;

// History dedup
let lastHistoryRaw = '';

// ─── Socket ─────────────────────────────────────────────────────────
const socket = io();

socket.on('connect', () => {
    statusBadge.textContent = 'Connected';
    statusBadge.classList.add('connected');
    const b = $('tts-status-badge');
    if (b) { b.textContent = 'Connected'; b.classList.add('connected'); }
});

socket.on('disconnect', () => {
    statusBadge.textContent = 'Disconnected';
    statusBadge.classList.remove('connected');
});

socket.on('status', (data) => console.log('[ISL]', data.message));

// ═══════════ SETUP SCREEN ═══════════

modeCards.forEach(card => {
    card.addEventListener('click', () => {
        modeCards.forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        selectedMode = card.dataset.mode;

        // Show hand preference only for sign-to-text
        handPrefSec.style.display = selectedMode === 'sign-to-text' ? 'block' : 'none';
    });
});

handBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        handBtns.forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        selectedHand = btn.dataset.hand;
    });
});

btnStart.addEventListener('click', () => {
    setupScreen.classList.add('hidden');
    if (selectedMode === 'sign-to-text') {
        detectorScreen.classList.remove('hidden');
    } else {
        textToSignScreen.classList.remove('hidden');
    }
});

btnBack.addEventListener('click', () => {
    stopCamera();
    detectorScreen.classList.add('hidden');
    setupScreen.classList.remove('hidden');
});

btnBackTts.addEventListener('click', () => {
    stopSlideshow();
    textToSignScreen.classList.add('hidden');
    setupScreen.classList.remove('hidden');
});

// ═══════════ SIGN → TEXT MODE ═══════════

socket.on('prediction', (data) => {
    // FPS
    frameCount++;
    const elapsed = (Date.now() - fpsStartTime) / 1000;
    if (elapsed >= 1) {
        fpsDisplay.textContent = `${Math.round(frameCount / elapsed)} FPS`;
        frameCount = 0;
        fpsStartTime = Date.now();
    }

    // Annotated frame (with hand landmarks)
    if (data.annotated_frame && cameraActive) {
        annotatedOverlay.src = data.annotated_frame;
        annotatedOverlay.style.display = 'block';
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
        if (data.status === 'confirmed') predictedWord.classList.add('active');
        else if (data.status === 'holding') predictedWord.classList.add('holding');
    } else {
        predictedWord.textContent = '—';
        predictedWord.className = 'predicted-word';
    }

    // Confidence
    const conf = Math.round(data.confidence * 100);
    confidenceFill.style.width = conf + '%';
    confidenceText.textContent = conf + '%';

    // Hold progress
    if (data.status === 'holding' && data.hold_progress > 0) {
        holdContainer.style.display = 'flex';
        const circumference = 188.5;
        holdRingFill.style.strokeDashoffset = circumference * (1 - data.hold_progress);
        holdText.textContent = Math.round(data.hold_progress * 100) + '%';
    } else {
        holdContainer.style.display = 'none';
    }

    // Capture snapshot on word confirmation
    if (data.snapshot && data.status === 'confirmed' && data.word) {
        wordSnapshots[data.word] = data.snapshot;
    }

    // Update word chips ONLY when sentence changes (prevents blinking)
    const sentenceKey = (data.sentence || []).join('|');
    if (sentenceKey !== lastSentenceKey) {
        lastSentenceKey = sentenceKey;
        updateWordChips(data.sentence);
    }

    // No-hand timer (5s)
    if (!data.hand_detected && data.no_hand_seconds > 0 && data.sentence && data.sentence.length > 0) {
        timerCard.style.display = 'block';
        const progress = Math.min(data.no_hand_seconds / 5, 1) * 100;
        timerFill.style.width = progress + '%';
        const remaining = Math.max(5 - Math.floor(data.no_hand_seconds), 0);
        timerText.textContent = `Sentence finalizes in ${remaining}s...`;
    } else {
        timerCard.style.display = 'none';
    }
});

// ─── Sentence Complete ──────────────────────────────────────────────
socket.on('sentence_complete', (data) => {
    lastCorrectedSentence = data.corrected;

    outputCard.style.display = 'block';

    // Build clickable sentence — click word to see its gesture popup
    buildClickableSentence(data.corrected, data.sentence || []);

    // Hindi hidden (English only)
    if (hindiSentence) hindiSentence.style.display = 'none';

    // Show click hint if we have snapshots
    if (Object.keys(wordSnapshots).length > 0 && clickHint) {
        clickHint.style.display = 'block';
    }

    // Keep word chips visible but don't clear snapshots
    timerCard.style.display = 'none';

    // History (deduplicate)
    if (data.raw !== lastHistoryRaw) {
        lastHistoryRaw = data.raw;
        addHistoryItem(data.raw, data.corrected);
    }

    // Speak in English
    speak(data.corrected);
});

// ─── Build Clickable Sentence ───────────────────────────────────────
function buildClickableSentence(correctedText, rawWords) {
    correctedSentence.innerHTML = '';
    const words = correctedText.split(/\s+/);

    words.forEach((word, i) => {
        const span = document.createElement('span');
        span.textContent = word;

        const matchKey = findSnapshotMatch(word, rawWords);
        if (matchKey) {
            span.className = 'sentence-word clickable';
            span.title = `Click to see gesture for "${matchKey}"`;
            span.addEventListener('click', () => {
                modalImage.src = wordSnapshots[matchKey];
                modalImage.style.display = 'block';
                modalWord.textContent = matchKey;
                gestureModal.style.display = 'flex';
            });
        } else {
            span.className = 'sentence-word';
        }

        correctedSentence.appendChild(span);
        if (i < words.length - 1) {
            correctedSentence.appendChild(document.createTextNode(' '));
        }
    });
}

function findSnapshotMatch(correctedWord, rawWords) {
    const upper = correctedWord.toUpperCase().replace(/[^A-Z]/g, '');
    if (wordSnapshots[upper]) return upper;
    for (const raw of rawWords) {
        if (wordSnapshots[raw] && raw.toUpperCase().startsWith(upper.slice(0, 3))) {
            return raw;
        }
    }
    return null;
}

// ─── Camera ─────────────────────────────────────────────────────────
btnCamera.addEventListener('click', () => {
    if (cameraActive) stopCamera();
    else startCamera();
});

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        streamRef = stream;
        video.srcObject = stream;
        cameraActive = true;

        video.style.transform = selectedHand === 'left' ? 'scaleX(1)' : 'scaleX(-1)';
        annotatedOverlay.style.transform = video.style.transform;

        cameraOverlay.classList.add('hidden');
        btnCamera.textContent = '⏹ Stop Camera';
        btnCamera.classList.add('active');

        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        sendInterval = setInterval(captureAndSend, 100);
    } catch (err) {
        console.error('Camera error:', err);
        alert('Camera access denied. Please allow camera access and try again.');
    }
}

function stopCamera() {
    if (streamRef) streamRef.getTracks().forEach(t => t.stop());
    if (sendInterval) clearInterval(sendInterval);
    cameraActive = false;
    video.srcObject = null;
    annotatedOverlay.style.display = 'none';
    cameraOverlay.classList.remove('hidden');
    btnCamera.innerHTML = '📷 Start Camera';
    btnCamera.classList.remove('active');
}

function captureAndSend() {
    if (!cameraActive || !video.videoWidth) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    socket.emit('frame', { image: canvas.toDataURL('image/jpeg', 0.7) });
}

// ─── Clear ──────────────────────────────────────────────────────────
btnClear.addEventListener('click', () => {
    socket.emit('clear_sentence');
    lastSentenceKey = '';
    updateWordChips([]);
    wordSnapshots = {};
    outputCard.style.display = 'none';
    if (clickHint) clickHint.style.display = 'none';
    predictedWord.textContent = '—';
    predictedWord.className = 'predicted-word';
    confidenceFill.style.width = '0%';
    confidenceText.textContent = '0%';
});

// ─── Export ─────────────────────────────────────────────────────────
btnCopy.addEventListener('click', () => {
    navigator.clipboard.writeText(lastCorrectedSentence).then(() => showToast('Copied to clipboard!'));
});

btnDownload.addEventListener('click', () => {
    let text = `ISL Detection Result\n${'='.repeat(30)}\n\n`;
    text += `Sentence: ${lastCorrectedSentence}\n`;
    text += `\nTimestamp: ${new Date().toLocaleString()}\n`;

    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `isl_sentence_${Date.now()}.txt`;
    a.click();
    showToast('Downloaded!');
});

// ─── Speak (English only) ───────────────────────────────────────────
btnSpeak.addEventListener('click', () => {
    if (lastCorrectedSentence) speak(lastCorrectedSentence);
});

function speak(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 0.9;
    utterance.pitch = 1;
    utterance.volume = 1;

    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'))
                   || voices.find(v => v.lang.startsWith('en'));
    if (preferred) utterance.voice = preferred;

    window.speechSynthesis.speak(utterance);
}

if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

// ─── Word Chips (with thumbnails) ───────────────────────────────────
function updateWordChips(words) {
    if (!words || words.length === 0) {
        wordChips.innerHTML = '<span class="word-placeholder">Words will appear here...</span>';
        return;
    }

    wordChips.innerHTML = '';
    words.forEach((w) => {
        const chip = document.createElement('div');
        chip.className = 'word-chip';

        // Show snapshot thumbnail if available
        const snap = wordSnapshots[w];
        if (snap) {
            const img = document.createElement('img');
            img.src = snap;
            img.alt = w;
            chip.appendChild(img);

            // Click to enlarge in modal
            chip.style.cursor = 'pointer';
            chip.addEventListener('click', () => {
                modalImage.src = snap;
                modalImage.style.display = 'block';
                modalWord.textContent = w;
                gestureModal.style.display = 'flex';
            });
        }

        const span = document.createElement('span');
        span.textContent = w;
        chip.appendChild(span);
        wordChips.appendChild(chip);
    });
}

function addHistoryItem(raw, corrected) {
    historyCard.style.display = 'block';
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `<div class="history-corrected">${corrected}</div>
                      <div class="history-raw">Raw: ${raw}</div>`;
    historyList.prepend(item);
    while (historyList.children.length > 10) historyList.removeChild(historyList.lastChild);
}

function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Modal
modalClose.addEventListener('click', () => gestureModal.style.display = 'none');
gestureModal.addEventListener('click', (e) => {
    if (e.target === gestureModal) gestureModal.style.display = 'none';
});

// ═══════════ TEXT → SIGN MODE ═══════════

btnTranslate.addEventListener('click', () => {
    const text = ttsInput.value.trim();
    if (!text) return;

    ttsSlideshow.style.display = 'block';
    ttsGrid.style.display = 'none';
    slideshowLoading.style.display = 'flex';
    slideshowWord.textContent = 'Generating...';
    slideshowProgress.innerHTML = '';
    stopSlideshow();

    const b = $('tts-status-badge');
    if (b) { b.textContent = 'Generating...'; b.classList.remove('connected'); }

    socket.emit('text_to_sign', { text });
});

ttsInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') btnTranslate.click();
});

socket.on('sign_results', (data) => {
    const b = $('tts-status-badge');
    if (b) { b.textContent = 'Ready'; b.classList.add('connected'); }

    if (data.error) {
        showToast(data.error);
        slideshowLoading.style.display = 'none';
        return;
    }

    slideshowData = data.words;
    slideshowIndex = 0;
    slideshowLoading.style.display = 'none';

    if (slideshowData.length === 0) {
        showToast('No words to translate');
        return;
    }

    // Build progress dots
    slideshowProgress.innerHTML = '';
    slideshowData.forEach((_, i) => {
        const dot = document.createElement('div');
        dot.className = 'dot' + (i === 0 ? ' active' : '');
        slideshowProgress.appendChild(dot);
    });

    showSlide(0);

    // Build grid
    ttsGrid.style.display = 'block';
    gestureGrid.innerHTML = '';
    slideshowData.forEach((item) => {
        const card = document.createElement('div');
        card.className = 'gesture-card';

        const wordEl = document.createElement('div');
        wordEl.className = 'gesture-card-word';
        wordEl.textContent = item.word;
        card.appendChild(wordEl);

        if (item.image) {
            // Dataset image
            const img = document.createElement('img');
            img.src = item.image;
            img.alt = item.word;
            img.className = 'gesture-card-img';
            card.appendChild(img);

            const tag = document.createElement('div');
            tag.className = 'gesture-source-tag dataset';
            tag.textContent = '📸 From Dataset';
            card.appendChild(tag);
        } else if (item.description) {
            // API fallback description
            const descEl = document.createElement('div');
            descEl.className = 'gesture-card-desc';
            descEl.innerHTML = formatDescription(item.description);
            card.appendChild(descEl);

            const tag = document.createElement('div');
            tag.className = 'gesture-source-tag api';
            tag.textContent = '🤖 AI Generated';
            card.appendChild(tag);
        }

        gestureGrid.appendChild(card);
    });
});

function formatDescription(desc) {
    return desc
        .split('\n')
        .filter(line => line.trim())
        .map(line => {
            line = line.replace(/^[\s•\-\*]+/, '').trim();
            return `<div class="desc-step">👉 ${line}</div>`;
        })
        .join('');
}

function showSlide(index) {
    if (index < 0 || index >= slideshowData.length) return;
    slideshowIndex = index;

    const item = slideshowData[index];
    slideshowWord.textContent = item.word;

    const area = document.querySelector('.slideshow-image-area');

    if (item.image) {
        // Show dataset image
        area.innerHTML = `<img src="${item.image}" alt="${item.word}" class="slideshow-img" style="display:block;">`;
    } else {
        // Show description
        area.innerHTML = `
            <div class="slideshow-desc-card">
                <div class="slideshow-desc-word">${item.word}</div>
                <div class="slideshow-desc-text">${formatDescription(item.description)}</div>
                <div class="gesture-source-tag api" style="margin-top:12px;">🤖 AI Generated (not in dataset)</div>
            </div>
        `;
    }

    const dots = slideshowProgress.querySelectorAll('.dot');
    dots.forEach((d, i) => d.classList.toggle('active', i === index));
}

btnPrev.addEventListener('click', () => { stopSlideshow(); showSlide(slideshowIndex - 1); });
btnNext.addEventListener('click', () => { stopSlideshow(); showSlide(slideshowIndex + 1); });

btnPlay.addEventListener('click', () => {
    if (slideshowPlaying) stopSlideshow();
    else startSlideshow();
});

function startSlideshow() {
    slideshowPlaying = true;
    btnPlay.textContent = '⏸ Pause';
    const speed = parseInt(speedSelect.value);
    slideshowTimer = setInterval(() => {
        showSlide((slideshowIndex + 1) % slideshowData.length);
    }, speed);
}

function stopSlideshow() {
    slideshowPlaying = false;
    btnPlay.textContent = '⏯ Play';
    if (slideshowTimer) { clearInterval(slideshowTimer); slideshowTimer = null; }
}

speedSelect.addEventListener('change', () => {
    if (slideshowPlaying) { stopSlideshow(); startSlideshow(); }
});
