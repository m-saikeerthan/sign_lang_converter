/**
 * ISL Detector — Web Frontend Logic (Refactored for New UI)
 */

// ─── Elements ───────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Screens
const screenHome     = $('screen-home');
const screenDetector = $('screen-detector');
const screenFuture   = $('screen-future');
const screenAbout    = $('screen-about');
const screenMore     = $('screen-more');

// Navigation
const navBtnHome     = $('nav-btn-home');
const navBtnDetector = $('nav-btn-detector');
const navBtnFuture   = $('nav-btn-future');
const navBtnAbout    = $('nav-btn-about');
const navBtnMore     = $('nav-btn-more');
const navItems       = document.querySelectorAll('.nav-item');

// Home
const btnHeroStart   = $('btn-hero-start');

// Detector - Camera & Controls
const video            = $('video');
const canvas           = $('canvas');
const ctx              = canvas.getContext('2d');
const cameraOverlay    = $('camera-overlay');
const annotatedOverlay = $('annotated-overlay');
const cameraStatus     = $('camera-status');
const statusDot        = $('status-dot');
const detectionStatus  = $('detection-status');
const btnCamera        = $('btn-camera');

// Detector - Subheader
const fpsDisplay       = $('fps-display');
const btnHandRight     = $('hand-right');
const btnHandLeft      = $('hand-left');
const btnClear         = $('btn-clear');

// Detector - Results
const predictedWord    = $('predicted-word');
const confidenceFill   = $('confidence-fill');
const confidenceText   = $('confidence-text');
const timerContainer   = $('timer-container');
const timerFill        = $('timer-fill');
const wordChips        = $('word-chips');
const outputCard       = $('output-card');
const correctedSentence = $('corrected-sentence');
const btnSpeak         = $('btn-speak');
const btnCopy          = $('btn-copy');
const toast            = $('toast');
const statusBadge      = $('status-badge');

// Modal Elements
const gestureModal = $('gesture-modal');
const modalClose   = $('modal-close');
const modalImage   = $('modal-image');
const modalWord    = $('modal-word');

// ─── State ──────────────────────────────────────────────────────────
let selectedHand = 'right';
let cameraActive = false;
let streamRef    = null;
let sendInterval = null;
let lastCorrectedSentence = '';
let frameCount    = 0;
let fpsStartTime  = Date.now();
let lastSentenceKey = '';

// Gesture gallery: word → snapshot
let wordSnapshots = {};

// ─── Socket ─────────────────────────────────────────────────────────
const socket = io();

socket.on('connect', () => {
    statusBadge.textContent = 'Connected';
    statusBadge.classList.remove('connecting');
});

socket.on('disconnect', () => {
    statusBadge.textContent = 'Disconnected';
    statusBadge.classList.add('connecting');
});

// ─── Navigation ─────────────────────────────────────────────────────
function switchScreen(screenId, activeNavBtn) {
    screenHome.classList.add('hidden');
    screenDetector.classList.add('hidden');
    if (screenFuture) screenFuture.classList.add('hidden');
    if (screenAbout) screenAbout.classList.add('hidden');
    if (screenMore) screenMore.classList.add('hidden');
    
    const targetScreen = $(screenId);
    if (targetScreen) targetScreen.classList.remove('hidden');

    navItems.forEach(item => item.classList.remove('active'));
    if (activeNavBtn) activeNavBtn.classList.add('active');

    if (screenId !== 'screen-detector' && cameraActive) {
        stopCamera();
    }
}

if (navBtnHome) navBtnHome.addEventListener('click', (e) => { e.preventDefault(); switchScreen('screen-home', navBtnHome); });
if (navBtnDetector) navBtnDetector.addEventListener('click', (e) => { e.preventDefault(); switchScreen('screen-detector', navBtnDetector); });
if (navBtnFuture) navBtnFuture.addEventListener('click', (e) => { e.preventDefault(); switchScreen('screen-future', navBtnFuture); });
if (navBtnAbout) navBtnAbout.addEventListener('click', (e) => { e.preventDefault(); switchScreen('screen-about', navBtnAbout); });
if (navBtnMore) navBtnMore.addEventListener('click', (e) => { e.preventDefault(); switchScreen('screen-more', navBtnMore); });

btnHeroStart.addEventListener('click', () => {
    switchScreen('screen-detector', navBtnDetector);
    if (!cameraActive) {
        startCamera();
    }
});

const btnHeroFuture = $('btn-hero-future');
if (btnHeroFuture) {
    btnHeroFuture.addEventListener('click', () => {
        switchScreen('screen-future', navBtnFuture);
    });
}

// ─── Hand Toggle ────────────────────────────────────────────────────
function setHandPreference(hand) {
    selectedHand = hand;
    btnHandRight.classList.toggle('active', hand === 'right');
    btnHandLeft.classList.toggle('active', hand === 'left');
    
    // Update mirror effect instantly if camera is active
    video.style.transform = hand === 'left' ? 'scaleX(1)' : 'scaleX(-1)';
    annotatedOverlay.style.transform = video.style.transform;
}

btnHandRight.addEventListener('click', () => setHandPreference('right'));
btnHandLeft.addEventListener('click', () => setHandPreference('left'));

// ─── Sign → Text Logic ──────────────────────────────────────────────

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
    cameraStatus.style.display = 'flex';
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
        if (data.status === 'confirmed') predictedWord.style.color = 'var(--green)';
        else if (data.status === 'holding') predictedWord.style.color = 'var(--accent-purple)';
        else predictedWord.style.color = 'var(--text-primary)';
    } else {
        predictedWord.textContent = '—';
        predictedWord.style.color = 'var(--text-primary)';
    }

    // Confidence
    const conf = Math.round(data.confidence * 100);
    confidenceFill.style.width = conf + '%';
    confidenceText.textContent = conf + '%';
    
    if (data.status === 'holding') confidenceFill.style.background = 'var(--accent-purple)';
    else if (data.status === 'confirmed') confidenceFill.style.background = 'var(--green)';
    else confidenceFill.style.background = 'var(--text-primary)';

    // Capture snapshot on word confirmation
    if (data.snapshot && data.status === 'confirmed' && data.word) {
        wordSnapshots[data.word] = data.snapshot;
    }

    // Update word chips
    const sentenceKey = (data.sentence || []).join('|');
    if (sentenceKey !== lastSentenceKey) {
        lastSentenceKey = sentenceKey;
        updateWordChips(data.sentence);
    }

    // No-hand timer
    if (!data.hand_detected && data.no_hand_seconds > 0 && data.sentence && data.sentence.length > 0) {
        timerContainer.style.display = 'block';
        const progress = Math.min(data.no_hand_seconds / 3, 1) * 100;
        timerFill.style.width = progress + '%';
    } else {
        timerContainer.style.display = 'none';
    }
});

// ─── Sentence Complete ──────────────────────────────────────────────
socket.on('sentence_complete', (data) => {
    lastCorrectedSentence = data.corrected;
    outputCard.style.display = 'block';
    
    // Make AI Corrected Sentence Clickable
    buildClickableSentence(data.corrected, data.sentence || []);
    
    timerContainer.style.display = 'none';
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

// ─── Camera Control ─────────────────────────────────────────────────
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
        video.play().catch(e => console.warn("video.play() prevented:", e));
        cameraActive = true;

        setHandPreference(selectedHand); // Apply mirroring

        cameraOverlay.style.display = 'none';
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
    cameraOverlay.style.display = 'flex';
    cameraStatus.style.display = 'none';
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
    outputCard.style.display = 'none';
    predictedWord.textContent = '—';
    predictedWord.style.color = 'var(--text-primary)';
    confidenceFill.style.width = '0%';
    confidenceText.textContent = '0%';
});

// ─── Export ─────────────────────────────────────────────────────────
btnCopy.addEventListener('click', () => {
    navigator.clipboard.writeText(lastCorrectedSentence).then(() => showToast('Copied to clipboard!'));
});

// ─── Speak ──────────────────────────────────────────────────────────
btnSpeak.addEventListener('click', () => {
    if (lastCorrectedSentence) speak(lastCorrectedSentence);
});

function speak(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
}

// ─── UI Helpers ─────────────────────────────────────────────────────
function updateWordChips(words) {
    if (!words || words.length === 0) {
        wordChips.innerHTML = '<span class="words-placeholder">Words will appear here...</span>';
        return;
    }
    wordChips.innerHTML = '';
    words.forEach((w) => {
        const chip = document.createElement('div');
        chip.className = 'word-chip';
        
        // Use snapshot thumbnail if exists
        const snap = wordSnapshots[w];
        if (snap) {
            const img = document.createElement('img');
            img.src = snap;
            img.alt = w;
            chip.appendChild(img);

            chip.addEventListener('click', () => {
                modalImage.src = snap;
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

// Modal Bindings
if(modalClose) modalClose.addEventListener('click', () => gestureModal.style.display = 'none');
if(gestureModal) gestureModal.addEventListener('click', (e) => {
    if (e.target === gestureModal) gestureModal.style.display = 'none';
});

function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 2500);
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
