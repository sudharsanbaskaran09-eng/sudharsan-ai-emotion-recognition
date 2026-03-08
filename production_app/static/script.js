document.addEventListener('DOMContentLoaded', () => {

    // Toggle logic
    const btnWebcam = document.getElementById('btn-webcam');
    const btnUpload = document.getElementById('btn-upload');
    const webcamContainer = document.getElementById('webcam-container');
    const uploadContainer = document.getElementById('upload-container');

    // Stats layout
    const sysStatus = document.getElementById('system-status');
    const countPeople = document.getElementById('count-people');
    const countAnimals = document.getElementById('count-animals');
    const latencyVal = document.getElementById('latency-val');

    // Emotion Display
    const emotionLabel = document.getElementById('emotion-label');
    const confidenceLabel = document.getElementById('confidence-label');
    const ageLabel = document.getElementById('age-label');
    const emotionBars = document.getElementById('emotion-bars');

    // Upload components
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewArea = document.getElementById('preview-area');
    const imagePreview = document.getElementById('image-preview');
    const uploadCanvas = document.getElementById('upload-canvas');
    const resetBtn = document.getElementById('reset-btn');
    const analyzeBtn = document.getElementById('analyze-btn');

    // Webcam components
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const startWebcamBtn = document.getElementById('start-webcam-btn');
    const stopWebcamBtn = document.getElementById('stop-webcam-btn');

    let stream = null;
    let webcamInterval = null;
    const WEBCAM_FPS = 5;
    let isPredicting = false;
    let selectedFile = null;

    // --- VIEW TOGGLER ---
    btnWebcam.addEventListener('click', () => {
        btnWebcam.classList.add('active');
        btnUpload.classList.remove('active');
        webcamContainer.classList.remove('hidden');
        uploadContainer.classList.add('hidden');
    });

    btnUpload.addEventListener('click', () => {
        btnUpload.classList.add('active');
        btnWebcam.classList.remove('active');
        uploadContainer.classList.remove('hidden');
        webcamContainer.classList.add('hidden');
        stopWebcam();
    });

    // --- UPLOAD LOGIC ---
    browseBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
    dropZone.addEventListener('click', (e) => { if (e.target !== browseBtn) fileInput.click(); });
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = "#3b82f6"; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = "rgba(255,255,255,0.15)"; });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault(); dropZone.style.borderColor = "rgba(255,255,255,0.15)";
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) return alert('Target must be an image.');
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.onload = () => {
                uploadCanvas.width = imagePreview.width;
                uploadCanvas.height = imagePreview.height;
                uploadCanvas.getContext('2d').clearRect(0, 0, uploadCanvas.width, uploadCanvas.height);
            };
            dropZone.classList.add('hidden');
            previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener('click', () => {
        selectedFile = null; fileInput.value = '';
        previewArea.classList.add('hidden');
        dropZone.classList.remove('hidden');
        uploadCanvas.getContext('2d').clearRect(0, 0, uploadCanvas.width, uploadCanvas.height);
        resetDashboard();
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        analyzeBtn.disabled = true;
        sysStatus.textContent = "Processing Static Target...";

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const startTick = performance.now();
            const res = await fetch('/api/predict', { method: 'POST', body: formData });
            const data = await res.json();
            const endTick = performance.now();

            latencyVal.innerHTML = `${Math.round(endTick - startTick)}<span style="font-size:12px;color:var(--text-muted)">ms</span>`;

            if (data.success) {
                renderDashboard(data.data, uploadCanvas, imagePreview, false);
                sysStatus.textContent = "Analysis Complete";
            } else {
                alert('Inference architecture failed: ' + data.error);
                sysStatus.textContent = "Inference Failed";
            }
        } catch (error) {
            sysStatus.textContent = "Network Timeout";
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    // --- WEBCAM LOGIC ---
    startWebcamBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
            webcamVideo.srcObject = stream;

            webcamVideo.onloadedmetadata = () => {
                webcamVideo.play();
                webcamCanvas.width = webcamVideo.clientWidth;
                webcamCanvas.height = webcamVideo.clientHeight;

                startWebcamBtn.classList.add('hidden');
                stopWebcamBtn.classList.remove('hidden');

                sysStatus.textContent = "Tracking Active Stream...";
                sysStatus.style.color = "#34d399";
                emotionBars.innerHTML = '<li style="color:var(--text-muted); padding:1rem 0;">Synchronizing neuro-links...</li>';

                webcamInterval = setInterval(processWebcamFrame, 1000 / WEBCAM_FPS);
            };
        } catch (e) {
            alert("Unable to access camera hardware.");
        }
    });

    stopWebcamBtn.addEventListener('click', stopWebcam);

    function stopWebcam() {
        if (stream) stream.getTracks().forEach(t => t.stop());
        stream = null;
        if (webcamInterval) clearInterval(webcamInterval);
        webcamInterval = null;

        startWebcamBtn.classList.remove('hidden');
        stopWebcamBtn.classList.add('hidden');

        webcamCanvas.getContext('2d').clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
        resetDashboard();
    }

    async function processWebcamFrame() {
        if (isPredicting) return;
        isPredicting = true;

        const hiddenCanvas = document.createElement('canvas');
        hiddenCanvas.width = webcamVideo.videoWidth;
        hiddenCanvas.height = webcamVideo.videoHeight;
        hiddenCanvas.getContext('2d').drawImage(webcamVideo, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

        const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.8);
        const formData = new FormData();
        formData.append('image_base64', dataUrl);

        try {
            const sTick = performance.now();
            const res = await fetch('/api/predict_frame', { method: 'POST', body: formData });
            const data = await res.json();
            const eTick = performance.now();

            latencyVal.innerHTML = `${Math.round(eTick - sTick)}<span style="font-size:12px;color:var(--text-muted)">ms</span>`;

            if (data.success) {
                // Ensure layout scaling matches strictly to responsive design flexbox elements!
                webcamCanvas.width = webcamVideo.clientWidth;
                webcamCanvas.height = webcamVideo.clientHeight;
                renderDashboard(data.data, webcamCanvas, webcamVideo, true);
            }
        } catch (e) {
            console.error(e);
        } finally {
            isPredicting = false;
        }
    }


    // --- UNIFIED DASHBOARD RENDERING ROUTINE ---
    function renderDashboard(data, canvas, sourceMedia, isWebcam) {

        countPeople.textContent = data.counts.people;
        countAnimals.textContent = data.counts.animals || 0; // Upload payload supports animal counts

        // 1. Plot bounding boxes on the grid layout
        drawBoxes(canvas, sourceMedia, data, isWebcam);

        // 2. Update Primary Subject Display Block
        if (data.faces.length > 0) {
            const subject = data.faces[0];
            emotionLabel.innerHTML = subject.prediction;
            confidenceLabel.innerHTML = `Confidence: <b>${subject.confidence.toFixed(1)}%</b>`;
            ageLabel.innerHTML = subject.age;

            emotionBars.innerHTML = '';
            const scores = subject.all_scores;
            if (scores) {
                const sorted = Object.keys(scores).sort((a, b) => scores[b] - scores[a]);
                sorted.forEach(emo => {
                    const val = scores[emo];
                    const li = document.createElement('li');
                    li.className = 'emotion-item';
                    li.innerHTML = `
                        <span class="emotion-name">${emo}</span>
                        <div class="bar-track">
                             <div class="bar-fill" style="width: 0%"></div>
                        </div>
                        <span class="emotion-value">${val}%</span>
                    `;
                    emotionBars.appendChild(li);
                    setTimeout(() => { li.querySelector('.bar-fill').style.width = `${val}%`; }, 10);
                });
            }
        } else {
            emotionLabel.innerHTML = "N/A";
            confidenceLabel.innerHTML = "Tracking Empty Sequence";
            ageLabel.innerHTML = "--";
            emotionBars.innerHTML = '<li style="color:var(--text-muted); padding:1rem 0;">Awaiting frontal or profile focus...</li>';
        }
    }

    function resetDashboard() {
        sysStatus.textContent = "Awaiting Input";
        sysStatus.style.color = "var(--text-muted)";
        countPeople.textContent = "0";
        countAnimals.textContent = "0";
        latencyVal.innerHTML = '--<span style="font-size:12px;color:var(--text-muted)">ms</span>';
        emotionLabel.innerHTML = "N/A";
        confidenceLabel.innerHTML = "Confidence: ---%";
        ageLabel.innerHTML = "--";
        emotionBars.innerHTML = '<li style="text-align:center; color: var(--text-muted); padding: 1rem 0;">Awaiting neuro-visual data...</li>';
    }


    // --- CANVAS BOX DRAWING LAYER ---
    function drawBoxes(canvas, media, data, isWebcam) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        let iW = isWebcam ? media.videoWidth : media.naturalWidth;
        let iH = isWebcam ? media.videoHeight : media.naturalHeight;

        const scaleX = canvas.width / iW;
        const scaleY = canvas.height / iH;

        // Overlay YOLO Objects in the background hierarchy
        data.objects.forEach(obj => {
            const [x, y, w, h] = obj.box;
            const sx = x * scaleX, sy = y * scaleY, sw = w * scaleX, sh = h * scaleY;

            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(16, 185, 129, 0.4)";
            ctx.strokeRect(sx, sy, sw, sh);

            ctx.fillStyle = "rgba(16, 185, 129, 0.4)";
            ctx.font = "bold 12px Inter";
            const txt = `${obj.label} ${obj.confidence.toFixed(0)}%`;
            ctx.fillRect(sx, sy - 18, ctx.measureText(txt).width + 8, 18);
            ctx.fillStyle = "#fff";
            ctx.fillText(txt, sx + 4, sy - 4);
        });

        // Target Deep Learning Faces directly connected to BiLSTM outputs
        data.faces.forEach((face, idx) => {
            const [x, y, w, h] = face.box;
            const sx = x * scaleX, sy = y * scaleY, sw = w * scaleX, sh = h * scaleY;

            const boxCol = idx === 0 ? '#3b82f6' : '#ec4899';
            ctx.strokeStyle = boxCol; ctx.lineWidth = 3;
            ctx.strokeRect(sx, sy, sw, sh);

            const fontSize = Math.max(12, Math.min(sh * 0.15, 20));
            ctx.font = `bold ${fontSize}px Inter`;
            const txt = `${face.prediction}`;

            ctx.shadowColor = "rgba(0,0,0,0.5)"; ctx.shadowBlur = 10;
            ctx.fillStyle = boxCol;
            ctx.fillRect(sx, sy - fontSize - 10, ctx.measureText(txt).width + 16, fontSize + 10);

            ctx.shadowBlur = 0; ctx.fillStyle = "#fff";
            ctx.fillText(txt, sx + 8, sy - 8);
        });
    }

});
