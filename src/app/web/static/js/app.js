/* ==========================================
   MULTIMODAL CLASSIFIER - MAIN APPLICATION
   ========================================== */

// Configuration
const API_URL = 'http://localhost:8000';

// Global State
let webcamStream = null;
let webcamSocket = null;
let isWebcamActive = false;
let lastFrameTime = Date.now();
let frameCount = 0;
let currentImageFile = null;
let capturedPhotoBlob = null;

/* ==========================================
   SERVER STATUS MANAGEMENT
   ========================================== */

/**
 * Check server health and update status display
 */
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        document.getElementById('status').innerHTML = `
            Server connected | Model: <strong>${data.model_type}</strong> | Classes: ${data.classes.length}
        `;
    } catch (error) {
        document.getElementById('status').innerHTML = 'Connection error with server';
    }
}

/**
 * Switch between different AI models
 * @param {string} modelType - Type of model ('resnet' or 'mobilenet')
 * @param {HTMLElement} buttonElement - Button that triggered the switch
 */
async function switchModel(modelType, buttonElement) {
    try {
        // Show loading state
        buttonElement.innerHTML = '<span class="loading"></span>';
        buttonElement.disabled = true;
        
        const response = await fetch(`${API_URL}/api/model/switch?model_type=${modelType}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            // Update button states
            document.querySelectorAll('.model-btn').forEach(btn => {
                btn.classList.remove('active');
                btn.disabled = false;
            });
            buttonElement.classList.add('active');
            
            // Restore button text
            buttonElement.innerHTML = modelType === 'resnet' ? 'ResNet50' : 'MobileNetV2';
            
            // Update status with notification
            await showModelSwitchNotification(modelType);
        } else {
            throw new Error(data.error || 'Error changing model');
        }
    } catch (error) {
        console.error('Error changing model:', error);
        alert('Error changing model: ' + error.message);
        
        // Restore button on error
        buttonElement.innerHTML = modelType === 'resnet' ? 'ResNet50' : 'MobileNetV2';
        buttonElement.disabled = false;
    }
}

/**
 * Show temporary notification when model is switched
 * @param {string} modelType - Type of model that was switched to
 */
async function showModelSwitchNotification(modelType) {
    const status = document.getElementById('status');
    const modelName = modelType === 'resnet' ? 'ResNet50' : 'MobileNetV2';
    
    // Show success notification
    status.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    status.style.color = 'white';
    status.innerHTML = `Model changed to <strong>${modelName}</strong>`;
    
    // Return to normal style after 3 seconds
    setTimeout(() => {
        status.style.background = 'rgba(255,255,255,0.9)';
        status.style.color = 'inherit';
        checkServerStatus();
    }, 3000);
}

/* ==========================================
   IMAGE CLASSIFICATION
   ========================================== */

/**
 * Handle image file selection and preview
 * @param {Event} event - File input change event
 */
function handleImageLoad(event) {
    const file = event.target.files[0];
    if (!file) return;

    currentImageFile = file;

    // Show preview
    const preview = document.getElementById('previewImage');
    preview.src = URL.createObjectURL(file);
    preview.classList.remove('hidden');

    // Show classify button
    document.getElementById('imageActions').classList.remove('hidden');
    document.getElementById('imageResult').classList.add('hidden');
}

/**
 * Classify the currently loaded image
 */
async function classifyImage() {
    if (!currentImageFile) {
        alert('Please load an image first');
        return;
    }

    // Show loading state
    showLoadingState('imageResult', 'Classifying...');

    // Upload and predict
    const formData = new FormData();
    formData.append('file', currentImageFile);

    try {
        const response = await fetch(`${API_URL}/api/predict/image`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success) {
            showResult('imageResult', data);
        } else {
            throw new Error(data.error || 'Error classifying');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('imageResult', error.message);
    }
}

/* ==========================================
   WEBCAM MANAGEMENT
   ========================================== */

/**
 * Toggle webcam on/off
 */
async function toggleWebcam() {
    if (isWebcamActive) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

/**
 * Start webcam stream
 */
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('webcamVideo');
        video.srcObject = webcamStream;
        video.classList.remove('hidden');
        document.getElementById('webcamBtn').textContent = 'Stop Webcam';
        document.getElementById('captureBtn').classList.remove('hidden');
        isWebcamActive = true;

        // Clear previous capture
        clearCapturedPhoto();

    } catch (error) {
        console.error('Error accessing webcam:', error);
        alert('Could not access webcam');
    }
}

/**
 * Stop webcam stream
 */
function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
    if (webcamSocket) {
        webcamSocket.close();
    }
    document.getElementById('webcamVideo').classList.add('hidden');
    document.getElementById('captureBtn').classList.add('hidden');
    document.getElementById('webcamBtn').textContent = 'Start Webcam';
    isWebcamActive = false;
}

/**
 * Clear captured photo and related UI elements
 */
function clearCapturedPhoto() {
    document.getElementById('capturedImage').classList.add('hidden');
    document.getElementById('classifyCaptureBtn').classList.add('hidden');
    document.getElementById('webcamResult').classList.add('hidden');
    capturedPhotoBlob = null;
}

/* ==========================================
   PHOTO CAPTURE AND CLASSIFICATION
   ========================================== */

/**
 * Capture a photo from the webcam
 */
function capturePhoto() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('capturedCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Convert to blob
    canvas.toBlob((blob) => {
        capturedPhotoBlob = blob;
        
        // Show preview
        const capturedImg = document.getElementById('capturedImage');
        capturedImg.src = URL.createObjectURL(blob);
        capturedImg.classList.remove('hidden');

        // Show classify button
        document.getElementById('classifyCaptureBtn').classList.remove('hidden');
        document.getElementById('webcamResult').classList.add('hidden');
    }, 'image/jpeg', 0.95);
}

/**
 * Classify the captured photo
 */
async function classifyCapturedPhoto() {
    if (!capturedPhotoBlob) {
        alert('Please capture a photo first');
        return;
    }

    // Show loading state
    showLoadingState('webcamResult', 'Classifying...');

    // Upload and predict
    const formData = new FormData();
    formData.append('file', capturedPhotoBlob, 'captured.jpg');

    try {
        const response = await fetch(`${API_URL}/api/predict/image`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.success) {
            showResult('webcamResult', data);
        } else {
            throw new Error(data.error || 'Error classifying');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('webcamResult', error.message);
    }
}

/* ==========================================
   UI HELPER FUNCTIONS
   ========================================== */

/**
 * Show loading state in result area
 * @param {string} elementId - ID of the element to update
 * @param {string} message - Loading message to display
 */
function showLoadingState(elementId, message) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="loading"></div> ${message}`;
    element.classList.remove('hidden');
}

/**
 * Show error message in result area
 * @param {string} elementId - ID of the element to update
 * @param {string} errorMessage - Error message to display
 */
function showError(elementId, errorMessage) {
    const element = document.getElementById(elementId);
    element.innerHTML = `
        <div style="background: #ff4444; color: white; padding: 15px; border-radius: 10px;">
            Error: ${errorMessage}
        </div>
    `;
}

/**
 * Display classification results
 * @param {string} elementId - ID of the element to update
 * @param {Object} data - Classification result data
 */
function showResult(elementId, data) {
    const element = document.getElementById(elementId);
    
    let top3HTML = '';
    if (data.top3_predictions) {
        top3HTML = '<div class="top3"><strong>Top 3 Predictions:</strong>';
        data.top3_predictions.forEach(pred => {
            top3HTML += `
                <div class="top3-item">
                    <span>${pred.class}</span>
                    <span>${pred.confidence.toFixed(1)}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${pred.confidence}%"></div>
                </div>
            `;
        });
        top3HTML += '</div>';
    }

    element.innerHTML = `
        <div class="result-box">
            <div class="result-main">${data.predicted_class}</div>
            <div class="confidence">Confidence: ${data.confidence.toFixed(1)}%</div>
            ${top3HTML}
        </div>
    `;
    element.classList.remove('hidden');
}

/* ==========================================
   DRAG AND DROP FUNCTIONALITY
   ========================================== */

/**
 * Initialize drag and drop for image upload
 */
function initializeDragAndDrop() {
    const imageUploadArea = document.getElementById('imageUploadArea');
    
    imageUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageUploadArea.classList.add('dragover');
    });
    
    imageUploadArea.addEventListener('dragleave', () => {
        imageUploadArea.classList.remove('dragover');
    });
    
    imageUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        imageUploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            document.getElementById('imageInput').files = e.dataTransfer.files;
            handleImageLoad({ target: { files: [file] } });
        }
    });
}

/* ==========================================
   INITIALIZATION
   ========================================== */

/**
 * Initialize application when DOM is ready
 */
function initializeApp() {
    // Check server status immediately
    checkServerStatus();
    
    // Initialize drag and drop
    initializeDragAndDrop();
    
    // Set up periodic status check (every 10 seconds)
    setInterval(checkServerStatus, 10000);
}

// Start application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
