/* ==========================================
   MULTIMODAL CLASSIFIER - MAIN APPLICATION
   ========================================== */

// Configuration
const API_URL = "http://localhost:8000";

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
    document.getElementById("status").innerHTML = `
            Server connected | Model: <strong>${data.model_type}</strong> | Classes: ${data.classes.length}
        `;
  } catch (error) {
    document.getElementById("status").innerHTML =
      "Connection error with server";
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

    const response = await fetch(
      `${API_URL}/api/model/switch?model_type=${modelType}`,
      {
        method: "POST",
      }
    );
    const data = await response.json();

    if (data.success) {
      // Update button states
      document.querySelectorAll(".model-btn").forEach((btn) => {
        btn.classList.remove("active");
        btn.disabled = false;
      });
      buttonElement.classList.add("active");

      // Restore button text
      buttonElement.innerHTML =
        modelType === "resnet" ? "ResNet50" : "MobileNetV2";

      // Update status with notification
      await showModelSwitchNotification(modelType);
    } else {
      throw new Error(data.error || "Error changing model");
    }
  } catch (error) {
    console.error("Error changing model:", error);
    alert("Error changing model: " + error.message);

    // Restore button on error
    buttonElement.innerHTML =
      modelType === "resnet" ? "ResNet50" : "MobileNetV2";
    buttonElement.disabled = false;
  }
}

/**
 * Show temporary notification when model is switched
 * @param {string} modelType - Type of model that was switched to
 */
async function showModelSwitchNotification(modelType) {
  const status = document.getElementById("status");
  const modelName = modelType === "resnet" ? "ResNet50" : "MobileNetV2";

  // Show success notification
  status.style.background = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
  status.style.color = "white";
  status.innerHTML = `Model changed to <strong>${modelName}</strong>`;

  // Return to normal style after 3 seconds
  setTimeout(() => {
    status.style.background = "rgba(255,255,255,0.9)";
    status.style.color = "inherit";
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
  const preview = document.getElementById("previewImage");
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");

  // Show classify button
  document.getElementById("imageActions").classList.remove("hidden");
  document.getElementById("imageResult").classList.add("hidden");
}

/**
 * Classify the currently loaded image
 */
async function classifyImage() {
  if (!currentImageFile) {
    alert("Please load an image first");
    return;
  }

  // Show loading state
  showLoadingState("imageResult", "Classifying...");

  // Upload and predict
  const formData = new FormData();
  formData.append("file", currentImageFile);

  try {
    const response = await fetch(`${API_URL}/api/predict/image`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.success) {
      showResult("imageResult", data);
    } else {
      throw new Error(data.error || "Error classifying");
    }
  } catch (error) {
    console.error("Error:", error);
    showError("imageResult", error.message);
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
    const video = document.getElementById("webcamVideo");
    video.srcObject = webcamStream;
    video.classList.remove("hidden");
    document.getElementById("webcamBtn").textContent = "Stop Webcam";
    document.getElementById("captureBtn").classList.remove("hidden");
    isWebcamActive = true;

    // Clear previous capture
    clearCapturedPhoto();
  } catch (error) {
    console.error("Error accessing webcam:", error);
    alert("Could not access webcam");
  }
}

/**
 * Stop webcam stream
 */
function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
  }
  if (webcamSocket) {
    webcamSocket.close();
  }
  document.getElementById("webcamVideo").classList.add("hidden");
  document.getElementById("captureBtn").classList.add("hidden");
  document.getElementById("webcamBtn").textContent = "Start Webcam";
  isWebcamActive = false;
}

/**
 * Clear captured photo and related UI elements
 */
function clearCapturedPhoto() {
  document.getElementById("capturedImage").classList.add("hidden");
  document.getElementById("classifyCaptureBtn").classList.add("hidden");
  document.getElementById("webcamResult").classList.add("hidden");
  capturedPhotoBlob = null;
}

/* ==========================================
   PHOTO CAPTURE AND CLASSIFICATION
   ========================================== */

/**
 * Capture a photo from the webcam
 */
function capturePhoto() {
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("capturedCanvas");
  const ctx = canvas.getContext("2d");

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0);

  // Convert to blob
  canvas.toBlob(
    (blob) => {
      capturedPhotoBlob = blob;

      // Show preview
      const capturedImg = document.getElementById("capturedImage");
      capturedImg.src = URL.createObjectURL(blob);
      capturedImg.classList.remove("hidden");

      // Show classify button
      document.getElementById("classifyCaptureBtn").classList.remove("hidden");
      document.getElementById("webcamResult").classList.add("hidden");
    },
    "image/jpeg",
    0.95
  );
}

/**
 * Classify the captured photo
 */
async function classifyCapturedPhoto() {
  if (!capturedPhotoBlob) {
    alert("Please capture a photo first");
    return;
  }

  // Show loading state
  showLoadingState("webcamResult", "Classifying...");

  // Upload and predict
  const formData = new FormData();
  formData.append("file", capturedPhotoBlob, "captured.jpg");

  try {
    const response = await fetch(`${API_URL}/api/predict/image`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (data.success) {
      showResult("webcamResult", data);
    } else {
      throw new Error(data.error || "Error classifying");
    }
  } catch (error) {
    console.error("Error:", error);
    showError("webcamResult", error.message);
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
  element.classList.remove("hidden");
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

  let top3HTML = "";
  if (data.top3_predictions) {
    top3HTML = '<div class="top3"><strong>Top 3 Predictions:</strong>';
    data.top3_predictions.forEach((pred) => {
      top3HTML += `
                <div class="top3-item">
                    <span>${pred.class}</span>
                    <span>${pred.confidence.toFixed(1)}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${
                      pred.confidence
                    }%"></div>
                </div>
            `;
    });
    top3HTML += "</div>";
  }

  element.innerHTML = `
        <div class="result-box">
            <div class="result-main">${data.predicted_class}</div>
            <div class="confidence">Confidence: ${data.confidence.toFixed(
              1
            )}%</div>
            ${top3HTML}
        </div>
    `;
  element.classList.remove("hidden");
}

/* ==========================================
   DRAG AND DROP FUNCTIONALITY
   ========================================== */

/**
 * Initialize drag and drop for image upload
 */
function initializeDragAndDrop() {
  const imageUploadArea = document.getElementById("imageUploadArea");

  imageUploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    imageUploadArea.classList.add("dragover");
  });

  imageUploadArea.addEventListener("dragleave", () => {
    imageUploadArea.classList.remove("dragover");
  });

  imageUploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    imageUploadArea.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      document.getElementById("imageInput").files = e.dataTransfer.files;
      handleImageLoad({ target: { files: [file] } });
    }
  });
}

/* ==========================================
   AUDIO RECORDING & TRANSCRIPTION
   ========================================== */

// Audio State
let mediaRecorder = null;
let audioChunks = [];
let recordingStartTime = null;
let recordingTimer = null;
let recordedAudioBlob = null;

// Metadata Recording State
let metadataRecorder = null;
let metadataChunks = [];
let metadataStartTime = null;
let metadataTimer = null;
let metadataAudioBlob = null;

/**
 * Toggle audio recording on/off
 */
async function toggleRecording() {
  const recordBtn = document.getElementById("recordBtn");
  const audioStatus = document.getElementById("audioStatus");
  const visualizer = document.getElementById("audioVisualizer");
  const bars = visualizer.querySelectorAll(".bar");

  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = () => {
        recordedAudioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const audioUrl = URL.createObjectURL(recordedAudioBlob);

        // Show playback controls
        const playback = document.getElementById("audioPlayback");
        playback.src = audioUrl;
        document.getElementById("recordingInfo").classList.remove("hidden");

        // Show transcribe button
        document.getElementById("transcribeBtn").classList.remove("hidden");

        // Stop visualizer animation
        bars.forEach((bar) => bar.classList.remove("active"));

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      recordingStartTime = Date.now();

      // Update UI
      recordBtn.innerHTML = '<span class="icon">⏹</span> Stop Recording';
      recordBtn.classList.add("recording");
      audioStatus.textContent = "Recording...";

      // Start visualizer animation
      bars.forEach((bar) => bar.classList.add("active"));

      // Start timer
      recordingTimer = setInterval(updateRecordingDuration, 100);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please check permissions.");
    }
  } else {
    // Stop recording
    mediaRecorder.stop();
    clearInterval(recordingTimer);

    // Update UI
    recordBtn.innerHTML = '<span class="icon">⏺</span> Start Recording';
    recordBtn.classList.remove("recording");
    audioStatus.textContent = "Recording complete";
  }
}

/**
 * Update recording duration display
 */
function updateRecordingDuration() {
  const elapsed = Date.now() - recordingStartTime;
  const seconds = Math.floor(elapsed / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  const display = `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  document.getElementById("recordingDuration").textContent = display;
}

/**
 * Send recorded audio for transcription
 */
async function transcribeAudio() {
  if (!recordedAudioBlob) {
    alert("Please record audio first");
    return;
  }

  // Show loading state
  showLoadingState("transcriptionResult", "Transcribing audio...");

  try {
    // Create form data with the audio blob
    const formData = new FormData();
    formData.append("file", recordedAudioBlob, "audio.wav");

    // Send to backend
    const response = await fetch(`${API_URL}/api/audio/transcribe`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    if (data.success) {
      // Show result
      const resultDiv = document.getElementById("transcriptionResult");
      resultDiv.innerHTML = `
        <div class="transcription-box">
          <div class="transcription-text">
            <strong>Transcription:</strong><br>
            ${data.transcription}
          </div>
          <div class="transcription-meta">
            Duration: ${
              document.getElementById("recordingDuration").textContent
            }
          </div>
        </div>
      `;
      resultDiv.classList.remove("hidden");
    } else {
      throw new Error(data.error || "Transcription failed");
    }
  } catch (error) {
    console.error("Transcription error:", error);
    alert("Error transcribing audio: " + error.message);

    // Hide loading state
    document.getElementById("transcriptionResult").classList.add("hidden");
  }
}

/**
 * Toggle metadata section visibility
 */
function toggleMetadataSection() {
  const section = document.getElementById("metadataSection");
  const icon = document.getElementById("metadataToggleIcon");

  if (section.classList.contains("hidden")) {
    section.classList.remove("hidden");
    icon.textContent = "▼";
  } else {
    section.classList.add("hidden");
    icon.textContent = "▶";
  }
}

/**
 * Toggle metadata recording on/off
 */
async function toggleMetadataRecording() {
  const recordBtn = document.getElementById("metadataRecordBtn");
  const metadataStatus = document.getElementById("metadataStatus");
  const visualizer = document.getElementById("metadataVisualizer");
  const bars = visualizer.querySelectorAll(".bar-small");
  const textInput = document.getElementById("metadataText");

  if (!textInput.value.trim()) {
    alert("Please enter the expected transcription first");
    return;
  }

  if (!metadataRecorder || metadataRecorder.state === "inactive") {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      metadataRecorder = new MediaRecorder(stream);
      metadataChunks = [];

      metadataRecorder.ondataavailable = (event) => {
        metadataChunks.push(event.data);
      };

      metadataRecorder.onstop = () => {
        metadataAudioBlob = new Blob(metadataChunks, { type: "audio/wav" });
        const audioUrl = URL.createObjectURL(metadataAudioBlob);

        // Show playback controls
        const playback = document.getElementById("metadataPlayback");
        playback.src = audioUrl;
        document
          .getElementById("metadataRecordingInfo")
          .classList.remove("hidden");

        // Stop visualizer animation
        bars.forEach((bar) => bar.classList.remove("active"));

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      metadataRecorder.start();
      metadataStartTime = Date.now();

      // Update UI
      recordBtn.innerHTML = '<span class="icon">⏹</span> Stop Recording';
      recordBtn.classList.add("recording");
      metadataStatus.textContent = "Recording...";

      // Start visualizer animation
      bars.forEach((bar) => bar.classList.add("active"));

      // Start timer
      metadataTimer = setInterval(updateMetadataDuration, 100);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please check permissions.");
    }
  } else {
    // Stop recording
    metadataRecorder.stop();
    clearInterval(metadataTimer);

    // Update UI
    recordBtn.innerHTML = '<span class="icon">⏺</span> Record Training Sample';
    recordBtn.classList.remove("recording");
    metadataStatus.textContent = "Recording complete";
  }
}

/**
 * Update metadata recording duration display
 */
function updateMetadataDuration() {
  const elapsed = Date.now() - metadataStartTime;
  const seconds = Math.floor(elapsed / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  const display = `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  document.getElementById("metadataDuration").textContent = display;
}

/**
 * Save metadata recording to server
 */
async function saveMetadataRecording() {
  if (!metadataAudioBlob) {
    alert("Please record audio first");
    return;
  }

  const transcription = document.getElementById("metadataText").value.trim();
  if (!transcription) {
    alert("Please enter the expected transcription");
    return;
  }

  // Get custom duration from input
  const durationInput = document.getElementById("recordingDurationInput");
  const duration = durationInput
    ? parseFloat(durationInput.value) || 3.0
    : (Date.now() - metadataStartTime) / 1000;

  // Show loading state
  showLoadingState("metadataSaveResult", "Saving training sample...");

  try {
    // Create form data
    const formData = new FormData();
    formData.append("file", metadataAudioBlob, "training_sample.wav");
    formData.append("text", transcription);
    formData.append("duration", duration.toString());
    formData.append("split", "train");

    // Send to backend
    const response = await fetch(`${API_URL}/api/audio/save-training-sample`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      const resultDiv = document.getElementById("metadataSaveResult");
      resultDiv.innerHTML = `
            <div class="result-box" style="background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);">
                <p><strong>✓ Training sample saved successfully!</strong></p>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    Duration: ${data.duration.toFixed(2)}s<br>
                    Transcription: "${data.transcription}"<br>
                    Split: ${data.split}<br>
                    File: ${data.file_path}
                </p>
            </div>
        `;
      resultDiv.classList.remove("hidden");

      // Reset form after 3 seconds
      setTimeout(() => {
        document.getElementById("metadataText").value = "";
        document
          .getElementById("metadataRecordingInfo")
          .classList.add("hidden");
        resultDiv.classList.add("hidden");
        metadataAudioBlob = null;
      }, 3000);
    } else {
      throw new Error(data.error || "Failed to save training sample");
    }
  } catch (error) {
    console.error("Error saving training sample:", error);
    const resultDiv = document.getElementById("metadataSaveResult");
    resultDiv.innerHTML = `
            <div class="result-box" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                <p><strong>❌ Error saving training sample</strong></p>
                <p style="font-size: 0.9em; margin-top: 10px;">${error.message}</p>
            </div>
        `;
    resultDiv.classList.remove("hidden");
  }
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
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp);
} else {
  initializeApp();
}
