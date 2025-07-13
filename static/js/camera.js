// Camera.js - Handle camera input and processing with pose landmarks visualization

import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

let webcamElement;
let webcamRunning = false;
let captureInterval;
let mediaStream = null;
let isProcessing = false; // Prevent frame processing overlap
let poseLandmarker = null;
let canvasElement;
let canvasCtx;
let drawingUtils;
let showAnnotations = true;

document.addEventListener('DOMContentLoaded', function() {
    webcamElement = document.getElementById('webcam');
    canvasElement = document.getElementById('annotation-canvas');
    canvasCtx = canvasElement.getContext('2d');
    const startCameraButton = document.getElementById('start-camera');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const toggleAnnotations = document.getElementById('toggle-annotations');
    
    // Initialize MediaPipe PoseLandmarker
    initializePoseLandmarker();
    
    // Start camera button event listener
    startCameraButton.addEventListener('click', function() {
        startCamera();
    });
    
    // Toggle annotations checkbox
    toggleAnnotations.addEventListener('change', function() {
        showAnnotations = this.checked;
        if (!showAnnotations) {
            // Clear the canvas when annotations are disabled
            canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        }
    });
    
    // Check if camera access should be automatic
    const autoStartCamera = localStorage.getItem('autoStartCamera') === 'true';
    if (autoStartCamera) {
        startCamera();
    }
});

// Initialize MediaPipe PoseLandmarker
async function initializePoseLandmarker() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });
        drawingUtils = new DrawingUtils(canvasCtx);
        console.log('MediaPipe PoseLandmarker initialized');
    } catch (error) {
        console.error('Error initializing MediaPipe:', error);
    }
}

// Start the webcam and begin processing
function startCamera() {
    if (webcamRunning) return;
    
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    
    navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user'
        },
        audio: false
    })
    .then(stream => {
        mediaStream = stream;
        webcamElement.srcObject = stream;
        
        // Hide the placeholder when video starts playing
        webcamElement.onloadedmetadata = function() {
            webcamRunning = true;
            cameraPlaceholder.classList.add('d-none');
            
            // Save preference for auto-start
            localStorage.setItem('autoStartCamera', 'true');
            
            // Start capturing frames
            startFrameCapture();
        };
    })
    .catch(error => {
        console.error('Error accessing webcam:', error);
        showNotification('Camera Error', 'Could not access the webcam. Please check permissions.', 'error');
    });
}

// Stop the webcam
function stopCamera() {
    if (!webcamRunning) return;
    
    // Stop the capture interval
    if (captureInterval) {
        clearInterval(captureInterval);
    }
    
    // Stop all tracks on the stream
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
    
    // Reset the video element
    webcamElement.srcObject = null;
    webcamRunning = false;
    
    // Show placeholder
    document.getElementById('camera-placeholder').classList.remove('d-none');
}

// Start capturing frames from the webcam
function startFrameCapture() {
    // Clear any existing interval
    if (captureInterval) {
        clearInterval(captureInterval);
    }
    
    // Set canvas size to match video
    updateCanvasSize();
    
    // Process frames at 8 FPS for real-time posture detection (125ms intervals)
    captureInterval = setInterval(captureAndProcessFrame, 125);
    
    // Do an initial capture immediately
    captureAndProcessFrame();
}

// Update canvas size to match video dimensions
function updateCanvasSize() {
    if (webcamElement.videoWidth > 0 && webcamElement.videoHeight > 0) {
        canvasElement.width = webcamElement.videoWidth;
        canvasElement.height = webcamElement.videoHeight;
        
        // Update canvas style to match video display size
        const rect = webcamElement.getBoundingClientRect();
        canvasElement.style.width = rect.width + 'px';
        canvasElement.style.height = rect.height + 'px';
    }
}

// Capture a frame from the webcam and process it
function captureAndProcessFrame() {
    if (!webcamRunning || isProcessing) return;
    
    isProcessing = true;
    
    // Create a canvas element to capture the frame
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // Optimize canvas size for real-time processing
    const targetWidth = 640;
    const aspectRatio = webcamElement.videoHeight / webcamElement.videoWidth;
    canvas.width = targetWidth;
    canvas.height = targetWidth * aspectRatio;
    
    // Draw the current video frame to the canvas
    context.drawImage(webcamElement, 0, 0, canvas.width, canvas.height);
    
    // Convert the canvas to a blob (image file)
    canvas.toBlob(function(blob) {
        // Create a form data object to send the image
        const formData = new FormData();
        formData.append('frame', blob, 'webcam.jpg');
        
        // Send the frame to the server for processing
        fetch('/api/process_frame', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Frame processed:', data);
            
            // Update posture status based on response
            updatePostureStatusFromData(data);
            
            // Draw pose landmarks if available and annotations are enabled
            if (showAnnotations && poseLandmarker) {
                drawPoseLandmarks();
            }
        })
        .catch(error => {
            console.error('Error processing frame:', error);
        })
        .finally(() => {
            isProcessing = false; // Allow next frame processing
        });
    }, 'image/jpeg', 0.8);
}

// Draw pose landmarks on the annotation canvas
async function drawPoseLandmarks() {
    if (!poseLandmarker || !webcamElement || !canvasElement || !showAnnotations) return;
    
    try {
        // Clear the canvas
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Detect pose landmarks directly from video element
        const startTimeMs = performance.now();
        const results = await poseLandmarker.detectForVideo(webcamElement, startTimeMs);
        
        if (results.landmarks && results.landmarks.length > 0) {
            // Save canvas context
            canvasCtx.save();
            
            // Draw landmarks and connections for each detected pose
            for (const landmark of results.landmarks) {
                // Draw landmarks as circles
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 10, 2),
                    fillColor: 'rgba(255, 255, 255, 0.8)',
                    color: 'rgba(0, 255, 0, 0.9)'
                });
                
                // Draw connections between landmarks
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, {
                    color: 'rgba(0, 255, 0, 0.6)',
                    lineWidth: 2
                });
            }
            
            // Restore canvas context
            canvasCtx.restore();
        }
    } catch (error) {
        console.error('Error drawing pose landmarks:', error);
    }
}

// Update the posture status based on processed frame data
function updatePostureStatusFromData(data) {
    const postureState = document.getElementById('posture-state');
    
    if (data.is_present && data.posture && data.posture !== 'unknown') {
        // Format posture name for display
        const formattedPosture = formatPostureNameSimple(data.posture);
        postureState.textContent = formattedPosture;
        
        // Set badge color based on posture classification
        let badgeClass;
        switch(data.posture) {
            case 'sitting_straight':
            case 'standing':
            case 'upright_sitting':
            case 'standing_good':
            case 'upright':
                badgeClass = 'badge bg-success';
                break;
            case 'leaning_forward':
            case 'left_sitting':
            case 'right_sitting':
            case 'forward_leaning':
            case 'leaning_left':
            case 'leaning_right':
            case 'standing_poor':
                badgeClass = 'badge bg-warning';
                break;
            case 'hunching_over':
            case 'lying':
            case 'slouched_sitting':
            case 'lying_down':
            case 'slouched':
                badgeClass = 'badge bg-danger';
                break;
            default:
                badgeClass = 'badge bg-primary';
        }
        postureState.className = badgeClass;
    } else if (data.is_present) {
        postureState.textContent = 'Present';
        postureState.className = 'badge bg-primary';
    } else {
        postureState.textContent = 'Not Present';
        postureState.className = 'badge bg-secondary';
    }
}

// Simple posture name formatting for the basic status badge
function formatPostureNameSimple(posture) {
    if (!posture || posture === 'unknown') {
        return 'Unknown';
    }
    
    const postureMap = {
        // Original posture types
        'sitting_straight': 'Sitting Straight',
        'hunching_over': 'Hunching Over',
        'left_sitting': 'Leaning Left',
        'right_sitting': 'Leaning Right',
        'leaning_forward': 'Leaning Forward',
        'lying': 'Lying Down',
        'standing': 'Standing',
        'upright': 'Upright',
        'slouched': 'Slouched',
        
        // New advanced detector posture types
        'upright_sitting': 'Upright Sitting',
        'slouched_sitting': 'Slouched Sitting',
        'leaning_left': 'Leaning Left',
        'leaning_right': 'Leaning Right',
        'forward_leaning': 'Forward Leaning',
        'standing_good': 'Standing (Good)',
        'standing_poor': 'Standing (Poor)',
        'lying_down': 'Lying Down'
    };
    
    return postureMap[posture] || posture.charAt(0).toUpperCase() + posture.slice(1).replace(/_/g, ' ');
}

// Event listener for page visibility change
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, pause camera to save resources
        if (captureInterval) {
            clearInterval(captureInterval);
        }
    } else {
        // Page is visible again, resume camera
        if (webcamRunning) {
            startFrameCapture();
        }
    }
});
