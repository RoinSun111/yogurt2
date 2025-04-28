// Camera.js - Handle camera input and processing

let webcamElement;
let webcamRunning = false;
let captureInterval;
let mediaStream = null;

document.addEventListener('DOMContentLoaded', function() {
    webcamElement = document.getElementById('webcam');
    const startCameraButton = document.getElementById('start-camera');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    
    // Start camera button event listener
    startCameraButton.addEventListener('click', function() {
        startCamera();
    });
    
    // Check if camera access should be automatic
    const autoStartCamera = localStorage.getItem('autoStartCamera') === 'true';
    if (autoStartCamera) {
        startCamera();
    }
});

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
    
    // Process frame every 6 seconds (10 frames per minute as specified)
    captureInterval = setInterval(captureAndProcessFrame, 6000);
    
    // Do an initial capture immediately
    captureAndProcessFrame();
}

// Capture a frame from the webcam and process it
function captureAndProcessFrame() {
    if (!webcamRunning) return;
    
    // Create a canvas element to capture the frame
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to match video
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    
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
        })
        .catch(error => {
            console.error('Error processing frame:', error);
        });
    }, 'image/jpeg', 0.8);
}

// Update the posture status based on processed frame data
function updatePostureStatusFromData(data) {
    const postureState = document.getElementById('posture-state');
    
    if (data.is_present) {
        if (data.posture === 'upright') {
            postureState.textContent = 'Upright';
            postureState.className = 'badge bg-success';
        } else if (data.posture === 'slouched') {
            postureState.textContent = 'Slouched';
            postureState.className = 'badge bg-warning';
        } else {
            postureState.textContent = 'Present';
            postureState.className = 'badge bg-primary';
        }
    } else {
        postureState.textContent = 'Not Present';
        postureState.className = 'badge bg-secondary';
    }
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
