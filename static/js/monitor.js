// Monitor.js - Handles visualization of posture and focus tracking

document.addEventListener('DOMContentLoaded', function() {
    // Initialize monitoring
    initializeMonitoring();
    
    // Setup event listeners
    document.getElementById('toggle-annotations').addEventListener('change', function() {
        const annotationCanvas = document.getElementById('annotation-canvas');
        annotationCanvas.style.display = this.checked ? 'block' : 'none';
    });
});

// Global tracking variables
let lastActivityState = 'unknown';
let lastPosture = 'unknown';
let lastFocused = false;
let timelineEvents = [];
const maxTimelineEvents = 10;
let landmarks = null;

// Initialize monitoring components
function initializeMonitoring() {
    console.log('Initializing monitoring view');
    
    // Test canvas accessibility
    testCanvas();
    
    // Initialize sound therapy controls
    initializeSoundTherapy();
    
    // Start camera with the existing camera module
    if (typeof startCamera === 'function') {
        startCamera(captureAndAnalyzeFrame);
    } else {
        console.error('Camera module not found');
    }
    
    // Event listener for screenshots
    document.addEventListener('keydown', function(e) {
        if (e.key === 's' || e.key === 'S') {
            captureScreenshot();
        }
    });
    
    // Initialize feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
    
    // Start periodic check for activity updates
    setInterval(updateActivityData, 1000);
}

// Test canvas functionality
function testCanvas() {
    console.log('Testing canvas...');
    const canvas = document.getElementById('annotation-canvas');
    const video = document.getElementById('webcam');
    
    if (!canvas) {
        console.error('Canvas not found!');
        return;
    }
    
    // Wait for video to load and set canvas size to match
    if (video && video.videoWidth > 0) {
        setupCanvas();
    } else {
        // Wait for video to load
        setTimeout(() => {
            setupCanvas();
        }, 2000);
    }
}

function setupCanvas() {
    const canvas = document.getElementById('annotation-canvas');
    const video = document.getElementById('webcam');
    
    if (!canvas || !video) return;
    
    // Get the actual displayed video dimensions
    const videoRect = video.getBoundingClientRect();
    
    // Set canvas size to match video display size
    canvas.width = videoRect.width;
    canvas.height = videoRect.height;
    
    console.log('Canvas sized to match video:', canvas.width, 'x', canvas.height);
}

// Initialize sound therapy controls
function initializeSoundTherapy() {
    console.log('Initializing sound therapy controls');
    
    // Wait for sound therapy system to be ready
    const initSoundSystem = () => {
        if (window.soundTherapy) {
            setupSoundControls();
        } else {
            setTimeout(initSoundSystem, 500);
        }
    };
    
    initSoundSystem();
}

function setupSoundControls() {
    const toggleSwitch = document.getElementById('toggle-sound-therapy');
    const volumeSlider = document.getElementById('sound-volume');
    
    if (toggleSwitch) {
        toggleSwitch.addEventListener('change', (e) => {
            if (e.target.checked) {
                window.soundTherapy.enable();
                console.log('Sound therapy enabled');
            } else {
                window.soundTherapy.disable();
                console.log('Sound therapy disabled');
            }
        });
    }
    
    if (volumeSlider) {
        volumeSlider.addEventListener('input', (e) => {
            const volume = e.target.value / 100;
            window.soundTherapy.setVolume(volume);
        });
    }
    
    console.log('Sound therapy controls setup complete');
}

// Capture and analyze frames
function captureAndAnalyzeFrame() {
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('output-canvas');
    const annotationCanvas = document.getElementById('annotation-canvas');
    
    if (!video || !canvas || !annotationCanvas) {
        console.error('Required elements not found');
        return;
    }
    
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(captureAndAnalyzeFrame);
        return;
    }
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    annotationCanvas.width = video.videoWidth;
    annotationCanvas.height = video.videoHeight;
    
    // Get the canvas context for drawing
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get frame data to send to server
    canvas.toBlob(function(blob) {
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');
        
        fetch('/api/process_frame', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Frame processed:', data);
            
            try {
                // Store landmarks for annotation if available
                if (data.pose_landmarks && data.pose_landmarks.length > 0) {
                    landmarks = data.pose_landmarks;
                    console.log('Stored landmarks:', landmarks.length);
                }
                
                console.log('About to update metrics...');
                
                // Update metrics display
                try {
                    updateMetrics(data);
                    console.log('Metrics updated successfully');
                } catch (error) {
                    console.error('Error in updateMetrics:', error);
                }
                
                // Draw annotations on the frame
                try {
                    console.log('About to draw annotations...');
                    drawAnnotations(data);
                    console.log('Annotations drawn successfully');
                } catch (error) {
                    console.error('Error in drawAnnotations:', error);
                }
                
                // Track events for timeline
                try {
                    trackEvents(data);
                    console.log('Events tracked successfully');
                } catch (error) {
                    console.error('Error in trackEvents:', error);
                }
                
                console.log('Frame processing completed');
            } catch (error) {
                console.error('Error in frame processing:', error);
            }
        })
        .catch(error => {
            console.error('Error processing frame:', error);
        })
        .finally(() => {
            // Continue capture loop
            requestAnimationFrame(captureAndAnalyzeFrame);
        });
    }, 'image/jpeg', 0.8);
}

// Update activity data from the API
function updateActivityData() {
    fetch('/api/activity_status')
        .then(response => response.json())
        .then(data => {
            // Update activity metrics
            updateActivityMetrics(data);
        })
        .catch(error => {
            console.error('Error fetching activity status:', error);
        });
}

// Update metrics on the page
function updateMetrics(data) {
    // Update focus and posture badges
    const focusedBadge = document.getElementById('focused-badge');
    const postureBadge = document.getElementById('posture-badge');
    
    if (focusedBadge && data.is_focused !== undefined) {
        focusedBadge.textContent = data.is_focused ? 'Focused' : 'Not Focused';
        focusedBadge.className = `badge ${data.is_focused ? 'bg-success' : 'bg-danger'}`;
    }
    
    if (postureBadge && data.posture) {
        postureBadge.textContent = data.posture.charAt(0).toUpperCase() + data.posture.slice(1);
        postureBadge.className = `badge ${data.posture === 'upright' ? 'bg-success' : data.posture === 'slouched' ? 'bg-warning' : 'bg-secondary'}`;
    }
    
    // Update posture angle
    const postureAngleValue = document.getElementById('posture-angle-value');
    const postureProgress = document.getElementById('posture-progress');
    const postureStatus = document.getElementById('posture-status');
    
    if (postureAngleValue && data.angle !== undefined) {
        postureAngleValue.textContent = `${Math.round(data.angle)}°`;
        
        if (postureProgress) {
            const anglePercent = Math.min(100, (data.angle / 30) * 100);
            postureProgress.style.width = `${anglePercent}%`;
            postureProgress.setAttribute('aria-valuenow', anglePercent);
            
            // Update color based on angle
            if (data.angle < 10) {
                postureProgress.className = 'progress-bar bg-success';
                if (postureStatus) postureStatus.textContent = 'Excellent';
            } else if (data.angle < 15) {
                postureProgress.className = 'progress-bar bg-info';
                if (postureStatus) postureStatus.textContent = 'Good';
            } else if (data.angle < 20) {
                postureProgress.className = 'progress-bar bg-warning';
                if (postureStatus) postureStatus.textContent = 'Fair';
            } else {
                postureProgress.className = 'progress-bar bg-danger';
                if (postureStatus) postureStatus.textContent = 'Poor';
            }
        }
    }
    
    // Update advanced posture metrics
    const postureQualityBadge = document.getElementById('posture-quality-badge');
    const postureDetailedMetrics = document.getElementById('posture-detailed-metrics');
    const postureFeedback = document.getElementById('posture-feedback');
    
    if (postureQualityBadge) {
        // Set badge text and color based on quality
        let qualityText, qualityClass;
        
        if (data.posture_quality && data.posture_quality !== 'unknown') {
            switch(data.posture_quality) {
                case 'excellent':
                    qualityText = 'Excellent';
                    qualityClass = 'bg-success';
                    break;
                case 'good':
                    qualityText = 'Good';
                    qualityClass = 'bg-info';
                    break;
                case 'fair':
                    qualityText = 'Fair';
                    qualityClass = 'bg-warning';
                    break;
                case 'poor':
                    qualityText = 'Poor';
                    qualityClass = 'bg-danger';
                    break;
                default:
                    qualityText = 'Analyzing';
                    qualityClass = 'bg-secondary';
            }
        } else {
            qualityText = 'Analyzing';
            qualityClass = 'bg-secondary';
        }
        
        postureQualityBadge.textContent = qualityText;
        postureQualityBadge.className = `badge ${qualityClass}`;
    }
    
    // Update detailed metrics if the container exists
    if (postureDetailedMetrics) {
        let metricsHTML = '';
        
        // Add neck angle metric if available
        if (data.neck_angle !== undefined) {
            const neckAngleClass = data.neck_angle > 20 ? 'text-danger' : 'text-success';
            metricsHTML += `
                <div class="d-flex justify-content-between mb-2">
                    <div class="metric-sublabel">Neck Angle</div>
                    <div class="${neckAngleClass}">${Math.round(data.neck_angle)}°</div>
                </div>
            `;
        }
        
        // Add shoulder alignment metric if available
        if (data.shoulder_alignment !== undefined) {
            const alignmentPercent = Math.round(data.shoulder_alignment * 100);
            let alignmentClass = 'text-success';
            if (alignmentPercent < 80) alignmentClass = 'text-danger';
            else if (alignmentPercent < 90) alignmentClass = 'text-warning';
            
            metricsHTML += `
                <div class="d-flex justify-content-between mb-2">
                    <div class="metric-sublabel">Shoulder Alignment</div>
                    <div class="${alignmentClass}">${alignmentPercent}%</div>
                </div>
            `;
        }
        
        // Add head position metric if available
        if (data.head_forward_position !== undefined) {
            const headPositionClass = data.head_forward_position > 0.1 ? 'text-warning' : 'text-success';
            const headPositionValue = Math.round(data.head_forward_position * 100);
            metricsHTML += `
                <div class="d-flex justify-content-between mb-2">
                    <div class="metric-sublabel">Head Position</div>
                    <div class="${headPositionClass}">${headPositionValue > 0 ? headPositionValue : 'Good'}</div>
                </div>
            `;
        }
        
        // Add symmetry score if available
        if (data.symmetry_score !== undefined) {
            const symmetryPercent = Math.round(data.symmetry_score * 100);
            let symmetryClass = 'text-success';
            if (symmetryPercent < 70) symmetryClass = 'text-danger';
            else if (symmetryPercent < 85) symmetryClass = 'text-warning';
            
            metricsHTML += `
                <div class="d-flex justify-content-between mb-2">
                    <div class="metric-sublabel">Body Symmetry</div>
                    <div class="${symmetryClass}">${symmetryPercent}%</div>
                </div>
            `;
        }
        
        // Update the container content
        postureDetailedMetrics.innerHTML = metricsHTML;
    }
    
    // Update posture feedback if available
    if (postureFeedback) {
        if (data.feedback) {
            postureFeedback.textContent = data.feedback;
        } else if (data.posture && data.posture !== 'unknown') {
            // Set feedback based on posture quality and basic posture
            if (data.posture_quality === 'excellent') {
                postureFeedback.textContent = 'Excellent posture! Keep it up!';
            } else if (data.posture_quality === 'good') {
                postureFeedback.textContent = 'Good posture detected. Stay consistent!';
            } else if (data.posture_quality === 'fair') {
                postureFeedback.textContent = 'Moderate posture. Consider minor adjustments.';
            } else if (data.posture_quality === 'poor') {
                postureFeedback.textContent = 'Poor posture detected. Please adjust your position.';
            } else if (data.posture === 'upright') {
                postureFeedback.textContent = 'Your posture looks good!';
            } else if (data.posture === 'slouched') {
                postureFeedback.textContent = 'Try to sit up straight to improve your posture.';
            } else {
                postureFeedback.textContent = 'Analyzing your posture...';
            }
        } else {
            postureFeedback.textContent = 'Position yourself in front of the camera for analysis.';
        }
    }
    
    // Update sound therapy based on posture data
    if (window.soundTherapy) {
        window.soundTherapy.updatePostureState(data);
    }
}

// Update activity metrics specifically
function updateActivityMetrics(data) {
    // Activity state
    const activityStateValue = document.getElementById('activity-state-value');
    const activitySubstateValue = document.getElementById('activity-substate-value');
    const activityProgress = document.getElementById('activity-progress');
    const timeInStateValue = document.getElementById('time-in-state-value');
    
    if (activityStateValue && data.activity_state) {
        // Format display text
        const displayState = data.activity_state.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        activityStateValue.textContent = displayState;
        
        // Set color based on state
        let stateClass = 'bg-secondary';
        if (data.activity_state === 'working') {
            stateClass = 'bg-success';
        } else if (data.activity_state === 'not_working') {
            stateClass = 'bg-warning';
        } else if (data.activity_state === 'distracted_by_others') {
            stateClass = 'bg-danger';
        } else if (data.activity_state === 'on_break') {
            stateClass = 'bg-info';
        } else if (data.activity_state === 'idle') {
            stateClass = 'bg-secondary';
        }
        
        if (activityProgress) {
            activityProgress.className = `progress-bar ${stateClass}`;
        }
    }
    
    if (activitySubstateValue) {
        if (data.working_substate) {
            const displaySubstate = data.working_substate.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            activitySubstateValue.textContent = displaySubstate;
        } else {
            activitySubstateValue.textContent = '-';
        }
    }
    
    if (timeInStateValue && data.time_in_state !== undefined) {
        // Format time
        if (data.time_in_state < 60) {
            timeInStateValue.textContent = `${data.time_in_state}s`;
        } else {
            const minutes = Math.floor(data.time_in_state / 60);
            const seconds = data.time_in_state % 60;
            timeInStateValue.textContent = `${minutes}m ${seconds}s`;
        }
        
        // Update progress based on recommended time for state
        if (activityProgress) {
            let maxDuration = 0;
            
            if (data.activity_state === 'working') {
                maxDuration = 25 * 60; // 25 minutes (pomodoro-like)
            } else if (data.activity_state === 'not_working') {
                maxDuration = 5 * 60; // 5 minutes
            } else if (data.activity_state === 'on_break') {
                maxDuration = 15 * 60; // 15 minutes
            } else if (data.activity_state === 'idle') {
                maxDuration = 2 * 60; // 2 minutes
            } else {
                maxDuration = 10 * 60; // 10 minutes default
            }
            
            const percentage = Math.min(100, (data.time_in_state / maxDuration) * 100);
            activityProgress.style.width = `${percentage}%`;
            activityProgress.setAttribute('aria-valuenow', percentage);
        }
    }
    
    // Head angle
    const headAngleValue = document.getElementById('head-angle-value');
    const headProgress = document.getElementById('head-progress');
    const headPosition = document.getElementById('head-position');
    
    if (headAngleValue && data.head_angle !== undefined) {
        headAngleValue.textContent = `${Math.round(data.head_angle)}°`;
        
        if (headProgress) {
            const anglePercent = Math.min(100, (data.head_angle / 90) * 100);
            headProgress.style.width = `${anglePercent}%`;
            headProgress.setAttribute('aria-valuenow', anglePercent);
            
            // Update position description
            if (headPosition) {
                if (data.head_angle < 10) {
                    headPosition.textContent = 'Centered';
                } else if (data.head_angle < 30) {
                    headPosition.textContent = 'Slightly Off';
                } else if (data.head_angle < 60) {
                    headPosition.textContent = 'Looking Away';
                } else {
                    headPosition.textContent = 'Turned Away';
                }
            }
        }
    }
    
    // Movement level
    const movementLevelValue = document.getElementById('movement-level-value');
    const movementProgress = document.getElementById('movement-progress');
    const movementStatus = document.getElementById('movement-status');
    
    if (movementLevelValue && data.movement_level !== undefined) {
        movementLevelValue.textContent = `${Math.round(data.movement_level)}%`;
        
        if (movementProgress) {
            const movementPercent = Math.min(100, data.movement_level);
            movementProgress.style.width = `${movementPercent}%`;
            movementProgress.setAttribute('aria-valuenow', movementPercent);
            
            // Update movement description
            if (movementStatus) {
                if (data.movement_level < 1) {
                    movementStatus.textContent = 'Very Low';
                } else if (data.movement_level < 5) {
                    movementStatus.textContent = 'Low';
                } else if (data.movement_level < 20) {
                    movementStatus.textContent = 'Medium';
                } else if (data.movement_level < 50) {
                    movementStatus.textContent = 'High';
                } else {
                    movementStatus.textContent = 'Very High';
                }
            }
        }
    }
}

// Draw annotations on the canvas
function drawAnnotations(data) {
    const canvas = document.getElementById('annotation-canvas');
    const video = document.getElementById('webcam');
    
    if (!canvas) {
        console.error('Annotation canvas not found');
        return;
    }
    
    if (!video || video.videoWidth === 0) {
        return;
    }
    
    // Ensure canvas is properly sized to match video display
    const videoRect = video.getBoundingClientRect();
    if (canvas.width !== videoRect.width || canvas.height !== videoRect.height) {
        canvas.width = videoRect.width;
        canvas.height = videoRect.height;
        console.log('Canvas resized to match video:', canvas.width, 'x', canvas.height);
    }
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Check if annotations are enabled
    if (!document.getElementById('toggle-annotations').checked) {
        console.log('Annotations disabled');
        return;
    }
    
    console.log('Drawing annotations with data:', {
        isPresent: data.is_present,
        poseLandmarks: data.pose_landmarks ? data.pose_landmarks.length : 0,
        canvasSize: { width: canvas.width, height: canvas.height }
    });
    
    // Test: Draw a simple red rectangle to verify canvas is working
    ctx.fillStyle = 'red';
    ctx.fillRect(10, 10, 50, 50);
    console.log('Test rectangle drawn');
    
    // Draw person detection box if present
    if (data.is_present) {
        // Draw posture indicator
        drawPostureIndicator(ctx, canvas.width, canvas.height, data.posture, data.angle);
        
        // Draw activity state label
        drawActivityLabel(ctx, canvas.width, canvas.height, data);
        
        // Draw head angle
        if (data.activity && data.activity.head_angle !== undefined) {
            drawHeadAngle(ctx, canvas.width, canvas.height, data.activity.head_angle);
        }
        
        // Draw pose landmarks if available
        if (data.pose_landmarks) {
            drawPoseLandmarks(ctx, data.pose_landmarks, canvas.width, canvas.height);
            drawPostureAnalysis(ctx, data.pose_landmarks, data, canvas.width, canvas.height);
        }
    } else {
        // Draw "No person detected" message
        ctx.font = '24px Arial';
        ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
        ctx.textAlign = 'center';
        ctx.fillText('No Person Detected', canvas.width / 2, canvas.height / 2);
    }
    
    // Draw focus indicator
    drawFocusIndicator(ctx, canvas.width, canvas.height, data.is_focused);
}

// Draw posture indicator
function drawPostureIndicator(ctx, width, height, posture, angle) {
    const x = 20;
    const y = height - 100;
    const boxWidth = 150;
    const boxHeight = 80;
    
    // Draw background box
    ctx.fillStyle = posture === 'upright' ? 'rgba(40, 167, 69, 0.7)' : 'rgba(255, 193, 7, 0.7)';
    ctx.fillRect(x, y, boxWidth, boxHeight);
    
    // Draw label
    ctx.font = '16px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText('Posture:', x + 10, y + 25);
    ctx.font = 'bold 20px Arial';
    ctx.fillText(posture.toUpperCase(), x + 10, y + 55);
    
    // Draw angle
    ctx.font = '14px Arial';
    ctx.fillText(`Angle: ${Math.round(angle)}°`, x + 10, y + 75);
}

// Pose landmark connections (simplified MediaPipe pose connections)
const POSE_CONNECTIONS = [
    // Face
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    // Arms
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    // Body
    [11, 12], [11, 23], [12, 24], [23, 24],
    // Legs
    [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32]
];

// Key landmarks for posture analysis
const KEY_LANDMARKS = {
    NOSE: 0,
    LEFT_EYE: 1,
    RIGHT_EYE: 2,
    LEFT_EAR: 7,
    RIGHT_EAR: 8,
    LEFT_SHOULDER: 11,
    RIGHT_SHOULDER: 12,
    LEFT_ELBOW: 13,
    RIGHT_ELBOW: 14,
    LEFT_WRIST: 15,
    RIGHT_WRIST: 16,
    LEFT_HIP: 23,
    RIGHT_HIP: 24
};

// Draw pose landmarks and connections
function drawPoseLandmarks(ctx, landmarks, width, height) {
    if (!landmarks || landmarks.length === 0) return;
    
    // Draw connections first (behind landmarks)
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.6)';
    ctx.lineWidth = 2;
    
    POSE_CONNECTIONS.forEach(connection => {
        const startIdx = connection[0];
        const endIdx = connection[1];
        
        if (startIdx < landmarks.length && endIdx < landmarks.length) {
            const startPoint = landmarks[startIdx];
            const endPoint = landmarks[endIdx];
            
            // Only draw if both landmarks are visible
            if (startPoint.visibility > 0.5 && endPoint.visibility > 0.5) {
                ctx.beginPath();
                ctx.moveTo(startPoint.x * width, startPoint.y * height);
                ctx.lineTo(endPoint.x * width, endPoint.y * height);
                ctx.stroke();
            }
        }
    });
    
    // Draw landmarks
    landmarks.forEach((landmark, index) => {
        if (landmark.visibility > 0.5) {
            const x = landmark.x * width;
            const y = landmark.y * height;
            
            // Use different colors for different body parts
            let color = 'rgba(255, 255, 0, 0.8)'; // Default yellow
            
            if (index === KEY_LANDMARKS.NOSE) {
                color = 'rgba(255, 0, 0, 0.9)'; // Red for nose
            } else if (index === KEY_LANDMARKS.LEFT_SHOULDER || index === KEY_LANDMARKS.RIGHT_SHOULDER) {
                color = 'rgba(0, 0, 255, 0.9)'; // Blue for shoulders
            } else if (index === KEY_LANDMARKS.LEFT_HIP || index === KEY_LANDMARKS.RIGHT_HIP) {
                color = 'rgba(255, 0, 255, 0.9)'; // Magenta for hips
            } else if (index === KEY_LANDMARKS.LEFT_EAR || index === KEY_LANDMARKS.RIGHT_EAR) {
                color = 'rgba(0, 255, 255, 0.9)'; // Cyan for ears
            }
            
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw landmark index for debugging (small text)
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(index.toString(), x, y - 8);
        }
    });
}

// Draw detailed posture analysis overlays
function drawPostureAnalysis(ctx, landmarks, data, width, height) {
    if (!landmarks || landmarks.length === 0) return;
    
    // Draw shoulder line and angle
    if (landmarks[KEY_LANDMARKS.LEFT_SHOULDER] && landmarks[KEY_LANDMARKS.RIGHT_SHOULDER]) {
        const leftShoulder = landmarks[KEY_LANDMARKS.LEFT_SHOULDER];
        const rightShoulder = landmarks[KEY_LANDMARKS.RIGHT_SHOULDER];
        
        if (leftShoulder.visibility > 0.5 && rightShoulder.visibility > 0.5) {
            const leftX = leftShoulder.x * width;
            const leftY = leftShoulder.y * height;
            const rightX = rightShoulder.x * width;
            const rightY = rightShoulder.y * height;
            
            // Draw shoulder line
            ctx.strokeStyle = 'rgba(255, 165, 0, 0.8)'; // Orange
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(leftX, leftY);
            ctx.lineTo(rightX, rightY);
            ctx.stroke();
            
            // Calculate and display shoulder alignment
            const heightDiff = Math.abs(leftY - rightY);
            const alignmentText = `Shoulder Alignment: ${(data.shoulder_alignment * 100).toFixed(1)}%`;
            
            ctx.fillStyle = 'rgba(255, 165, 0, 0.9)';
            ctx.font = '14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(alignmentText, 10, 30);
        }
    }
    
    // Draw spine line (shoulder to hip midpoints)
    if (landmarks[KEY_LANDMARKS.LEFT_SHOULDER] && landmarks[KEY_LANDMARKS.RIGHT_SHOULDER] &&
        landmarks[KEY_LANDMARKS.LEFT_HIP] && landmarks[KEY_LANDMARKS.RIGHT_HIP]) {
        
        const leftShoulder = landmarks[KEY_LANDMARKS.LEFT_SHOULDER];
        const rightShoulder = landmarks[KEY_LANDMARKS.RIGHT_SHOULDER];
        const leftHip = landmarks[KEY_LANDMARKS.LEFT_HIP];
        const rightHip = landmarks[KEY_LANDMARKS.RIGHT_HIP];
        
        if (leftShoulder.visibility > 0.5 && rightShoulder.visibility > 0.5 &&
            leftHip.visibility > 0.5 && rightHip.visibility > 0.5) {
            
            const shoulderMidX = ((leftShoulder.x + rightShoulder.x) / 2) * width;
            const shoulderMidY = ((leftShoulder.y + rightShoulder.y) / 2) * height;
            const hipMidX = ((leftHip.x + rightHip.x) / 2) * width;
            const hipMidY = ((leftHip.y + rightHip.y) / 2) * height;
            
            // Draw spine line
            ctx.strokeStyle = 'rgba(255, 0, 255, 0.8)'; // Magenta
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.moveTo(shoulderMidX, shoulderMidY);
            ctx.lineTo(hipMidX, hipMidY);
            ctx.stroke();
            
            // Display spine angle
            const spineAngleText = `Spine Angle: ${data.angle.toFixed(1)}°`;
            ctx.fillStyle = 'rgba(255, 0, 255, 0.9)';
            ctx.font = '14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(spineAngleText, 10, 50);
        }
    }
    
    // Draw head forward position indicator
    if (landmarks[KEY_LANDMARKS.NOSE] && landmarks[KEY_LANDMARKS.LEFT_SHOULDER] && landmarks[KEY_LANDMARKS.RIGHT_SHOULDER]) {
        const nose = landmarks[KEY_LANDMARKS.NOSE];
        const leftShoulder = landmarks[KEY_LANDMARKS.LEFT_SHOULDER];
        const rightShoulder = landmarks[KEY_LANDMARKS.RIGHT_SHOULDER];
        
        if (nose.visibility > 0.5 && leftShoulder.visibility > 0.5 && rightShoulder.visibility > 0.5) {
            const noseX = nose.x * width;
            const noseY = nose.y * height;
            const shoulderMidX = ((leftShoulder.x + rightShoulder.x) / 2) * width;
            const shoulderMidY = ((leftShoulder.y + rightShoulder.y) / 2) * height;
            
            // Draw line from nose to shoulder midpoint
            ctx.strokeStyle = 'rgba(255, 255, 0, 0.6)'; // Yellow
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(noseX, noseY);
            ctx.lineTo(shoulderMidX, shoulderMidY);
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
            
            // Display head forward position
            const headForwardText = `Head Forward: ${(data.head_forward_position * 100).toFixed(1)}%`;
            ctx.fillStyle = 'rgba(255, 255, 0, 0.9)';
            ctx.font = '14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(headForwardText, 10, 70);
        }
    }
    
    // Draw neck angle indicator
    if (landmarks[KEY_LANDMARKS.LEFT_EAR] && landmarks[KEY_LANDMARKS.RIGHT_EAR] && landmarks[KEY_LANDMARKS.NOSE]) {
        const leftEar = landmarks[KEY_LANDMARKS.LEFT_EAR];
        const rightEar = landmarks[KEY_LANDMARKS.RIGHT_EAR];
        const nose = landmarks[KEY_LANDMARKS.NOSE];
        
        // Use the most visible ear
        const ear = leftEar.visibility > rightEar.visibility ? leftEar : rightEar;
        
        if (ear.visibility > 0.5 && nose.visibility > 0.5) {
            const earX = ear.x * width;
            const earY = ear.y * height;
            const noseX = nose.x * width;
            const noseY = nose.y * height;
            
            // Draw neck line
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)'; // Cyan
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(earX, earY);
            ctx.lineTo(noseX, noseY);
            ctx.stroke();
            
            // Display neck angle
            const neckAngleText = `Neck Angle: ${data.neck_angle.toFixed(1)}°`;
            ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
            ctx.font = '14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(neckAngleText, 10, 90);
        }
    }
    
    // Draw posture quality indicator in top right
    const qualityColor = {
        'excellent': 'rgba(0, 255, 0, 0.9)', // Green
        'good': 'rgba(135, 206, 235, 0.9)', // Light blue
        'fair': 'rgba(255, 255, 0, 0.9)', // Yellow
        'poor': 'rgba(255, 0, 0, 0.9)' // Red
    };
    
    ctx.fillStyle = qualityColor[data.posture_quality] || 'rgba(128, 128, 128, 0.9)';
    ctx.font = 'bold 18px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(`Posture: ${data.posture_quality.toUpperCase()}`, width - 20, 30);
    
    // Draw symmetry score
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.font = '14px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Symmetry: ${(data.symmetry_score * 100).toFixed(1)}%`, 10, 110);
}

// Draw activity label
function drawActivityLabel(ctx, width, height, data) {
    if (!data.activity) return;
    
    const x = width - 170;
    const y = height - 100;
    const boxWidth = 150;
    const boxHeight = 80;
    
    // Determine color based on activity state
    let bgColor = 'rgba(108, 117, 125, 0.7)'; // Default gray
    if (data.activity.activity_state === 'working') {
        bgColor = 'rgba(40, 167, 69, 0.7)'; // Green
    } else if (data.activity.activity_state === 'not_working') {
        bgColor = 'rgba(255, 193, 7, 0.7)'; // Yellow
    } else if (data.activity.activity_state === 'distracted_by_others') {
        bgColor = 'rgba(220, 53, 69, 0.7)'; // Red
    } else if (data.activity.activity_state === 'on_break') {
        bgColor = 'rgba(13, 202, 240, 0.7)'; // Cyan
    }
    
    // Draw background box
    ctx.fillStyle = bgColor;
    ctx.fillRect(x, y, boxWidth, boxHeight);
    
    // Draw label
    ctx.font = '16px Arial';
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'left';
    ctx.fillText('Activity:', x + 10, y + 25);
    
    // Format activity state for display
    const displayState = data.activity.activity_state.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    ctx.font = 'bold 18px Arial';
    ctx.fillText(displayState, x + 10, y + 50);
    
    // Draw substate if available
    if (data.activity.working_substate) {
        const displaySubstate = data.activity.working_substate.charAt(0).toUpperCase() + data.activity.working_substate.slice(1);
        ctx.font = '14px Arial';
        ctx.fillText(displaySubstate, x + 10, y + 75);
    }
}

// Draw focus indicator
function drawFocusIndicator(ctx, width, height, isFocused) {
    // Draw focus indicator in the top center
    const x = width / 2;
    const y = 30;
    const radius = 15;
    
    // Draw circle
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = isFocused ? 'rgba(40, 167, 69, 0.7)' : 'rgba(220, 53, 69, 0.7)';
    ctx.fill();
    
    // Draw icon
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '16px Arial';
    ctx.fillText(isFocused ? '✓' : 'X', x, y);
    
    // Draw label
    ctx.font = '14px Arial';
    ctx.fillText(isFocused ? 'Focused' : 'Distracted', x, y + 25);
}

// Draw head angle
function drawHeadAngle(ctx, width, height, headAngle) {
    const x = width / 2;
    const y = 80;
    const radius = 25;
    
    // Draw head outline
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw angle line
    const angleRad = headAngle * (Math.PI / 180);
    const lineLength = radius * 1.5;
    
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.sin(angleRad) * lineLength, y - Math.cos(angleRad) * lineLength);
    ctx.strokeStyle = 'rgba(255, 193, 7, 0.8)';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw angle text
    ctx.font = '14px Arial';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.textAlign = 'center';
    ctx.fillText(`${Math.round(headAngle)}°`, x, y + radius + 15);
}

// Draw pose landmarks (placeholder, actual implementation would use the landmarks from backend)
function drawPoseLandmarks(ctx, landmarks, width, height) {
    // This would be implemented if the backend provides the landmarks
    // For now, this is a placeholder
    console.log('Drawing landmarks not implemented yet');
}

// Track events for the timeline
function trackEvents(data) {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    // Check for state changes
    if (data.activity && data.activity.activity_state !== lastActivityState) {
        addTimelineEvent({
            type: 'activity',
            time: timeString,
            description: `Activity changed to ${data.activity.activity_state.replace(/_/g, ' ')}`,
            details: data.activity.working_substate ? 
                     `Substate: ${data.activity.working_substate}` : ''
        });
        lastActivityState = data.activity.activity_state;
    }
    
    // Check for posture changes
    if (data.posture !== lastPosture) {
        addTimelineEvent({
            type: 'posture',
            time: timeString,
            description: `Posture changed to ${data.posture}`,
            details: `Angle: ${Math.round(data.angle)}°`
        });
        lastPosture = data.posture;
    }
    
    // Check for focus changes
    if (data.is_focused !== lastFocused) {
        addTimelineEvent({
            type: 'focus',
            time: timeString,
            description: data.is_focused ? 'Now focused' : 'Focus lost',
            details: ''
        });
        lastFocused = data.is_focused;
    }
}

// Add event to timeline
function addTimelineEvent(event) {
    // Add to the beginning of the array
    timelineEvents.unshift(event);
    
    // Limit the number of events
    if (timelineEvents.length > maxTimelineEvents) {
        timelineEvents = timelineEvents.slice(0, maxTimelineEvents);
    }
    
    // Update the timeline display
    updateTimelineDisplay();
    
    // Show notification only for non-posture events
    if (event.type !== 'posture') {
        showDetectionToast(event);
    }
}

// Update timeline display
function updateTimelineDisplay() {
    const timeline = document.getElementById('detection-timeline');
    if (!timeline) return;
    
    timeline.innerHTML = '';
    
    timelineEvents.forEach(event => {
        const item = document.createElement('div');
        item.className = `timeline-item ${event.type}`;
        
        const content = document.createElement('div');
        content.className = 'timeline-item-content';
        
        const header = document.createElement('div');
        header.className = 'd-flex justify-content-between';
        
        const title = document.createElement('strong');
        title.textContent = event.description;
        
        const time = document.createElement('span');
        time.className = 'timeline-item-time';
        time.textContent = event.time;
        
        header.appendChild(title);
        header.appendChild(time);
        content.appendChild(header);
        
        if (event.details) {
            const details = document.createElement('div');
            details.className = 'text-muted small mt-1';
            details.textContent = event.details;
            content.appendChild(details);
        }
        
        item.appendChild(content);
        timeline.appendChild(item);
    });
}

// Show detection toast notification
function showDetectionToast(event) {
    const toast = document.getElementById('detection-toast');
    const title = document.getElementById('detection-title');
    const message = document.getElementById('detection-message');
    const time = document.getElementById('detection-time');
    
    if (!toast || !title || !message || !time) return;
    
    // Set content
    title.textContent = event.type.charAt(0).toUpperCase() + event.type.slice(1);
    message.textContent = event.description + (event.details ? ` (${event.details})` : '');
    time.textContent = event.time;
    
    // Show toast using Bootstrap's toast API
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

// Capture a screenshot
function captureScreenshot() {
    const canvas = document.getElementById('annotation-canvas');
    if (!canvas) return;
    
    // Convert canvas to image URL
    const imageUrl = canvas.toDataURL('image/png');
    
    // Create a temporary link and trigger download
    const a = document.createElement('a');
    a.href = imageUrl;
    a.download = `posture-monitor-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
    a.click();
}