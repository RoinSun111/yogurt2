// Posture.js - Handle posture analysis and feedback

document.addEventListener('DOMContentLoaded', function() {
    // No initialization needed for posture module as it's updated via the dashboard.js
    // This file can be extended with specific posture-related functionality
});

// Get posture correction advice based on angle and posture type
function getPostureAdvice(posture, angle) {
    if (posture === 'upright') {
        return "Great posture! Keep it up for optimal focus.";
    } else if (posture === 'slouched') {
        if (angle > 25) {
            return "You're significantly slouched. Please sit upright and align your spine.";
        } else if (angle > 15) {
            return "You're slightly slouched. Try to straighten your back.";
        }
    }
    
    return "Try to maintain an upright posture with your back straight.";
}

// Set up posture alert thresholds and timeout
let postureAlertTimeout = null;
const POSTURE_ALERT_THRESHOLD = 15; // Angle in degrees
const POSTURE_ALERT_DELAY = 60000; // One minute between alerts

// Check if a posture alert should be triggered
function checkPostureAlert(posture, angle) {
    if (posture === 'slouched' && angle > POSTURE_ALERT_THRESHOLD) {
        // Only show an alert if we haven't shown one recently
        if (!postureAlertTimeout) {
            showPostureAlert(angle);
            
            // Set a timeout to prevent too many alerts
            postureAlertTimeout = setTimeout(() => {
                postureAlertTimeout = null;
            }, POSTURE_ALERT_DELAY);
        }
    }
}

// Show a posture alert/notification
function showPostureAlert(angle) {
    const message = `Your posture needs correction! Current angle: ${Math.round(angle)}°`;
    showNotification('Posture Alert', message, 'warning');
    
    // Optionally play a gentle sound alert
    // playPostureCorrectionSound();
}

// Function to be called when posture data is updated
function updatePostureDisplay(data) {
    const postureState = document.getElementById('posture-state');
    const postureFeedback = document.getElementById('posture-feedback');
    const postureAdvice = document.getElementById('posture-advice');
    const angleIndicator = document.getElementById('angle-indicator');
    
    // Update the angle indicator
    const angle = data.angle;
    angleIndicator.textContent = Math.round(angle) + '°';
    angleIndicator.style.width = Math.min(100, (angle / 30) * 100) + '%';
    
    // Update the posture state and feedback
    if (data.posture === 'upright') {
        postureState.textContent = 'Upright';
        postureState.className = 'badge bg-success';
        postureFeedback.textContent = 'Great posture!';
        angleIndicator.className = 'progress-bar bg-success';
    } else if (data.posture === 'slouched') {
        postureState.textContent = 'Slouched';
        postureState.className = 'badge bg-warning';
        postureFeedback.textContent = 'Slouching detected';
        angleIndicator.className = 'progress-bar bg-warning';
        
        // Check if we should show an alert
        checkPostureAlert(data.posture, angle);
    } else {
        postureState.textContent = 'Unknown';
        postureState.className = 'badge bg-secondary';
        postureFeedback.textContent = 'Checking posture...';
        angleIndicator.className = 'progress-bar';
    }
    
    // Update the advice text
    postureAdvice.textContent = getPostureAdvice(data.posture, angle);
}
