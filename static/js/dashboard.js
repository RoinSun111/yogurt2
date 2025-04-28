// Dashboard.js - Main dashboard functionality and initialization

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeFocusChart();
    initializeDailyStatsChart();
    
    // Set up refresh intervals
    setInterval(updateFocusScore, 30000); // Update focus score every 30 seconds
    setInterval(updatePostureStatus, 5000); // Update posture status every 5 seconds
    setInterval(updateWaterIntake, 60000); // Update water intake every minute
    
    // Initial data load
    updateFocusScore();
    updatePostureStatus();
    updateWaterIntake();
    
    // Initialize notification system
    initializeNotifications();
});

// Focus score chart initialization
function initializeFocusChart() {
    const ctx = document.getElementById('focus-chart').getContext('2d');
    window.focusChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: [
                    'rgb(13, 110, 253)',
                    'rgba(255, 255, 255, 0.1)'
                ],
                borderWidth: 0,
                circumference: 360,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

// Daily stats chart initialization
function initializeDailyStatsChart() {
    const ctx = document.getElementById('daily-stats-chart').getContext('2d');
    window.dailyStatsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: generateHourLabels(),
            datasets: [{
                label: 'Focus Score',
                data: Array(24).fill(null),
                borderColor: 'rgb(13, 110, 253)',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Focus Score (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Generate hour labels for the daily stats chart
function generateHourLabels() {
    const hours = [];
    for (let i = 0; i < 24; i++) {
        hours.push(i.toString().padStart(2, '0') + ':00');
    }
    return hours;
}

// Update focus score from API
function updateFocusScore() {
    fetch('/api/focus_score')
        .then(response => response.json())
        .then(data => {
            // Update score display
            document.getElementById('focus-score').textContent = data.score;
            
            // Update working time display
            const hours = Math.floor(data.working_time / 3600);
            const minutes = Math.floor((data.working_time % 3600) / 60);
            document.getElementById('working-time').textContent = 
                (hours > 0 ? hours + 'h ' : '') + minutes + 'm';
            
            // Update chart
            window.focusChart.data.datasets[0].data = [data.score, 100 - data.score];
            window.focusChart.update();
            
            // Update focus state badge
            const focusState = document.getElementById('focus-state');
            if (data.score > 70) {
                focusState.textContent = 'Focused';
                focusState.className = 'badge bg-success';
            } else if (data.score > 40) {
                focusState.textContent = 'Moderate';
                focusState.className = 'badge bg-warning';
            } else {
                focusState.textContent = 'Distracted';
                focusState.className = 'badge bg-danger';
            }
        })
        .catch(error => {
            console.error('Error fetching focus score:', error);
            showNotification('Error', 'Could not update focus score', 'error');
        });
}

// Update posture status from API
function updatePostureStatus() {
    fetch('/api/posture')
        .then(response => response.json())
        .then(data => {
            // Update posture state badge
            const postureState = document.getElementById('posture-state');
            const postureFeedback = document.getElementById('posture-feedback');
            const postureAdvice = document.getElementById('posture-advice');
            
            // Update angle indicator
            const angleIndicator = document.getElementById('angle-indicator');
            const angle = data.angle;
            angleIndicator.textContent = Math.round(angle) + '°';
            angleIndicator.style.width = Math.min(100, (angle / 30) * 100) + '%';
            
            if (data.posture === 'upright') {
                postureState.textContent = 'Upright';
                postureState.className = 'badge bg-success';
                postureFeedback.textContent = 'Great posture!';
                postureAdvice.textContent = 'Maintain this upright position for optimal focus.';
                angleIndicator.className = 'progress-bar bg-success';
            } else if (data.posture === 'slouched') {
                postureState.textContent = 'Slouched';
                postureState.className = 'badge bg-warning';
                postureFeedback.textContent = 'Slouching detected';
                postureAdvice.textContent = 'Try to sit up straight to improve your focus.';
                angleIndicator.className = 'progress-bar bg-warning';
                
                // Show notification if needed
                if (angle > 20) {
                    showNotification('Posture Alert', 'You are slouching! Please sit up straight.', 'warning');
                }
            } else {
                postureState.textContent = 'Unknown';
                postureState.className = 'badge bg-secondary';
                postureFeedback.textContent = 'Checking posture...';
                postureAdvice.textContent = 'Please position yourself in front of the camera.';
                angleIndicator.className = 'progress-bar';
            }
        })
        .catch(error => {
            console.error('Error fetching posture status:', error);
        });
}

// Initialize notification system
function initializeNotifications() {
    // Check if notifications are supported
    if (!("Notification" in window)) {
        console.log("This browser does not support notifications");
        return;
    }
    
    // Request permission for notifications
    if (Notification.permission !== "granted" && Notification.permission !== "denied") {
        Notification.requestPermission();
    }
}

// Show a notification
function showNotification(title, message, type = 'info') {
    // Toast notification
    const toast = document.getElementById('notification-toast');
    const toastTitle = document.getElementById('toast-title');
    const toastMessage = document.getElementById('toast-message');
    const toastTime = document.getElementById('toast-time');
    
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    toastTime.textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    const toastInstance = new bootstrap.Toast(toast);
    toastInstance.show();
    
    // Browser notification if available and permitted
    if (Notification.permission === "granted") {
        new Notification(title, {
            body: message,
            icon: '/static/icons/notification-icon.svg'
        });
    }
}
