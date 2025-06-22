// Dashboard.js - Main dashboard functionality and initialization

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeFocusChart();
    initializeHealthScoreChart();
    
    // Set up refresh intervals
    setInterval(updateFocusScore, 30000); // Update focus score every 30 seconds
    setInterval(updatePostureStatus, 5000); // Update posture status every 5 seconds
    setInterval(updateWaterIntake, 60000); // Update water intake every minute
    setInterval(updateHealthScore, 30000); // Update health score every 30 seconds
    
    // Initial data load
    updateFocusScore();
    updatePostureStatus();
    updateWaterIntake();
    updateHealthScore();
    
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

// Health score chart initialization
function initializeHealthScoreChart() {
    const ctx = document.getElementById('health-score-chart').getContext('2d');
    window.healthScoreChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: [
                    'rgb(255, 106, 0)',
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

// Update health score from API
function updateHealthScore() {
    // Calculate health score based on focus, posture, and water intake
    Promise.all([
        fetch('/api/focus_score').then(r => r.json()),
        fetch('/api/posture').then(r => r.json()),
        fetch('/api/water_intake').then(r => r.json())
    ]).then(([focusData, postureData, waterData]) => {
        // Calculate component scores
        const focusScore = focusData.score || 0;
        
        // Calculate posture score based on quality
        let postureScore = 0;
        switch(postureData.posture_quality) {
            case 'excellent': postureScore = 100; break;
            case 'good': postureScore = 80; break;
            case 'fair': postureScore = 60; break;
            case 'poor': postureScore = 40; break;
            default: postureScore = 0;
        }
        
        // Calculate hydration score (2000ml target)
        const hydrationPercent = Math.min(100, (waterData.amount / 2000) * 100);
        const hydrationScore = hydrationPercent;
        
        // Calculate overall health score (weighted average)
        const healthScore = Math.round((focusScore * 0.4 + postureScore * 0.4 + hydrationScore * 0.2));
        
        // Update health score display
        document.getElementById('health-score-value').textContent = healthScore;
        document.getElementById('health-focus-score').textContent = Math.round(focusScore);
        document.getElementById('health-posture-score').textContent = Math.round(postureScore);
        document.getElementById('health-hydration-score').textContent = Math.round(hydrationScore);
        
        // Update health score chart
        if (window.healthScoreChart) {
            window.healthScoreChart.data.datasets[0].data = [healthScore, 100 - healthScore];
            window.healthScoreChart.update();
        }
        
        // Update health status badge and text
        const healthBadge = document.getElementById('health-score-badge');
        const healthStatusText = document.getElementById('health-status-text');
        const achievementDisplay = document.getElementById('achievement-display');
        
        if (healthScore >= 95) {
            healthBadge.textContent = 'Elite Performer';
            healthBadge.className = 'badge bg-warning';
            healthStatusText.textContent = 'Outstanding health metrics!';
            achievementDisplay.style.display = 'block';
        } else if (healthScore >= 80) {
            healthBadge.textContent = 'Excellent';
            healthBadge.className = 'badge bg-success';
            healthStatusText.textContent = 'Great work maintaining healthy habits!';
            achievementDisplay.style.display = 'none';
        } else if (healthScore >= 60) {
            healthBadge.textContent = 'Good';
            healthBadge.className = 'badge bg-info';
            healthStatusText.textContent = 'Good progress, keep it up!';
            achievementDisplay.style.display = 'none';
        } else if (healthScore >= 40) {
            healthBadge.textContent = 'Fair';
            healthBadge.className = 'badge bg-warning';
            healthStatusText.textContent = 'Room for improvement in your habits.';
            achievementDisplay.style.display = 'none';
        } else {
            healthBadge.textContent = 'Needs Work';
            healthBadge.className = 'badge bg-danger';
            healthStatusText.textContent = 'Focus on building healthier work habits.';
            achievementDisplay.style.display = 'none';
        }
        
    }).catch(error => {
        console.error('Error calculating health score:', error);
    });
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
            
            // Set base styles based on posture quality
            let qualityClass, qualityText;
            switch(data.posture_quality) {
                case 'excellent':
                    qualityClass = 'bg-success';
                    qualityText = 'Excellent';
                    break;
                case 'good':
                    qualityClass = 'bg-info';
                    qualityText = 'Good';
                    break;
                case 'fair':
                    qualityClass = 'bg-warning';
                    qualityText = 'Fair';
                    break;
                case 'poor':
                    qualityClass = 'bg-danger';
                    qualityText = 'Poor';
                    break;
                default:
                    qualityClass = 'bg-secondary';
                    qualityText = 'Unknown';
            }
            
            // Update detailed metrics if they exist in the container
            const detailedMetricsContainer = document.getElementById('posture-detailed-metrics');
            if (detailedMetricsContainer) {
                let metricsHTML = '';
                
                // Add neck angle metric
                if (data.neck_angle !== undefined) {
                    const neckAngleClass = data.neck_angle > 20 ? 'text-danger' : 'text-success';
                    metricsHTML += `
                        <div class="metric-item">
                            <div class="metric-label">Neck Angle</div>
                            <div class="metric-value ${neckAngleClass}">${Math.round(data.neck_angle)}°</div>
                        </div>
                    `;
                }
                
                // Add shoulder alignment metric
                if (data.shoulder_alignment !== undefined) {
                    const alignmentPercent = Math.round(data.shoulder_alignment * 100);
                    let alignmentClass = 'text-success';
                    if (alignmentPercent < 80) alignmentClass = 'text-danger';
                    else if (alignmentPercent < 90) alignmentClass = 'text-warning';
                    
                    metricsHTML += `
                        <div class="metric-item">
                            <div class="metric-label">Shoulder Alignment</div>
                            <div class="metric-value ${alignmentClass}">${alignmentPercent}%</div>
                        </div>
                    `;
                }
                
                // Add head forward position
                if (data.head_forward_position !== undefined) {
                    const headForwardClass = data.head_forward_position > 0.1 ? 'text-warning' : 'text-success';
                    const headForwardValue = Math.round(data.head_forward_position * 100);
                    metricsHTML += `
                        <div class="metric-item">
                            <div class="metric-label">Head Position</div>
                            <div class="metric-value ${headForwardClass}">${headForwardValue > 0 ? headForwardValue : 'Good'}</div>
                        </div>
                    `;
                }
                
                // Add symmetry score
                if (data.symmetry_score !== undefined) {
                    const symmetryPercent = Math.round(data.symmetry_score * 100);
                    let symmetryClass = 'text-success';
                    if (symmetryPercent < 70) symmetryClass = 'text-danger';
                    else if (symmetryPercent < 85) symmetryClass = 'text-warning';
                    
                    metricsHTML += `
                        <div class="metric-item">
                            <div class="metric-label">Body Symmetry</div>
                            <div class="metric-value ${symmetryClass}">${symmetryPercent}%</div>
                        </div>
                    `;
                }
                
                detailedMetricsContainer.innerHTML = metricsHTML;
            }
            
            // Display posture quality badge
            if (data.posture === 'upright' || data.posture === 'slouched') {
                postureState.textContent = qualityText;
                postureState.className = `badge ${qualityClass}`;
                
                // Use feedback from advanced analysis if available
                if (data.feedback) {
                    postureAdvice.textContent = data.feedback;
                    postureFeedback.textContent = `Posture Quality: ${qualityText}`;
                } else {
                    // Fallback to basic feedback
                    if (data.posture === 'upright') {
                        postureFeedback.textContent = 'Good posture detected';
                        postureAdvice.textContent = 'Maintain this upright position for optimal focus.';
                    } else {
                        postureFeedback.textContent = 'Slouching detected';
                        postureAdvice.textContent = 'Try to sit up straight to improve your focus.';
                    }
                }
                
                // Set angle indicator color based on quality
                angleIndicator.className = `progress-bar ${qualityClass}`;
                
                // Show recommendation if available
                if (data.recommendation) {
                    const recommendationEl = document.getElementById('posture-recommendation');
                    if (recommendationEl) {
                        recommendationEl.textContent = data.recommendation;
                    }
                }
                
                // Show trend info if available
                if (data.trend) {
                    const trendEl = document.getElementById('posture-trend');
                    if (trendEl) {
                        let trendIcon, trendClass;
                        switch(data.trend) {
                            case 'positive':
                                trendIcon = 'trending_up';
                                trendClass = 'text-success';
                                break;
                            case 'negative':
                                trendIcon = 'trending_down';
                                trendClass = 'text-danger';
                                break;
                            default:
                                trendIcon = 'trending_flat';
                                trendClass = 'text-warning';
                        }
                        
                        trendEl.innerHTML = `<i class="material-icons ${trendClass}">${trendIcon}</i> ${data.trend.charAt(0).toUpperCase() + data.trend.slice(1)} trend`;
                    }
                }
            } else {
                // Unknown posture state
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
