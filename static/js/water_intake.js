// Water Intake Tracking and Visualization

let waterRemindInterval;
const WATER_REMINDER_INTERVAL = 2 * 60 * 60 * 1000; // 2 hours in milliseconds

document.addEventListener('DOMContentLoaded', function() {
    // Set up water intake buttons
    setupWaterButtons();
    
    // Set up water reminder timer
    setupWaterReminders();
    
    // Initial water intake data
    updateWaterIntake();
});

// Set up water intake buttons
function setupWaterButtons() {
    const waterButtons = document.querySelectorAll('.water-btn');
    
    waterButtons.forEach(button => {
        button.addEventListener('click', function() {
            const amount = parseInt(this.dataset.amount);
            addWaterIntake(amount);
        });
    });
}

// Add water intake
function addWaterIntake(amount) {
    fetch('/api/add_water', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ amount: amount })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update the water intake display
            updateWaterIntake();
            
            // Show a notification
            showNotification(
                'Water Intake', 
                `Added ${amount}ml of water. Stay hydrated!`,
                'info'
            );
        }
    })
    .catch(error => {
        console.error('Error adding water intake:', error);
    });
}

// Update water intake from API
function updateWaterIntake() {
    fetch('/api/water_intake')
        .then(response => response.json())
        .then(data => {
            // Update water amount display
            document.getElementById('water-amount').textContent = 
                `${data.total_intake} / ${data.goal} ml`;
            
            // Update water level visualization
            const percentage = Math.min(100, (data.total_intake / data.goal) * 100);
            document.getElementById('water-level').style.height = `${percentage}%`;
            
            // Update last drink time
            document.getElementById('last-drink-time').textContent = data.last_drink;
            
            // Apply color based on percentage
            const waterLevel = document.getElementById('water-level');
            if (percentage < 25) {
                waterLevel.style.backgroundColor = 'rgba(220, 53, 69, 0.7)'; // Danger red
            } else if (percentage < 50) {
                waterLevel.style.backgroundColor = 'rgba(255, 193, 7, 0.7)'; // Warning yellow
            } else {
                waterLevel.style.backgroundColor = 'rgba(13, 110, 253, 0.7)'; // Primary blue
            }
        })
        .catch(error => {
            console.error('Error fetching water intake:', error);
        });
}

// Set up water reminder timer
function setupWaterReminders() {
    // Clear any existing interval
    if (waterRemindInterval) {
        clearInterval(waterRemindInterval);
    }
    
    // Check water intake and remind every 2 hours
    waterRemindInterval = setInterval(checkWaterReminder, WATER_REMINDER_INTERVAL);
    
    // Also check once after 10 minutes for the first reminder
    setTimeout(checkWaterReminder, 10 * 60 * 1000);
}

// Check if a water reminder should be shown
function checkWaterReminder() {
    fetch('/api/water_intake')
        .then(response => response.json())
        .then(data => {
            const currentTime = new Date();
            const lastDrinkTime = data.last_drink !== 'No drinks today' ? 
                new Date(`${currentTime.toDateString()} ${data.last_drink}`) : 
                new Date(0); // If no drinks, set to epoch time
            
            const timeSinceLastDrink = currentTime - lastDrinkTime;
            
            // If it's been over 2 hours since the last drink, show a reminder
            if (timeSinceLastDrink > WATER_REMINDER_INTERVAL) {
                showWaterReminder();
            }
            
            // Also remind if intake is below target for the time of day
            const hoursInDay = currentTime.getHours() + (currentTime.getMinutes() / 60);
            const targetIntakeForCurrentTime = (data.goal / 16) * hoursInDay; // Spread over 16 waking hours
            
            if (data.total_intake < targetIntakeForCurrentTime * 0.8) { // 20% tolerance
                showWaterReminder();
            }
        })
        .catch(error => {
            console.error('Error checking water reminder:', error);
        });
}

// Show a water reminder notification
function showWaterReminder() {
    showNotification(
        'Hydration Reminder', 
        'Time to drink some water! Stay hydrated for better focus.',
        'info'
    );
}

// Detect if user drinks water by camera (placeholder function)
// This would be implemented if the TinyML model for detecting water bottle interactions is added
function detectWaterDrinking(frameData) {
    // This would be implemented with actual ML detection
    // For now, this is a placeholder
    if (frameData && frameData.waterBottleInteraction) {
        addWaterIntake(250); // Default to 250ml when detecting a drink
    }
}
