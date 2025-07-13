
#!/usr/bin/env python3
"""
Script to train the SitPose posture classification model
"""

import logging
import sys
import os
from utils.sitpose_trainer import SitPoseTrainer

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting SitPose model training...")
    
    # Create trainer
    trainer = SitPoseTrainer()
    
    try:
        # Train the model
        model, scaler, accuracy = trainer.train_model()
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model accuracy: {accuracy:.3f}")
        logger.info(f"Model saved to: {trainer.model_path}")
        logger.info(f"Scaler saved to: {trainer.scaler_path}")
        logger.info(f"Metadata saved to: {trainer.metadata_path}")
        
        # Test the model with a simple prediction
        logger.info("Testing model prediction...")
        
        # Create dummy landmarks for testing
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        dummy_landmarks = []
        
        # Create a simple upright pose
        for i in range(33):
            # Simple normalized coordinates for sitting straight pose
            if i == mp_pose.PoseLandmark.NOSE.value:
                landmark = type('obj', (object,), {'x': 0.5, 'y': 0.2, 'z': 0.0})()
            elif i in [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value]:
                x_offset = -0.1 if i == mp_pose.PoseLandmark.LEFT_SHOULDER.value else 0.1
                landmark = type('obj', (object,), {'x': 0.5 + x_offset, 'y': 0.4, 'z': 0.0})()
            elif i in [mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value]:
                x_offset = -0.05 if i == mp_pose.PoseLandmark.LEFT_HIP.value else 0.05
                landmark = type('obj', (object,), {'x': 0.5 + x_offset, 'y': 0.6, 'z': 0.0})()
            elif i in [mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value]:
                x_offset = -0.05 if i == mp_pose.PoseLandmark.LEFT_KNEE.value else 0.05
                landmark = type('obj', (object,), {'x': 0.5 + x_offset, 'y': 0.8, 'z': 0.0})()
            else:
                # Default position for other landmarks
                landmark = type('obj', (object,), {'x': 0.5, 'y': 0.5, 'z': 0.0})()
            
            dummy_landmarks.append(landmark)
        
        # Test prediction
        posture, confidence = trainer.predict_posture(dummy_landmarks)
        logger.info(f"Test prediction: {posture} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
