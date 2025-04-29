"""
Activity Recognition Model Trainer

This module handles the training and conversion of TensorFlow models to TFLite 
for activity recognition based on pose landmark features.

Requirements:
- TensorFlow 2.x
- NumPy
- MediaPipe (for preprocessing)
"""

import numpy as np
import os
import time
import logging
import json

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as e:
    print(f"TensorFlow import error: {e}. Using fallback.")
    tf = None
    keras = None

logger = logging.getLogger(__name__)

# Activity state mapping
ACTIVITY_STATES = {
    0: "working",
    1: "not_working",
    2: "distracted_by_others",
    3: "on_break",
    4: "idle"
}

# Working substates mapping
WORKING_SUBSTATES = {
    0: None,  # Generic working
    1: "typing",
    2: "writing",
    3: "reading",
    4: "phone_use"
}

class ActivityModelTrainer:
    """
    Trains and converts models for activity recognition from pose landmarks.
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the trainer with the model directory.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.activity_model_path = os.path.join(model_dir, "activity_recognition.h5")
        self.activity_tflite_path = os.path.join(model_dir, "activity_recognition.tflite")
        self.substate_model_path = os.path.join(model_dir, "working_substate.h5")
        self.substate_tflite_path = os.path.join(model_dir, "working_substate.tflite")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Metadata for TFLite model
        self.activity_metadata = {
            "name": "Activity Recognition Model",
            "description": "Classifies user activity based on pose landmarks",
            "version": "1.0",
            "author": "Smart Work Focus App",
            "classes": ACTIVITY_STATES
        }
        
        self.substate_metadata = {
            "name": "Working Substate Model",
            "description": "Classifies specific working activities",
            "version": "1.0",
            "author": "Smart Work Focus App",
            "classes": WORKING_SUBSTATES
        }
        
        # Save metadata
        with open(os.path.join(model_dir, "activity_metadata.json"), "w") as f:
            json.dump(self.activity_metadata, f)
            
        with open(os.path.join(model_dir, "substate_metadata.json"), "w") as f:
            json.dump(self.substate_metadata, f)
    
    def build_activity_model(self, input_shape=(195,)):  # 33 landmarks x 3 (x,y,z) = 99 features
        """
        Build and compile a neural network for activity classification.
        
        Args:
            input_shape: Shape of the input feature vector
            
        Returns:
            The compiled model
        """
        if tf is None or keras is None:
            logger.error("TensorFlow not available for building model")
            return None
            
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dropout(0.2),  # Add dropout to prevent overfitting
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(ACTIVITY_STATES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_substate_model(self, input_shape=(195,)):
        """
        Build and compile a neural network for working substate classification.
        
        Args:
            input_shape: Shape of the input feature vector
            
        Returns:
            The compiled model
        """
        if tf is None or keras is None:
            logger.error("TensorFlow not available for building model")
            return None
            
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(len(WORKING_SUBSTATES), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_activity_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the activity classification model.
        
        Args:
            X_train: Training features (pose landmarks)
            y_train: Training labels (activity states)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if tf is None:
            logger.error("TensorFlow not available for training")
            return None
            
        model = self.build_activity_model()
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the model
        model.save(self.activity_model_path)
        logger.info(f"Activity model saved to {self.activity_model_path}")
        
        return history
    
    def train_substate_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the working substate classification model.
        
        Args:
            X_train: Training features (pose landmarks)
            y_train: Training labels (working substates)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        if tf is None:
            logger.error("TensorFlow not available for training")
            return None
            
        model = self.build_substate_model()
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the model
        model.save(self.substate_model_path)
        logger.info(f"Substate model saved to {self.substate_model_path}")
        
        return history
    
    def convert_to_tflite(self, model_path, tflite_path):
        """
        Convert a saved Keras model to TensorFlow Lite format.
        
        Args:
            model_path: Path to the saved Keras model
            tflite_path: Path where to save the TFLite model
            
        Returns:
            True if conversion was successful, False otherwise
        """
        if tf is None:
            logger.error("TensorFlow not available for conversion")
            return False
            
        try:
            # Load the model
            model = tf.keras.models.load_model(model_path)
            
            # Convert the model
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Configure optimization options for TinyML
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Ensure models run well on low-power devices
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            
            tflite_model = converter.convert()
            
            # Save the model
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
                
            logger.info(f"TFLite model saved to {tflite_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error converting model to TFLite: {str(e)}")
            return False
    
    def generate_sample_data(self, num_samples=1000):
        """
        Generate synthetic training data for demonstration purposes.
        In a real implementation, this would be replaced with actual
        collected pose landmark data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            X: Feature data
            y_activity: Activity state labels
            y_substate: Working substate labels
        """
        # Generate random pose landmark data
        # 33 landmarks (MediaPipe) x 3 (x,y,z) coords + 2 additional features (head angle, movement)
        X = np.random.rand(num_samples, 33*3+2) * 2 - 1  # Scale to [-1, 1]
        
        # Generate activity labels
        y_activity = np.random.randint(0, len(ACTIVITY_STATES), size=num_samples)
        
        # Generate substate labels
        y_substate = np.random.randint(0, len(WORKING_SUBSTATES), size=num_samples)
        
        return X, y_activity, y_substate
    
    def train_and_convert_models(self):
        """
        Train both models and convert them to TFLite format.
        
        Returns:
            True if successful, False otherwise
        """
        # Generate sample data (in real implementation, use collected data)
        X, y_activity, y_substate = self.generate_sample_data()
        
        # Train models
        activity_history = self.train_activity_model(X, y_activity)
        substate_history = self.train_substate_model(X, y_substate)
        
        if activity_history is None or substate_history is None:
            logger.error("Failed to train models")
            return False
        
        # Convert models to TFLite
        activity_converted = self.convert_to_tflite(
            self.activity_model_path,
            self.activity_tflite_path
        )
        
        substate_converted = self.convert_to_tflite(
            self.substate_model_path,
            self.substate_tflite_path
        )
        
        return activity_converted and substate_converted
    
    def save_empty_tflite_files(self):
        """
        Save empty TFLite files as placeholders when TensorFlow is not available.
        This allows the main application to detect that TFLite files exist but
        will need to fall back to heuristic-based approaches.
        
        Returns:
            True if files were created, False otherwise
        """
        try:
            # Create placeholder model files
            with open(self.activity_tflite_path, 'wb') as f:
                f.write(b'TFLite Model Placeholder')
                
            with open(self.substate_tflite_path, 'wb') as f:
                f.write(b'TFLite Model Placeholder')
                
            logger.info("Created placeholder TFLite model files")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create placeholder files: {str(e)}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = ActivityModelTrainer()
    
    if tf is not None:
        logger.info("Starting model training...")
        success = trainer.train_and_convert_models()
        
        if success:
            logger.info("Successfully trained and converted models")
        else:
            logger.error("Failed to train or convert models")
            trainer.save_empty_tflite_files()
    else:
        logger.warning("TensorFlow not available, creating placeholder model files")
        trainer.save_empty_tflite_files()