"""
TensorFlow Interface for TinyML Integration

This module serves as the primary interface for TensorFlow functionality
in the Smart Work Focus Score application. It handles model training,
conversion, and inference with proper TensorFlow/TensorFlow Lite abstractions.
"""

import logging
import numpy as np
import os
import json
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# First, try to import TensorFlow/TensorFlow Lite
try:
    # Try tflite_runtime first (smaller, optimized for inference)
    import tflite_runtime.interpreter as tflite
    USE_TFLITE_RUNTIME = True
    TENSORFLOW_AVAILABLE = True
    logger.info("Using TensorFlow Lite Runtime for inference")
except ImportError:
    try:
        # Fall back to full TensorFlow
        import tensorflow as tf
        USE_TFLITE_RUNTIME = False
        TENSORFLOW_AVAILABLE = True
        logger.info("Using TensorFlow for inference")
    except ImportError:
        # No ML frameworks available
        TENSORFLOW_AVAILABLE = False
        logger.warning("TensorFlow/TensorFlow Lite not available. Using heuristic methods only.")

class TensorFlowInterface:
    """
    Handles all TensorFlow-related operations, including:
    - Model creation and training
    - Model conversion (TF -> TFLite)
    - Input preprocessing
    - Inference
    
    Designed to gracefully handle cases where TensorFlow is not available.
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the TensorFlow interface.
        
        Args:
            model_dir: Directory where models are/will be stored
        """
        self.model_dir = model_dir
        self.tf_available = TENSORFLOW_AVAILABLE
        self.use_tflite_runtime = USE_TFLITE_RUNTIME
        
        # Ensure model directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Model paths
        self.activity_model_path = os.path.join(model_dir, "activity_recognition.tflite")
        self.working_substate_model_path = os.path.join(model_dir, "working_substate.tflite")
        
        # Metadata paths
        self.activity_metadata_path = os.path.join(model_dir, "activity_metadata.json")
        self.working_substate_metadata_path = os.path.join(model_dir, "substate_metadata.json")
        
        # Load models if available
        self.activity_interpreter = None
        self.working_substate_interpreter = None
        self.load_models()
        
        # Load or create metadata
        self.activity_metadata = self.load_metadata(self.activity_metadata_path)
        self.working_substate_metadata = self.load_metadata(self.working_substate_metadata_path)
        
    def load_models(self):
        """
        Load TensorFlow Lite models if available
        
        Returns:
            bool: True if models were loaded successfully, False otherwise
        """
        if not self.tf_available:
            return False
            
        success = True
        
        # Try to load activity recognition model
        if os.path.exists(self.activity_model_path) and os.path.getsize(self.activity_model_path) > 100:
            try:
                if self.use_tflite_runtime:
                    self.activity_interpreter = tflite.Interpreter(model_path=self.activity_model_path)
                else:
                    self.activity_interpreter = tf.lite.Interpreter(model_path=self.activity_model_path)
                    
                self.activity_interpreter.allocate_tensors()
                self.activity_input_details = self.activity_interpreter.get_input_details()
                self.activity_output_details = self.activity_interpreter.get_output_details()
                logger.info(f"Successfully loaded activity model from {self.activity_model_path}")
            except Exception as e:
                logger.error(f"Failed to load activity model: {e}")
                success = False
        else:
            logger.warning(f"Activity model not found at {self.activity_model_path} or is too small")
            success = False
            
        # Try to load working substate model
        if os.path.exists(self.working_substate_model_path) and os.path.getsize(self.working_substate_model_path) > 100:
            try:
                if self.use_tflite_runtime:
                    self.working_substate_interpreter = tflite.Interpreter(model_path=self.working_substate_model_path)
                else:
                    self.working_substate_interpreter = tf.lite.Interpreter(model_path=self.working_substate_model_path)
                    
                self.working_substate_interpreter.allocate_tensors()
                self.working_substate_input_details = self.working_substate_interpreter.get_input_details()
                self.working_substate_output_details = self.working_substate_interpreter.get_output_details()
                logger.info(f"Successfully loaded working substate model from {self.working_substate_model_path}")
            except Exception as e:
                logger.error(f"Failed to load working substate model: {e}")
                success = False
        else:
            logger.warning(f"Working substate model not found at {self.working_substate_model_path} or is too small")
            success = False
            
        return success
    
    def load_metadata(self, metadata_path):
        """
        Load model metadata from JSON file
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            dict: Metadata or empty dict if not found
        """
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.debug(f"Loaded metadata from {metadata_path}")
                    return metadata
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_path}: {e}")
                
        return {}
    
    def create_placeholder_models(self):
        """
        Create empty placeholder TFLite models and metadata files
        
        Used when TensorFlow is not available but we want to maintain
        the expected file structure.
        
        Returns:
            bool: True if files were created successfully, False otherwise
        """
        try:
            # Create activity recognition model placeholder
            with open(self.activity_model_path, 'wb') as f:
                f.write(b'TFLite Model Placeholder')
                
            # Create working substate model placeholder
            with open(self.working_substate_model_path, 'wb') as f:
                f.write(b'TFLite Model Placeholder')
                
            # Create activity metadata if it doesn't exist
            if not os.path.exists(self.activity_metadata_path):
                activity_metadata = {
                    "name": "Activity Recognition Model",
                    "description": "Classifies user activity based on pose landmarks",
                    "version": "1.0",
                    "author": "Smart Work Focus App",
                    "classes": {
                        "0": "working",
                        "1": "not_working",
                        "2": "distracted_by_others",
                        "3": "on_break",
                        "4": "idle"
                    }
                }
                
                with open(self.activity_metadata_path, 'w') as f:
                    json.dump(activity_metadata, f)
                    
            # Create working substate metadata if it doesn't exist
            if not os.path.exists(self.working_substate_metadata_path):
                working_substate_metadata = {
                    "name": "Working Substate Model",
                    "description": "Classifies specific working activities",
                    "version": "1.0",
                    "author": "Smart Work Focus App",
                    "classes": {
                        "0": None,
                        "1": "typing",
                        "2": "writing",
                        "3": "reading",
                        "4": "phone_use"
                    }
                }
                
                with open(self.working_substate_metadata_path, 'w') as f:
                    json.dump(working_substate_metadata, f)
                    
            logger.info("Created placeholder TFLite model files and metadata")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create placeholder files: {e}")
            return False
            
    def prepare_input_features(self, pose_landmarks, head_angle, movement_level):
        """
        Prepare input features for model inference
        
        Args:
            pose_landmarks: List of pose landmarks from MediaPipe
            head_angle: Calculated head angle in degrees
            movement_level: Movement level (0-100)
            
        Returns:
            numpy array: Input features in format expected by TFLite model
        """
        try:
            # Extract landmark features
            features = []
            
            # Use only x, y, z coordinates from landmarks
            for lm in pose_landmarks:
                if hasattr(lm, 'x') and hasattr(lm, 'y'):
                    features.extend([lm.x, lm.y, lm.z if hasattr(lm, 'z') else 0.0])
                else:
                    # Handle case where landmarks might be in different format
                    features.extend([0.0, 0.0, 0.0])
            
            # Add derived features like head angle and movement level
            features.append(head_angle)
            features.append(movement_level)
            
            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)
            
            # Reshape according to the model's expected input
            if self.tf_available and hasattr(self, 'activity_input_details'):
                input_shape = self.activity_input_details[0]['shape']
                
                # Make sure input has the right dimensions
                if len(features_array) != np.prod(input_shape[1:]):
                    # Pad or truncate to match expected input size
                    target_size = np.prod(input_shape[1:])
                    if len(features_array) < target_size:
                        # Pad with zeros
                        features_array = np.pad(
                            features_array, 
                            (0, target_size - len(features_array)), 
                            'constant'
                        )
                    else:
                        # Truncate
                        features_array = features_array[:target_size]
                
                # Reshape to the expected input shape
                features_array = features_array.reshape(input_shape)
            else:
                # Default reshape (batch size of 1)
                features_array = np.expand_dims(features_array, axis=0)
                
            return features_array
            
        except Exception as e:
            logger.error(f"Error preparing input features: {e}")
            # Return a zero array as fallback
            return np.zeros((1, 100), dtype=np.float32)
    
    def predict_activity(self, input_features):
        """
        Predict activity state using TFLite model
        
        Args:
            input_features: Prepared input features
            
        Returns:
            tuple: (activity_state, confidence)
        """
        if not self.tf_available or self.activity_interpreter is None:
            return None, 0.0
            
        try:
            # Set tensor
            self.activity_interpreter.set_tensor(
                self.activity_input_details[0]['index'],
                input_features
            )
            
            # Run inference
            self.activity_interpreter.invoke()
            
            # Get results
            output = self.activity_interpreter.get_tensor(
                self.activity_output_details[0]['index']
            )
            
            # Get prediction
            prediction_idx = int(np.argmax(output[0]))
            confidence = float(output[0][prediction_idx])
            
            # Map to activity state
            if self.activity_metadata and 'classes' in self.activity_metadata:
                activity_state = self.activity_metadata['classes'].get(str(prediction_idx))
            else:
                # Default mapping if metadata not available
                activity_states = {
                    0: "working",
                    1: "not_working",
                    2: "distracted_by_others",
                    3: "on_break",
                    4: "idle"
                }
                activity_state = activity_states.get(prediction_idx)
                
            return activity_state, confidence
            
        except Exception as e:
            logger.error(f"Error predicting activity state: {e}")
            return None, 0.0
            
    def predict_working_substate(self, input_features):
        """
        Predict working substate using TFLite model
        
        Args:
            input_features: Prepared input features
            
        Returns:
            tuple: (working_substate, confidence)
        """
        if not self.tf_available or self.working_substate_interpreter is None:
            return None, 0.0
            
        try:
            # Set tensor
            self.working_substate_interpreter.set_tensor(
                self.working_substate_input_details[0]['index'],
                input_features
            )
            
            # Run inference
            self.working_substate_interpreter.invoke()
            
            # Get results
            output = self.working_substate_interpreter.get_tensor(
                self.working_substate_output_details[0]['index']
            )
            
            # Get prediction
            prediction_idx = int(np.argmax(output[0]))
            confidence = float(output[0][prediction_idx])
            
            # Map to working substate
            if self.working_substate_metadata and 'classes' in self.working_substate_metadata:
                working_substate = self.working_substate_metadata['classes'].get(str(prediction_idx))
            else:
                # Default mapping if metadata not available
                working_substates = {
                    0: None,  # Generic working
                    1: "typing",
                    2: "writing",
                    3: "reading",
                    4: "phone_use"
                }
                working_substate = working_substates.get(prediction_idx)
                
            return working_substate, confidence
            
        except Exception as e:
            logger.error(f"Error predicting working substate: {e}")
            return None, 0.0
    
    def is_model_available(self):
        """
        Check if TensorFlow models are available and loaded
        
        Returns:
            bool: True if models are available, False otherwise
        """
        return (
            self.tf_available and 
            self.activity_interpreter is not None and 
            self.working_substate_interpreter is not None
        )