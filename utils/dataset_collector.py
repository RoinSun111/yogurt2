
import cv2
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime
import mediapipe as mp
from .camera_processor import CameraProcessor
from .posture_analyzer import PostureAnalyzer

logger = logging.getLogger(__name__)

class PostureDatasetCollector:
    """
    Collect real posture data for training improved models
    """
    
    def __init__(self, data_dir="posture_data"):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "posture_dataset.csv")
        self.images_dir = os.path.join(data_dir, "images")
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize components
        self.camera_processor = CameraProcessor()
        self.posture_analyzer = PostureAnalyzer()
        
        # Posture classes for labeling
        self.posture_classes = [
            'sitting_straight',
            'hunching_over', 
            'left_sitting',
            'right_sitting',
            'leaning_forward',
            'lying',
            'standing'
        ]
    
    def collect_labeled_sample(self, image_path, posture_label):
        """
        Collect a single labeled sample from an image
        """
        try:
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return False
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analyze posture to get landmarks
            posture_data = self.posture_analyzer.analyze(image_rgb)
            
            if not posture_data['is_present']:
                logger.warning("No person detected in image")
                return False
            
            # Extract landmarks
            landmarks = posture_data.get('_raw_landmarks')
            if not landmarks:
                logger.warning("No landmarks extracted")
                return False
            
            # Extract SitPose features
            features = self.posture_analyzer.sitpose_trainer.extract_sitpose_features(landmarks)
            
            # Create sample record
            sample = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'posture_label': posture_label,
                'angle': posture_data['angle'],
                'neck_angle': posture_data['neck_angle'],
                'shoulder_alignment': posture_data['shoulder_alignment'],
                'head_forward_position': posture_data['head_forward_position'],
                'spine_curvature': posture_data['spine_curvature'],
                'symmetry_score': posture_data['symmetry_score']
            }
            
            # Add all 42 features
            feature_names = self.posture_analyzer.sitpose_trainer._get_feature_names()
            for i, feature_name in enumerate(feature_names):
                sample[f'feature_{i:02d}_{feature_name}'] = features[i]
            
            # Save to CSV
            df = pd.DataFrame([sample])
            
            if os.path.exists(self.data_file):
                df.to_csv(self.data_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.data_file, index=False)
            
            logger.info(f"Sample collected: {posture_label} from {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting sample: {str(e)}")
            return False
    
    def load_dataset(self):
        """
        Load the collected dataset
        """
        if not os.path.exists(self.data_file):
            logger.warning("No dataset file found")
            return None, None
        
        try:
            df = pd.read_csv(self.data_file)
            
            # Extract features (columns starting with 'feature_')
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            X = df[feature_cols].values
            
            # Extract labels
            y = df['posture_label'].values
            
            # Convert labels to numeric
            label_map = {label: i for i, label in enumerate(self.posture_classes)}
            y_numeric = np.array([label_map.get(label, -1) for label in y])
            
            # Filter out unknown labels
            valid_indices = y_numeric >= 0
            X = X[valid_indices]
            y_numeric = y_numeric[valid_indices]
            
            logger.info(f"Loaded dataset with {len(X)} samples and {X.shape[1]} features")
            return X, y_numeric
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None, None
    
    def train_with_collected_data(self):
        """
        Train the SitPose model using collected real data
        """
        X, y = self.load_dataset()
        
        if X is None or len(X) < 50:  # Need minimum samples
            logger.warning("Not enough real data, using synthetic dataset")
            self.posture_analyzer.sitpose_trainer.train_model()
            return
        
        logger.info(f"Training with {len(X)} real samples")
        model, scaler, accuracy = self.posture_analyzer.sitpose_trainer.train_model(X, y)
        
        return model, scaler, accuracy
    
    def get_collection_stats(self):
        """
        Get statistics about the collected dataset
        """
        if not os.path.exists(self.data_file):
            return {"total_samples": 0, "class_distribution": {}}
        
        try:
            df = pd.read_csv(self.data_file)
            
            stats = {
                "total_samples": len(df),
                "class_distribution": df['posture_label'].value_counts().to_dict(),
                "date_range": {
                    "first": df['timestamp'].min(),
                    "last": df['timestamp'].max()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"total_samples": 0, "class_distribution": {}}

# Command line interface for data collection
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect posture dataset")
    parser.add_argument("--collect", type=str, help="Image path to collect")
    parser.add_argument("--label", type=str, help="Posture label", 
                       choices=['sitting_straight', 'hunching_over', 'left_sitting', 
                               'right_sitting', 'leaning_forward', 'lying', 'standing'])
    parser.add_argument("--train", action="store_true", help="Train model with collected data")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    collector = PostureDatasetCollector()
    
    if args.collect and args.label:
        success = collector.collect_labeled_sample(args.collect, args.label)
        print(f"Collection {'successful' if success else 'failed'}")
    
    elif args.train:
        collector.train_with_collected_data()
        print("Training completed")
    
    elif args.stats:
        stats = collector.get_collection_stats()
        print(f"Dataset statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Class distribution: {stats['class_distribution']}")
    
    else:
        parser.print_help()
