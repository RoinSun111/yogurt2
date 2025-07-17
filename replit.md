# SmartDesk Focus Monitor

## Overview

This application is a comprehensive work focus and posture monitoring system designed for sit/stand desks. It uses computer vision, machine learning, and real-time analytics to track user activity, posture, and focus patterns while maintaining privacy through on-device processing.

The system leverages TinyML (TensorFlow Lite) for real-time pose detection and activity classification, providing users with actionable insights about their work habits without compromising privacy.

## System Architecture

### Frontend Architecture
- **Framework**: Flask with Jinja2 templating
- **UI Framework**: Bootstrap 5 with dark theme
- **Real-time Updates**: JavaScript with periodic AJAX calls
- **Charts**: Chart.js for data visualization
- **Icons**: Feather Icons and Font Awesome
- **Camera Integration**: MediaPipe for browser-based pose detection

### Backend Architecture
- **Web Framework**: Flask (Python)
- **WSGI Server**: Gunicorn for production deployment
- **Database ORM**: SQLAlchemy with Flask-SQLAlchemy
- **ML Framework**: TensorFlow Lite for on-device inference
- **Computer Vision**: MediaPipe for pose estimation and face mesh analysis

### Data Storage
- **Primary Database**: PostgreSQL for production (with SQLite fallback for development)
- **Schema**: Relational database with models for:
  - FocusScore: Time-series focus tracking
  - PostureStatus: Detailed posture analysis with angles and quality metrics
  - ActivityStatus: Activity state classification and working substates
  - WaterIntake: Hydration tracking
  - MoodboardSettings: Widget configurations

## Key Components

### 1. Activity Recognition System
- **Purpose**: Classifies user activity into 5 states (working, not_working, distracted_by_others, on_break, idle)
- **Implementation**: TinyML models with pose landmark feature extraction
- **Working Substates**: Further classification into typing, writing, reading, phone_use
- **Files**: `utils/activity_analyzer.py`, `models/activity_recognition.tflite`

### 2. Posture Analysis Engine
- **Purpose**: Real-time posture monitoring with detailed biomechanical analysis
- **Metrics**: Shoulder-hip angle, neck angle, symmetry score, spine curvature
- **Classification**: Excellent/Good/Fair/Poor posture quality with specific feedback
- **Files**: `utils/posture_analyzer.py`

### 3. Focus Calculation
- **Purpose**: Determines focus state based on presence and posture
- **Logic**: Present + upright posture = focused state
- **Scoring**: Daily focus percentage based on time-series data
- **Files**: `utils/focus_calculator.py`

### 4. Eye Tracking Module
- **Purpose**: Advanced gaze direction and blink rate analysis
- **Technology**: MediaPipe Face Mesh for eye landmark detection
- **Metrics**: Gaze direction, focus score, blink rate, screen attention
- **Files**: `utils/eye_tracker.py`

### 5. Camera Processing
- **Purpose**: Handles image preprocessing and camera input
- **Optimization**: Automatic resizing to 640px width for performance
- **Format**: RGB conversion for MediaPipe compatibility
- **Files**: `utils/camera_processor.py`

### 6. Machine Learning Pipeline
- **Training**: Custom TensorFlow models with conversion to TFLite
- **Deployment**: On-device inference with graceful fallbacks
- **Models**: Activity recognition and working substate classification
- **Files**: `utils/model_trainer.py`, `utils/tensorflow_interface.py`

## Data Flow

1. **Image Capture**: Camera input processed through `camera_processor.py`
2. **Pose Detection**: MediaPipe extracts body landmarks
3. **Feature Extraction**: Pose landmarks converted to activity features
4. **ML Inference**: TFLite models classify activity and posture
5. **Focus Calculation**: Combine presence, posture, and activity for focus score
6. **Database Storage**: Time-series data stored in PostgreSQL
7. **Dashboard Updates**: Real-time metrics displayed via AJAX

## External Dependencies

### Python Packages
- **Flask**: Web framework and routing
- **SQLAlchemy**: Database ORM and migrations
- **MediaPipe**: Computer vision and pose detection
- **OpenCV**: Image processing utilities
- **TensorFlow Lite**: On-device ML inference
- **Gunicorn**: Production WSGI server
- **Psycopg2**: PostgreSQL database adapter

### Frontend Libraries
- **Bootstrap 5**: UI framework with dark theme
- **Chart.js**: Data visualization and charts
- **Feather Icons**: Lightweight icon system
- **MediaPipe Web**: Browser-based pose detection

### System Requirements
- **Camera**: USB webcam or integrated camera (480p minimum)
- **Database**: PostgreSQL 16 (with SQLite fallback)
- **Platform**: Linux-based systems (Raspberry Pi compatible)

## Deployment Strategy

### Development Environment
- **Database**: SQLite for rapid prototyping
- **Server**: Flask development server with auto-reload
- **Configuration**: Environment variables for database URLs

### Production Environment
- **Server**: Gunicorn with multi-worker configuration
- **Database**: PostgreSQL with connection pooling
- **Deployment**: Replit autoscale with proxy configuration
- **Port Binding**: Internal port 5000, external port 80

### Privacy and Security
- **On-Device Processing**: All computer vision runs locally
- **No Video Streaming**: Only aggregated metrics are stored
- **Local Inference**: TinyML models prevent data transmission
- **Session Management**: Secure session handling with environment-based secrets

## Changelog
- July 17, 2025: Redesigned moodboard UI following Dieter Rams design principles - clean, functional, minimal aesthetic with subtle colors, refined typography, and reduced visual noise
- July 2, 2025: Enhanced posture detection accuracy with MediaPipe Face Mesh integration for precise neck angle calculation using 468 facial landmarks
- June 28, 2025: Refined UI with WHOOP-inspired design - minimal, data-focused interface with darker backgrounds and cleaner metrics
- June 22, 2025: Added comprehensive gamification system with achievements, streaks, and health score rewards
- June 15, 2025: Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.
UI Design preference: Dieter Rams inspired design - clean, functional, minimal with good typography and subtle visual hierarchy.