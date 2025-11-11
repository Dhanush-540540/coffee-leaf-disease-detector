from flask import Flask, render_template, request, jsonify, Response, send_from_directory, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
from PIL import Image
import base64
import io
import json
from datetime import datetime
import logging
from twilio.rest import Client
import re
import time
import threading
from queue import Queue

# Import configuration
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db = SQLAlchemy(app)
mail = Mail(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Global variables for video processing
camera = None
video_processing = False
frame_queue = Queue()

# Load your trained model
try:
    model = YOLO('best.pt')
    logger.info("✅ best.pt model loaded successfully!")
except Exception as e:
    logger.warning(f"❌ best.pt not found, trying last.pt: {e}")
    try:
        model = YOLO('last.pt')
        logger.info("✅ last.pt model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Both models failed, using yolov8n: {e}")
        model = YOLO('yolov8n.pt')
        logger.info("✅ Using yolov8n as fallback")

# Debug model information
print("=" * 60)
print("🔍 MODEL INFORMATION")
print("=" * 60)
print(f"Model loaded: {model is not None}")
if model:
    print(f"Model classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    print(f"Model device: {model.device}")
print("=" * 60)

# Enhanced disease mapping
def create_enhanced_mapping():
    """Create intelligent mapping based on model classes"""
    if not model or not hasattr(model, 'names'):
        return {}
    
    mapping = {}
    for class_id, class_name in model.names.items():
        class_lower = class_name.lower()
        
        # Enhanced pattern matching
        if any(word in class_lower for word in ['rust', 'hemileia', 'orange', 'yellow', 'powdery']):
            mapping[class_name] = 'rust'
        elif any(word in class_lower for word in ['miner', 'tunnel', 'mine', 'larva', 'worm', 'tunneling']):
            mapping[class_name] = 'miner'
        elif any(word in class_lower for word in ['phoma', 'spot', 'cercospora', 'brown', 'dark', 'black', 'circular']):
            mapping[class_name] = 'phoma'
        elif any(word in class_lower for word in ['healthy', 'good', 'normal', 'green', 'leaf']):
            mapping[class_name] = 'healthy'
        else:
            # Default to healthy for unknown classes
            mapping[class_name] = 'healthy'
    
    return mapping

DYNAMIC_MAPPING = create_enhanced_mapping()
print(f"🤖 Enhanced mapping: {DYNAMIC_MAPPING}")

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Enhanced disease information
DISEASE_INFO = {
    "healthy": {
        "name": "Healthy Leaf",
        "description": "The coffee leaf is healthy with no visible diseases.",
        "treatment": "No treatment needed. Maintain good agricultural practices.",
        "pesticides": [],
        "preventive_measures": [
            "Regular monitoring of plants",
            "Proper spacing for air circulation",
            "Balanced fertilization",
            "Adequate irrigation management"
        ],
        "severity": "None",
        "color": (76, 175, 80),  # Green
        "text_color": "#4CAF50"
    },
    "rust": {
        "name": "Coffee Leaf Rust",
        "description": "Characterized by yellow-orange powdery spots on the undersides of leaves. Caused by fungus Hemileia vastatrix.",
        "treatment": "Immediate fungicide application and cultural controls",
        "pesticides": [
            {"name": "Copper-based Fungicides", "type": "Protective", "dosage": "2-3g/liter", "frequency": "Every 15-21 days"},
            {"name": "Triazole Fungicides", "type": "Systemic", "dosage": "1-1.5ml/liter", "frequency": "Every 30 days"}
        ],
        "application_method": "Spray thoroughly on both sides of leaves, especially undersides",
        "preventive_measures": [
            "Plant resistant varieties",
            "Proper pruning for air circulation",
            "Avoid overhead irrigation",
            "Remove and burn infected leaves"
        ],
        "severity": "High",
        "color": (244, 67, 54),  # Red
        "text_color": "#f44336"
    },
    "miner": {
        "name": "Leaf Miner",
        "description": "Caused by larvae that tunnel through leaves creating winding trails or mines.",
        "treatment": "Insecticide application and biological controls",
        "pesticides": [
            {"name": "Abamectin", "type": "Systemic", "dosage": "0.5-1ml/liter", "frequency": "Every 15 days"},
            {"name": "Spinosad", "type": "Biological", "dosage": "0.5-1ml/liter", "frequency": "Every 10-14 days"}
        ],
        "application_method": "Spray on both sides of leaves, focus on new growth",
        "preventive_measures": [
            "Monitor for adult flies regularly",
            "Remove and destroy heavily infested leaves",
            "Use yellow sticky traps",
            "Encourage natural predators"
        ],
        "severity": "Medium",
        "color": (255, 152, 0),  # Orange
        "text_color": "#FF9800"
    },
    "phoma": {
        "name": "Phoma Leaf Spot", 
        "description": "Irregular brown spots that may cause leaf drop. Caused by Phoma spp. fungus.",
        "treatment": "Fungicide treatment and sanitation practices",
        "pesticides": [
            {"name": "Copper Hydroxide", "type": "Protective", "dosage": "2-3g/liter", "frequency": "Every 15 days"},
            {"name": "Thiophanate-methyl", "type": "Systemic", "dosage": "1-1.5g/liter", "frequency": "Every 21 days"}
        ],
        "application_method": "Spray on leaves and stems, focus on infected areas",
        "preventive_measures": [
            "Improve soil drainage",
            "Avoid wounding plants",
            "Proper sanitation of tools",
            "Remove infected plant parts"
        ],
        "severity": "Medium",
        "color": (139, 69, 19),  # Brown
        "text_color": "#8B4513"
    }
}

def get_disease_info(class_name, confidence):
    base_info = DISEASE_INFO.get(class_name, {
        'name': class_name.title(),
        'description': 'No detailed information available.',
        'treatment': 'Consult with agricultural expert.',
        'severity': 'Unknown',
        'color': (102, 126, 234),
        'text_color': '#667eea'
    })
    
    if confidence > 0.8:
        base_info['reliability'] = 'High'
    elif confidence > 0.6:
        base_info['reliability'] = 'Medium'
    else:
        base_info['reliability'] = 'Low'
    
    return base_info

def map_class_to_disease(original_class):
    """Map the model's class names to our disease categories"""
    return DYNAMIC_MAPPING.get(original_class, 'healthy')

# Enhanced detection functions
def enhanced_detection_pipeline(image):
    """Main detection pipeline with multiple validation steps"""
    try:
        print("\n" + "="*50)
        print("🔄 STARTING ENHANCED DETECTION PIPELINE")
        print("="*50)
        
        original_shape = image.shape
        print(f"📐 Input image: {original_shape[1]}x{original_shape[0]}")
        
        # Step 1: Smart preprocessing
        processed_img = smart_preprocessing(image)
        
        # Step 2: Multi-stage detection
        all_detections = multi_scale_detection(processed_img, original_shape)
        
        # Step 3: Advanced filtering
        filtered_detections = advanced_filtering(all_detections, image.shape)
        
        # Step 4: Create visualization
        result_image = create_detection_visualization(image, filtered_detections)
        
        print(f"📊 FINAL RESULTS: {len(filtered_detections)} valid detections")
        for det in filtered_detections:
            print(f"   ✅ {det['class']} (conf: {det['confidence']:.3f})")
        
        return filtered_detections, result_image
        
    except Exception as e:
        print(f"❌ Detection pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return [], image

def smart_preprocessing(image):
    """Adaptive preprocessing based on image characteristics"""
    h, w = image.shape[:2]
    
    # Resize to optimal size for detection
    target_size = 640
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    print(f"📏 Resized to: {new_w}x{new_h}")
    
    # Only enhance if image is dark or low contrast
    brightness = np.mean(resized)
    contrast = np.std(resized)
    
    if brightness < 100 or contrast < 40:
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        print("🌈 Applied contrast enhancement")
        return enhanced
    
    return resized

def multi_scale_detection(image, original_shape):
    """Run detection at multiple scales and confidence levels"""
    all_detections = []
    
    # Try different confidence thresholds
    conf_thresholds = [0.4, 0.25, 0.15]
    
    for conf in conf_thresholds:
        print(f"🎯 Confidence threshold: {conf}")
        results = model(image, conf=conf, iou=0.5)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detection = process_detection(box, image.shape, original_shape, 1.0)
                if detection and is_meaningful_detection(detection, original_shape):
                    all_detections.append(detection)
    
    return all_detections

def process_detection(box, processed_shape, original_shape, scale):
    """Process individual detection and convert coordinates"""
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = model.names[cls]
    mapped_class = DYNAMIC_MAPPING.get(class_name, 'healthy')
    
    # Get bounding box in processed image coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    # Scale back to original image coordinates
    scale_x = original_shape[1] / processed_shape[1] / scale
    scale_y = original_shape[0] / processed_shape[0] / scale
    
    bbox_original = [
        x1 * scale_x,
        y1 * scale_y, 
        x2 * scale_x,
        y2 * scale_y
    ]
    
    return {
        'class': mapped_class,
        'original_class': class_name,
        'confidence': conf,
        'bbox': bbox_original,
        'disease_info': get_disease_info(mapped_class, conf)
    }

def is_meaningful_detection(detection, image_shape):
    """Validate if detection is meaningful"""
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox
    
    # Check coordinates
    if x1 >= x2 or y1 >= y2:
        return False
    
    # Check if coordinates are within image bounds
    if x1 < 0 or y1 < 0 or x2 > image_shape[1] or y2 > image_shape[0]:
        return False
    
    # Check size
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = image_shape[0] * image_shape[1]
    area_ratio = bbox_area / image_area
    
    if area_ratio < 0.001 or area_ratio > 0.3:
        return False
    
    # Check aspect ratio
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0
    
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
        return False
    
    # Confidence check based on class
    if detection['class'] != 'healthy' and detection['confidence'] < 0.3:
        return False
    
    return True

def advanced_filtering(detections, image_shape):
    """Apply advanced filtering to remove false positives"""
    if not detections:
        return []
    
    # Remove duplicates using NMS
    filtered = []
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    while detections:
        best = detections.pop(0)
        filtered.append(best)
        
        # Remove overlapping detections
        detections = [det for det in detections if calculate_iou(best['bbox'], det['bbox']) < 0.4]
    
    # Group by class and filter unlikely combinations
    return filter_unlikely_combinations(filtered)

def filter_unlikely_combinations(detections):
    """Filter unlikely disease combinations"""
    if not detections:
        return []
    
    classes_present = set(det['class'] for det in detections)
    
    # If we have both healthy and disease detections, apply logic
    if 'healthy' in classes_present and len(classes_present) > 1:
        disease_detections = [det for det in detections if det['class'] != 'healthy']
        healthy_detections = [det for det in detections if det['class'] == 'healthy']
        
        # If we have confident disease detections, keep them and remove healthy
        confident_diseases = [det for det in disease_detections if det['confidence'] > 0.6]
        if confident_diseases:
            return disease_detections
    
    return detections

def create_detection_visualization(image, detections):
    """Create visualization with accurate bounding boxes"""
    result_img = image.copy()
    h, w = image.shape[:2]
    
    # Calculate dynamic sizes
    font_scale = max(0.5, min(h, w) / 800)
    thickness = max(1, int(min(h, w) / 400))
    box_thickness = max(2, int(min(h, w) / 300))
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        class_name = det['class']
        confidence = det['confidence']
        color = det['disease_info']['color']
        
        # Draw bounding box
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, box_thickness)
        
        # Draw label
        label = f"{class_name.upper()} {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Label background
        cv2.rectangle(result_img, 
                     (x1, y1 - label_height - baseline - 5),
                     (x1 + label_width, y1),
                     color, -1)
        
        # Label text
        cv2.putText(result_img, label,
                   (x1, y1 - baseline - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Add overall status
    if not detections or all(det['class'] == 'healthy' for det in detections):
        status_text = "HEALTHY - No diseases detected"
        color = (0, 180, 0)
    else:
        disease_count = len([det for det in detections if det['class'] != 'healthy'])
        status_text = f"FOUND {disease_count} DISEASE SPOTS"
        color = (0, 0, 255)
    
    cv2.putText(result_img, status_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, color, thickness + 1)
    
    return result_img

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    
    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# VIDEO PROCESSING FUNCTION - ADD THIS
def process_video_file(video_path):
    """Process video file and return analysis results"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'status': 'error',
                'message': 'Could not open video file'
            }
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"🎥 Processing video: {total_frames} frames, {fps} FPS")
        
        frame_analysis = []
        frame_count = 0
        diseases_detected = set()
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Analyze every 5th frame to save time (adjust as needed)
            if frame_count % 5 == 0:
                # Resize frame for faster processing
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Run detection on the frame
                detections, _ = enhanced_detection_pipeline(frame_resized)
                
                frame_diseases = set()
                for detection in detections:
                    if detection['class'] != 'healthy' and detection['confidence'] > 0.5:
                        frame_diseases.add(detection['class'])
                        diseases_detected.add(detection['class'])
                
                frame_analysis.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps if fps > 0 else 0,
                    'diseases_detected': list(frame_diseases),
                    'total_detections': len([d for d in detections if d['class'] != 'healthy'])
                })
                
                processed_frames += 1
                if frame_diseases:
                    print(f"📊 Frame {frame_count}: {list(frame_diseases)}")
            
            frame_count += 1
            
            # Limit processing for very long videos
            if frame_count > 1000:  # Process max 1000 frames
                break
        
        cap.release()
        
        # Calculate video duration
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'total_frames_processed': frame_count,
            'frames_analyzed': processed_frames,
            'diseases_detected': list(diseases_detected),
            'frame_analysis': frame_analysis,
            'video_duration': duration,
            'analysis_summary': {
                'total_disease_frames': len([f for f in frame_analysis if f['diseases_detected']]),
                'most_common_disease': max(set([d for f in frame_analysis for d in f['diseases_detected']]), 
                                         key=[d for f in frame_analysis for d in f['diseases_detected']].count) 
                if diseases_detected else 'None'
            },
            'status': 'success',
            'message': f'Processed {processed_frames} frames from video'
        }
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Video processing failed: {str(e)}'
        }

# Test image creation functions
def create_healthy_leaf():
    """Create a healthy leaf image"""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    # Leaf shape
    cv2.ellipse(img, (250, 250), (200, 300), 0, 0, 360, (40, 180, 40), -1)
    cv2.ellipse(img, (250, 250), (190, 290), 0, 0, 360, (30, 160, 30), -1)
    return img

def create_rust_disease():
    """Create rust disease pattern (orange/yellow spots)"""
    img = create_healthy_leaf()
    # Rust spots (orange/yellow)
    for i in range(8):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        size = np.random.randint(10, 25)
        cv2.circle(img, (x, y), size, (0, 100, 255), -1)  # Orange
    return img

def create_phoma_disease():
    """Create phoma disease pattern (brown/dark spots)"""
    img = create_healthy_leaf()
    # Phoma spots (brown)
    for i in range(6):
        x = np.random.randint(100, 400)
        y = np.random.randint(100, 400)
        size = np.random.randint(8, 20)
        cv2.circle(img, (x, y), size, (0, 0, 100), -1)  # Brown
    return img

def generate_frames():
    """Generate frames with real-time detection for webcam"""
    global camera, video_processing
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while video_processing:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run real-time detection with higher confidence
            results = model(frame, conf=0.5)
            
            # Draw detections
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    mapped_class = map_class_to_disease(class_name)
                    
                    # Only show confident disease detections
                    if conf > 0.5 and mapped_class != 'healthy':
                        # Get disease info
                        disease_info = get_disease_info(mapped_class, conf)
                        color = disease_info['color']
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{disease_info['name']} {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                     (x1 + label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Enhanced notification functions with better error handling
def check_email_config():
    """Check if email configuration is properly set up"""
    required_configs = ['MAIL_SERVER', 'MAIL_PORT', 'MAIL_USERNAME', 'MAIL_PASSWORD', 'MAIL_USE_TLS']
    missing_configs = []
    
    for config in required_configs:
        if not app.config.get(config):
            missing_configs.append(config)
    
    if missing_configs:
        print(f"❌ Missing email configurations: {missing_configs}")
        return False
    
    print("✅ Email configuration appears to be set up")
    return True

def check_sms_config():
    """Check if SMS configuration is properly set up"""
    required_configs = ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_PHONE_NUMBER']
    missing_configs = []
    
    for config in required_configs:
        if not app.config.get(config) or app.config.get(config).startswith('your-'):
            missing_configs.append(config)
    
    if missing_configs:
        print(f"❌ Missing or invalid SMS configurations: {missing_configs}")
        return False
    
    print("✅ SMS configuration appears to be set up")
    return True

def send_email_notification(user, detection_results):
    """Send email notification with treatment recommendations"""
    try:
        # Check if email is configured
        if not check_email_config():
            print(f"📧 [EMAIL NOT SENT] Configuration incomplete. Would send to: {user.email}")
            print(f"📧 Analysis completed for user: {user.username}")
            return {
                'sent': False,
                'reason': 'Email configuration incomplete',
                'message': 'Email notifications are not configured'
            }

        # Determine subject based on results
        if detection_results['metrics']['leaf_status'] == 'healthy':
            subject = f"✅ Healthy Coffee Leaf Report - {datetime.now().strftime('%Y-%m-%d')}"
        else:
            subject = f"⚠️ Coffee Leaf Disease Detection Results - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Create email content
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #4a7c2a; text-align: center;">☕ Coffee Leaf Disease Detection Report</h2>
                <p>Hello <strong>{user.username}</strong>,</p>
                <p>Here are the results of your coffee leaf analysis:</p>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <h3 style="color: #4a7c2a;">📊 Detection Summary</h3>
                    <p><strong>Leaf Status:</strong> {detection_results['metrics']['leaf_status'].title()}</p>
                    <p><strong>Total Disease Spots Found:</strong> {detection_results['metrics']['spots_found']}</p>
                    <p><strong>Diseases Identified:</strong> {detection_results['metrics']['diseases_identified']}</p>
                    <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                </div>
        """
        
        if detection_results['predictions'] and detection_results['metrics']['leaf_status'] != 'healthy':
            # Group predictions by disease type
            disease_groups = {}
            for pred in detection_results['predictions']:
                if pred['class'] != 'healthy':
                    disease_class = pred['class']
                    if disease_class not in disease_groups:
                        disease_groups[disease_class] = []
                    disease_groups[disease_class].append(pred)
            
            for disease_class, predictions in disease_groups.items():
                disease_info = predictions[0]['disease_info']
                avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
                
                html_content += f"""
                <div style="border-left: 4px solid {disease_info['text_color']}; padding: 15px; margin: 15px 0; background: #fff;">
                    <h3 style="color: {disease_info['text_color']}; margin-top: 0;">
                        {disease_info['name']} ({len(predictions)} spots, {avg_confidence:.1%} confidence)
                    </h3>
                    <p><strong>Description:</strong> {disease_info['description']}</p>
                    <p><strong>Severity:</strong> {disease_info['severity']}</p>
                    
                    <h4 style="color: #4a7c2a;">🛡️ Recommended Treatments:</h4>
                """
                
                if disease_info.get('pesticides'):
                    for pesticide in disease_info['pesticides']:
                        html_content += f"""
                        <div style="background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 5px;">
                            <strong>{pesticide['name']}</strong> ({pesticide['type']})<br>
                            <small>Dosage: {pesticide['dosage']} | Frequency: {pesticide['frequency']}</small>
                        </div>
                        """
                else:
                    html_content += "<p>No chemical treatment needed. Focus on preventive measures.</p>"
                
                html_content += f"""
                    <h4 style="color: #4a7c2a;">🌱 Preventive Measures:</h4>
                    <ul>
                """
                for measure in disease_info['preventive_measures']:
                    html_content += f"<li>{measure}</li>"
                
                html_content += """
                    </ul>
                </div>
                """
        else:
            html_content += """
            <div style="text-align: center; padding: 20px; background: #e8f5e8; border-radius: 8px;">
                <h3 style="color: #4a7c2a;">✅ No Diseases Detected</h3>
                <p>Your coffee plants appear healthy! Continue with good agricultural practices.</p>
                <h4 style="color: #4a7c2a;">🌱 Maintenance Tips:</h4>
                <ul style="text-align: left; display: inline-block;">
                    <li>Continue regular monitoring</li>
                    <li>Maintain proper watering schedule</li>
                    <li>Ensure adequate sunlight</li>
                    <li>Practice good soil management</li>
                </ul>
            </div>
            """
        
        html_content += """
                <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 8px;">
                    <h4 style="color: #1976d2;">⚠️ Important Safety Notes:</h4>
                    <ul>
                        <li>Always wear protective equipment when applying pesticides</li>
                        <li>Follow recommended waiting periods before harvest</li>
                        <li>Consult local agricultural experts for specific recommendations</li>
                        <li>Test treatments on small areas first</li>
                    </ul>
                </div>
                
                <hr style="margin: 30px 0;">
                <p style="text-align: center; color: #666; font-size: 0.9em;">
                    This report was generated by Coffee Leaf Disease Detection System<br>
                    For questions, please contact our support team.
                </p>
            </div>
        </body>
        </html>
        """
        
        msg = Message(
            subject=subject,
            recipients=[user.email],
            html=html_content,
            sender=app.config.get('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])
        )
        
        mail.send(msg)
        logger.info(f"✅ Email notification sent to {user.email}")
        return {
            'sent': True,
            'to': user.email,
            'message': 'Email sent successfully'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")
        return {
            'sent': False,
            'reason': str(e),
            'message': f'Failed to send email: {str(e)}'
        }

def send_sms_notification(user, detection_results):
    """Send SMS notification with key findings"""
    try:
        # Check if SMS is configured
        if not check_sms_config():
            print(f"📱 [SMS NOT SENT] Configuration incomplete. Would send to: {user.phone}")
            return {
                'sent': False,
                'reason': 'SMS configuration incomplete',
                'message': 'SMS notifications are not configured'
            }

        if not user.phone:
            logger.warning("User phone number not available")
            return {
                'sent': False,
                'reason': 'No phone number',
                'message': 'User phone number not available'
            }

        # Clean phone number
        phone = re.sub(r'\D', '', user.phone)
        if not phone.startswith('+'):
            phone = '+91' + phone  # Default to India, adjust as needed

        client = Client(app.config['TWILIO_ACCOUNT_SID'], app.config['TWILIO_AUTH_TOKEN'])
        
        if detection_results['predictions'] and detection_results['metrics']['leaf_status'] != 'healthy':
            diseases = list(set(pred['class'] for pred in detection_results['predictions'] if pred['class'] != 'healthy'))
            disease_names = [DISEASE_INFO.get(d, {'name': d.title()})['name'] for d in diseases]
            message_body = f"""☕ Coffee Leaf Analysis Complete!

Diseases found: {', '.join(disease_names)}
Spots detected: {detection_results['metrics']['spots_found']}

Check your email for detailed treatment recommendations.

Stay safe! Wear protective gear when applying treatments."""
        else:
            message_body = """☕ Coffee Leaf Analysis Complete!

Great news! No diseases detected. Your plants are healthy.

Continue with good agricultural practices.

Check your email for full report."""
        
        message = client.messages.create(
            body=message_body.strip(),
            from_=app.config['TWILIO_PHONE_NUMBER'],
            to=phone
        )
        
        logger.info(f"✅ SMS notification sent to {user.phone}")
        return {
            'sent': True,
            'to': user.phone,
            'message': 'SMS sent successfully'
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to send SMS: {e}")
        return {
            'sent': False,
            'reason': str(e),
            'message': f'Failed to send SMS: {str(e)}'
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return render_template('login.html', register=True)
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('login.html', register=True)
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('login.html', register=True)
        
        new_user = User(username=username, email=email, phone=phone)
        new_user.set_password(password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('login.html', register=True)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'classes_available': list(model.names.values()) if model else []
    })

@app.route('/model_info')
def model_info():
    """Detailed information about the loaded model"""
    try:
        model_info = {
            'model_loaded': model is not None,
            'model_path': getattr(model, 'ckpt_path', 'Unknown'),
            'model_device': str(model.device),
            'model_classes': dict(model.names) if model else {},
            'num_classes': len(model.names) if model else 0,
            'dynamic_mapping': DYNAMIC_MAPPING
        }
        
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/diagnose_model')
def diagnose_model():
    """Diagnose model performance with test images"""
    try:
        # Test with sample images
        test_images = {
            'healthy': create_healthy_leaf(),
            'rust': create_rust_disease(),
            'phoma': create_phoma_disease()
        }
        
        results = {}
        for name, img in test_images.items():
            detections, _ = enhanced_detection_pipeline(img)
            results[name] = {
                'detections': detections,
                'expected': name,
                'matched': any(d['class'] == name for d in detections)
            }
        
        return jsonify({
            'model_classes': dict(model.names),
            'mapping': DYNAMIC_MAPPING,
            'test_results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/check_notifications')
def check_notifications():
    """Check notification configuration status"""
    email_status = check_email_config()
    sms_status = check_sms_config()
    
    return jsonify({
        'email_configured': email_status,
        'sms_configured': sms_status,
        'email_config': {
            'MAIL_SERVER': app.config.get('MAIL_SERVER'),
            'MAIL_PORT': app.config.get('MAIL_PORT'),
            'MAIL_USERNAME': '***' if app.config.get('MAIL_USERNAME') else None,
            'MAIL_USE_TLS': app.config.get('MAIL_USE_TLS')
        },
        'sms_config': {
            'TWILIO_ACCOUNT_SID': '***' if app.config.get('TWILIO_ACCOUNT_SID') else None,
            'TWILIO_PHONE_NUMBER': app.config.get('TWILIO_PHONE_NUMBER')
        }
    })

# Enhanced prediction endpoint
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Enhanced prediction endpoint with improved detection"""
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    try:
        # Read image
        image = Image.open(file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Check image size
        if image.size[0] < 100 or image.size[1] < 100:
            return jsonify({'status': 'error', 'message': 'Image too small'})
        
        # Convert to OpenCV
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        print(f"📸 Processing uploaded image: {image.width}x{image.height}")
        
        # Use enhanced detection pipeline
        detections, result_image = enhanced_detection_pipeline(image_cv)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate metrics
        disease_detections = [d for d in detections if d['class'] != 'healthy']
        leaf_status = 'healthy' if not disease_detections else 'diseased'
        
        response = {
            'predictions': detections,
            'image_with_boxes': image_base64,
            'metrics': {
                'total_detections': len(detections),
                'spots_found': len(disease_detections),
                'diseases_identified': len(set(d['class'] for d in disease_detections)),
                'leaf_status': leaf_status
            },
            'status': 'success',
            'message': f'Found {len(disease_detections)} disease spots' if disease_detections else 'Leaf is healthy'
        }
        
        # Send notifications and include their status in response
        email_result = send_email_notification(current_user, response)
        sms_result = send_sms_notification(current_user, response)
        
        response['notifications'] = {
            'email': email_result,
            'sms': sms_result
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'Error processing image: {str(e)}'
        })

# Video Upload and Processing
@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    """Handle video upload and processing"""
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video uploaded'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No video selected'})
    
    try:
        # Save uploaded video
        filename = secure_filename(file.filename)
        video_path = os.path.join('static', 'uploads', f"{int(time.time())}_{filename}")
        file.save(video_path)
        
        print(f"🎥 Video saved: {video_path}")
        
        # Process video using the defined function
        result = process_video_file(video_path)
        result['video_path'] = video_path
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'Error processing video: {str(e)}'
        })

# Live Camera Routes
@app.route('/start_webcam')
@login_required
def start_webcam():
    """Start webcam for live detection"""
    global video_processing
    video_processing = True
    return jsonify({'status': 'success', 'message': 'Webcam started'})

@app.route('/stop_webcam')
@login_required
def stop_webcam():
    """Stop webcam"""
    global video_processing, camera
    video_processing = False
    if camera:
        camera.release()
        camera = None
    return jsonify({'status': 'success', 'message': 'Webcam stopped'})

@app.route('/video_feed')
@login_required
def video_feed():
    """Video feed route for live camera"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_image', methods=['POST'])
@login_required
def capture_image():
    """Capture image from webcam and analyze it"""
    global camera
    
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
        
        success, frame = camera.read()
        if success:
            # Analyze the captured frame
            detections, result_image = enhanced_detection_pipeline(frame)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', result_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate metrics
            disease_detections = [d for d in detections if d['class'] != 'healthy']
            leaf_status = 'healthy' if not disease_detections else 'diseased'
            
            response = {
                'predictions': detections,
                'image_with_boxes': image_base64,
                'metrics': {
                    'total_detections': len(detections),
                    'spots_found': len(disease_detections),
                    'diseases_identified': len(set(d['class'] for d in disease_detections)),
                    'leaf_status': leaf_status
                },
                'status': 'success',
                'message': f'Found {len(disease_detections)} disease spots' if disease_detections else 'Leaf is healthy',
                'timestamp': datetime.now().isoformat(),
                'image_size': f"{frame.shape[1]}x{frame.shape[0]}"
            }
            
            # Send notifications and include their status in response
            email_result = send_email_notification(current_user, response)
            sms_result = send_sms_notification(current_user, response)
            
            response['notifications'] = {
                'email': email_result,
                'sms': sms_result
            }
            
            return jsonify(response)
        else:
            return jsonify({'status': 'error', 'message': 'Failed to capture image from camera'})
            
    except Exception as e:
        logger.error(f"Image capture error: {e}")
        return jsonify({
            'status': 'error', 
            'message': f'Error capturing image: {str(e)}'
        })

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    # Create uploads directory if it doesn't exist
    os.makedirs('static/uploads', exist_ok=True)
    
    print("🚀 Starting Enhanced Coffee Leaf Disease Detection System...")
    print("🎯 Improved Features:")
    print("   - Enhanced detection pipeline with multi-stage validation")
    print("   - Smart preprocessing and filtering")
    print("   - Accurate bounding box placement")
    print("   - Better disease classification")
    print("   - Improved notification handling with better error reporting")
    print("   - Video processing functionality added")
    print("📧 Notifications: Email & SMS alerts (with configuration checks)")
    print("🔐 Authentication: User login/registration")
    print("🌐 Access: http://localhost:5000")
    print("\n🔧 Available Routes:")
    print("   - /dashboard (Main interface)")
    print("   - /model_info (Model diagnostics)")
    print("   - /diagnose_model (Test model with sample images)")
    print("   - /check_notifications (Check notification configuration)")
    print("   - /predict (Enhanced image analysis)")
    print("   - /upload_video (Video analysis) - NOW FIXED!")
    print("   - /video_feed (Live camera stream)")
    print("   - /capture_image (Capture from camera)")
    
    # Check notification configurations
    print("\n🔔 Checking notification configurations...")
    check_email_config()
    check_sms_config()
    
    app.run(debug=True, host='0.0.0.0', port=5000)