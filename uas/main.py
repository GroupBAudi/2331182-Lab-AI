"""
Two-Stage Emotion Detection System
- Stage 1: Face Detection using YOLO
- Stage 2: Emotion Classification
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class EmotionDetector:
    """Real-time emotion detection system"""
    
    def __init__(self, face_model='yolov8n-face.pt', emotion_model='best_emotion_model.pt'):
        """
        Initialize detector with pre-trained models
        
        Args:
            face_model: Path to face detection model
            emotion_model: Path to emotion classification model
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}\n")
        
        # Load models
        print("Loading models...")
        try:
            self.face_detector = YOLO(face_model)
            print(f"✓ Face detector loaded: {face_model}")
        except:
            print(f"⚠ {face_model} not found, using yolov8n.pt")
            self.face_detector = YOLO('yolov8n.pt')
        
        if not Path(emotion_model).exists():
            raise FileNotFoundError(f"Emotion model not found: {emotion_model}")
        
        self.emotion_classifier = YOLO(emotion_model)
        print(f"✓ Emotion classifier loaded: {emotion_model}\n")
    
    def detect_emotions(self, frame, conf_face=0.5, conf_emotion=0.3):
        """
        Detect emotions in a single frame
        
        Returns:
            annotated_frame: Frame with bounding boxes and labels
            emotions: List of detected emotions with confidence
        """
        # Stage 1: Detect faces
        face_results = self.face_detector(frame, conf=conf_face, verbose=False)
        
        if not face_results or not face_results[0].boxes or len(face_results[0].boxes) == 0:
            return frame, []
        
        # Stage 2: Classify emotions
        emotions = []
        annotated = frame.copy()
        
        for box in face_results[0].boxes:
            # Extract face region
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Classify emotion
            emotion_results = self.emotion_classifier(face_crop, conf=conf_emotion, verbose=False)
            
            if emotion_results:
                probs = emotion_results[0].probs
                emotion = emotion_results[0].names[probs.top1]
                confidence = probs.top1conf.item()
                
                emotions.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                # Draw annotations
                color = self._get_color(emotion)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                label = f"{emotion}: {confidence:.0%}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated, emotions
    
    def process_webcam(self, conf_face=0.5, conf_emotion=0.3):
        """Run real-time emotion detection from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("Starting webcam... Press 'q' to quit\n")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect emotions
            annotated, emotions = self.detect_emotions(frame, conf_face, conf_emotion)
            
            # Add info overlay
            info = f"Frame: {frame_count} | Faces: {len(emotions)}"
            cv2.putText(annotated, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Emotion Detection', annotated)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Processed {frame_count} frames")
    
    def process_image(self, image_path, output_path=None, conf_face=0.5, conf_emotion=0.3):
        """Process a single image file"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read image {image_path}")
            return
        
        annotated, emotions = self.detect_emotions(img, conf_face, conf_emotion)
        
        print(f"\nDetected {len(emotions)} face(s):")
        for i, e in enumerate(emotions, 1):
            print(f"  Face {i}: {e['emotion']} ({e['confidence']:.0%})")
        
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"\n✓ Saved to: {output_path}")
        
        # Display result
        cv2.imshow('Result', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_video(self, video_path, output_path=None, conf_face=0.5, conf_emotion=0.3):
        """Process video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Setup video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"Processing video... Press 'q' to quit\n")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            annotated, emotions = self.detect_emotions(frame, conf_face, conf_emotion)
            
            info = f"Frame: {frame_count} | Faces: {len(emotions)}"
            cv2.putText(annotated, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Processing Video', annotated)
            
            if output_path:
                out.write(annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Processed {frame_count} frames")
    
    def _get_color(self, emotion):
        """Map emotions to colors"""
        colors = {
            'angry': (0, 0, 255),
            'disgusted': (0, 128, 128),
            'fearful': (128, 0, 128),
            'happy': (0, 255, 0),
            'neutral': (128, 128, 128),
            'sad': (255, 0, 0),
            'surprised': (0, 255, 255)
        }
        return colors.get(emotion.lower(), (255, 255, 255))


def main():
    """Main function with usage examples"""
    print("=" * 60)
    print("TWO-STAGE EMOTION DETECTION SYSTEM")
    print("=" * 60 + "\n")
    
    # Initialize detector
    detector = EmotionDetector(
        face_model='yolov8n-face.pt',
        emotion_model='best_emotion_model.pt'
    )
    
    detector.process_webcam()

if __name__ == "__main__":
    main()