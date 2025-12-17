import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import shutil


class TwoStageEmotionDetector:
    """
    Two-stage emotion detection system:
    Stage 1: Detect faces using pre-trained YOLO
    Stage 2: Classify emotions from detected face regions
    """
    
    def __init__(self, face_model='yolov8n-face.pt', emotion_model=None, device='cuda'):
        """
        Initialize the two-stage detector.
        
        Args:
            face_model: Pre-trained face detection model path
            emotion_model: Trained emotion classifier path (None for training mode)
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load face detection model
        print("Loading face detection model...")
        try:
            self.face_detector = YOLO(face_model)
            print(f"✓ Face detector loaded: {face_model}")
        except:
            print(f"⚠ {face_model} not found, falling back to yolov8n.pt")
            print("  For better results, download yolov8n-face.pt from:")
            print("  https://github.com/akanametov/yolov8-face")
            self.face_detector = YOLO('yolov8n.pt')
        
        # Load or initialize emotion classifier
        if emotion_model and Path(emotion_model).exists():
            self.emotion_classifier = YOLO(emotion_model)
            print(f"✓ Emotion classifier loaded: {emotion_model}")
        else:
            self.emotion_classifier = YOLO('yolov8n-cls.pt')
            print("✓ Emotion classifier initialized for training")
    
    def train_emotion_classifier(self, dataset_dir, epochs=20, imgsz=224, batch=96, patience=15):
        """
        Train the emotion classification model.
        Face detector remains pre-trained and is not modified.
        
        Dataset structure expected:
        dataset_dir/
            train/
                angry/
                happy/
                sad/
                ...
            test/ (or val/)
                angry/
                happy/
                ...
        
        Args:
            dataset_dir: Root directory containing train/test folders
            epochs: Number of training epochs
            imgsz: Input image size
            batch: Batch size
            patience: Early stopping patience (epochs without improvement)
        """
        print(f"\n{'='*60}")
        print("Training Emotion Classifier")
        print(f"{'='*60}\n")
        
        results = self.emotion_classifier.train(
            data=dataset_dir,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            patience=patience,
            save=True,
            plots=True,
            project='emotion_runs',
            name='emotion_classifier'
        )
        
        print("\n✓ Training completed!")
        
        # Automatically copy best model to root directory
        best_model_path = Path('emotion_runs/emotion_classifier/weights/best.pt')
        root_model_path = Path('best_emotion_model.pt')
        
        if best_model_path.exists():
            shutil.copy2(best_model_path, root_model_path)
            print(f"✓ Best model automatically saved to: {root_model_path.absolute()}")
        
        return results
    
    def validate_emotion_classifier(self, dataset_dir):
        """
        Validate the trained emotion classifier.
        
        Args:
            dataset_dir: Root directory containing validation data
        """
        print("\nValidating emotion classifier...")
        return self.emotion_classifier.val(data=dataset_dir)
    
    def detect_and_classify(self, image_path, conf_face=0.5, conf_emotion=0.3, 
                           save_output=False, output_path=None):
        """
        Perform two-stage emotion detection on an image.
        
        Args:
            image_path: Path to image file or numpy array
            conf_face: Confidence threshold for face detection (0.0-1.0)
            conf_emotion: Confidence threshold for emotion classification (0.0-1.0)
            save_output: Whether to save annotated output image
            output_path: Path for saving annotated image
            
        Returns:
            annotated_img: Image with bounding boxes and labels
            emotions_detected: List of detected emotions with metadata
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image {image_path}")
                return None, []
        else:
            img = image_path
        
        # Stage 1: Detect faces
        face_results = self.face_detector(img, conf=conf_face, verbose=False)
        
        # Check if any faces detected
        if (face_results is None or len(face_results) == 0 or 
            face_results[0].boxes is None or len(face_results[0].boxes) == 0):
            return img, []
        
        # Stage 2: Classify emotion for each detected face
        emotions_detected = []
        annotated_img = img.copy()
        
        for box in face_results[0].boxes:
            # Extract face region
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face_crop = img[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue
            
            # Classify emotion
            emotion_results = self.emotion_classifier(face_crop, conf=conf_emotion, verbose=False)
            
            if len(emotion_results) > 0:
                probs = emotion_results[0].probs
                top_class = probs.top1
                confidence = probs.top1conf.item()
                emotion = emotion_results[0].names[top_class]
                
                emotions_detected.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
                
                # Annotate image
                color = self._get_emotion_color(emotion)
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label = f"{emotion}: {confidence:.2%}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Print results
        print(f"\nDetected {len(emotions_detected)} face(s):")
        for i, result in enumerate(emotions_detected, 1):
            print(f"  Face {i}: {result['emotion']} ({result['confidence']:.2%})")
        
        # Save output if requested
        if save_output and output_path:
            cv2.imwrite(output_path, annotated_img)
            print(f"\n✓ Saved annotated image to: {output_path}")
        
        return annotated_img, emotions_detected
    
    def process_video(self, video_source, output_path=None, conf_face=0.5, conf_emotion=0.3):
        """
        Process video file or webcam stream for emotion detection.
        
        Args:
            video_source: Video file path, 'webcam', or camera index (0, 1, etc.)
            output_path: Optional path to save output video
            conf_face: Face detection confidence threshold
            conf_emotion: Emotion classification confidence threshold
        """
        # Open video source
        if video_source == 'webcam':
            video_source = 0
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Setup video writer if output requested
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("\nProcessing video... Press 'q' to quit")
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, emotions = self.detect_and_classify(
                frame, conf_face=conf_face, conf_emotion=conf_emotion
            )
            
            # Always use the frame (annotated or original)
            display_frame = annotated_frame if annotated_frame is not None else frame
            
            # Add frame info overlay
            info_text = f"Frame: {frame_count} | Faces: {len(emotions) if emotions else 0}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Emotion Detection', display_frame)
            
            # Save frame if recording
            if output_path:
                out.write(display_frame)
            
            # Check for quit command (MUST be after imshow)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        print(f"\n✓ Video processing completed! Processed {frame_count} frames")
    
    def _get_emotion_color(self, emotion):
        """Map emotions to BGR colors for visualization."""
        colors = {
            'angry': (0, 0, 255),        # Red
            'disgusted': (0, 128, 128),  # Teal
            'fearful': (128, 0, 128),    # Purple
            'happy': (0, 255, 0),        # Green
            'neutral': (128, 128, 128),  # Gray
            'sad': (255, 0, 0),          # Blue
            'surprised': (0, 255, 255),  # Yellow
            'nothing': (200, 200, 200)   # Light gray
        }
        return colors.get(emotion.lower(), (255, 255, 255))
    
    def export_emotion_model(self, format='onnx'):
        """
        Export emotion classifier to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
        """
        print(f"Exporting emotion classifier to {format}...")
        self.emotion_classifier.export(format=format)
        
        # Copy exported model to root directory
        if format == 'onnx':
            onnx_path = Path('emotion_runs/emotion_classifier/weights/best.onnx')
            root_onnx_path = Path('best_emotion_model.onnx')
            if onnx_path.exists():
                shutil.copy2(onnx_path, root_onnx_path)
                print(f"✓ ONNX model saved to: {root_onnx_path.absolute()}")
        
        print("✓ Export completed!")


def check_trained_model():
    """
    Check if a trained emotion model already exists.
    
    Returns:
        Path to trained model if exists, None otherwise
    """
    # Check multiple possible locations
    possible_paths = [
        Path('best_emotion_model.pt'),  # Root directory
        Path('emotion_runs/emotion_classifier/weights/best.pt'),  # Training output
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"\n✓ Found trained model: {path.absolute()}")
            return str(path)
    
    return None


def main():
    """Main training and inference pipeline with automatic model detection."""
    
    print("\n" + "="*60)
    print("TWO-STAGE EMOTION DETECTION SYSTEM")
    print("="*60 + "\n")
    
    # Check if trained model exists
    trained_model_path = check_trained_model()
    
    if trained_model_path:
        print("\n" + "="*60)
        print("TRAINED MODEL FOUND - SKIPPING TRAINING")
        print("="*60)
        print(f"\nUsing existing model: {trained_model_path}")
        print("\nTo retrain, delete or rename the model file:")
        print("  - best_emotion_model.pt")
        print("  - emotion_runs/emotion_classifier/weights/best.pt\n")
        
        # Load model for inference
        detector = TwoStageEmotionDetector(
            face_model='yolov8n-face.pt',
            emotion_model=trained_model_path,
            device='cuda'
        )
        
    else:
        print("\n" + "="*60)
        print("NO TRAINED MODEL FOUND - STARTING TRAINING")
        print("="*60 + "\n")
        
        # Initialize detector for training
        detector = TwoStageEmotionDetector(
            face_model='yolov8n-face.pt',
            device='cuda'
        )
        
        # Configure your dataset path
        dataset_path = "data"
        
        # Check if dataset exists
        if not Path(dataset_path).exists():
            print(f"\n⚠ Error: Dataset directory not found: {dataset_path}")
            print("Please create dataset with structure:")
            print("  data/")
            print("    train/")
            print("      angry/")
            print("      happy/")
            print("      ...")
            print("    test/")
            print("      angry/")
            print("      happy/")
            print("      ...")
            return
        
        # Train the emotion classifier
        detector.train_emotion_classifier(
            dataset_path,
            epochs=20,
            batch=96,
            imgsz=224,
            patience=15
        )
        
        # Validate trained model
        detector.validate_emotion_classifier(dataset_path)
        
        # Export model
        print("\n" + "="*60)
        print("EXPORTING MODEL")
        print("="*60 + "\n")
        detector.export_emotion_model(format='onnx')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print("\nModel saved to root directory:")
        print("  - best_emotion_model.pt")
        print("  - best_emotion_model.onnx\n")
    
    # ==================== INFERENCE MODE ====================
    
    print("\n" + "="*60)
    print("STARTING INFERENCE")
    print("="*60 + "\n")
    
    # Example usage - uncomment what you need:
    
    # Test on single image
    # detector.detect_and_classify(
    #     'test_image.jpg',
    #     save_output=True,
    #     output_path='result.jpg'
    # )
    
    # Test on video file
    # detector.process_video(
    #     'test_video.mp4',
    #     output_path='output_video.mp4'
    # )
    
    # Real-time webcam detection
    detector.process_video('webcam')


if __name__ == "__main__":
    main()