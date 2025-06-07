#!/usr/bin/env python3
"""
Person Detection with YOLO11n ONNX Model
This script processes images in the 'splitted' folder and extracts person detections
using the YOLO11n ONNX model, saving the cropped person images to a 'detection' folder.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

class PersonDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Initialize the Person Detector with YOLO11n ONNX model
        
        Args:
            model_path: Path to the YOLO11n ONNX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input shape
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        # COCO class names (person is class 0)
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        print(f"Loaded YOLO11n model from {model_path}")
        print(f"Input shape: {input_shape}")
        print(f"Input size: {self.input_width}x{self.input_height}")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for YOLO inference
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image, scale_x, scale_y
        """
        original_height, original_width = image.shape[:2]
        
        # Calculate scale factors
        scale_x = self.input_width / original_width
        scale_y = self.input_height / original_height
        
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB and normalize
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale_x, scale_y

    def postprocess_detections(self, outputs: List[np.ndarray], scale_x: float, scale_y: float, 
                             original_width: int, original_height: int) -> List[Tuple[int, int, int, int, float]]:
        """
        Post-process YOLO detections to get person bounding boxes
        
        Args:
            outputs: Raw model outputs
            scale_x, scale_y: Scale factors from preprocessing
            original_width, original_height: Original image dimensions
            
        Returns:
            List of person detections as (x, y, w, h, confidence)
        """
        # Get the main output (detections)
        detections = outputs[0][0]  # Remove batch dimension
        
        boxes = []
        confidences = []
        
        # Process each detection
        for detection in detections.T:  # Transpose to get (num_detections, 85)
            # Extract box coordinates and scores
            x_center, y_center, width, height = detection[:4]
            scores = detection[4:]  # Class scores
            
            # Get class with highest score
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Only keep person detections (class_id == 0) above threshold
            if class_id == 0 and confidence >= self.conf_threshold:
                # Convert from center format to corner format
                x1 = int((x_center - width / 2) / scale_x)
                y1 = int((y_center - height / 2) / scale_y)
                w = int(width / scale_x)
                h = int(height / scale_y)
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))
                w = min(w, original_width - x1)
                h = min(h, original_height - y1)
                
                boxes.append([x1, y1, w, h])
                confidences.append(float(confidence))
        
        # Apply Non-Maximum Suppression
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            
            final_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    confidence = confidences[i]
                    final_detections.append((x, y, w, h, confidence))
            
            return final_detections
        
        return []

    def detect_persons(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of person detections as (x, y, w, h, confidence)
        """
        original_height, original_width = image.shape[:2]
        
        # Preprocess image
        input_tensor, scale_x, scale_y = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Post-process detections
        detections = self.postprocess_detections(outputs, scale_x, scale_y, original_width, original_height)
        
        return detections

    def process_images(self, input_folder: str, output_folder: str):
        """
        Process all images in the input folder and save person detections
        
        Args:
            input_folder: Path to folder containing input images
            output_folder: Path to folder where person detections will be saved
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output directory: {output_folder}")
        
        total_persons_detected = 0
        images_with_persons = 0
        
        for i, image_file in enumerate(sorted(image_files)):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Failed to load {image_file}")
                continue
            
            # Detect persons
            detections = self.detect_persons(image)
            
            if detections:
                images_with_persons += 1
                
                # Save each detected person
                for j, (x, y, w, h, confidence) in enumerate(detections):
                    # Crop person region
                    person_crop = image[y:y+h, x:x+w]
                    
                    # Generate output filename
                    base_name = image_file.stem
                    output_filename = f"{base_name}_person_{j+1}_conf{confidence:.2f}.jpg"
                    output_filepath = output_path / output_filename
                    
                    # Save cropped person image
                    cv2.imwrite(str(output_filepath), person_crop)
                    total_persons_detected += 1
                    
                    print(f"  Saved person {j+1} (conf: {confidence:.2f}) -> {output_filename}")
            else:
                print(f"  No persons detected")
        
        print(f"\nProcessing complete!")
        print(f"Images processed: {len(image_files)}")
        print(f"Images with persons: {images_with_persons}")
        print(f"Total persons detected: {total_persons_detected}")
        print(f"Person images saved to: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Detect persons in images using YOLO11n ONNX model")
    parser.add_argument("--model", default="yolo11n.onnx", help="Path to YOLO11n ONNX model")
    parser.add_argument("--input", default="splitted", help="Input folder containing images")
    parser.add_argument("--output", default="detection", help="Output folder for person detections")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        return
    
    # Check if input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' not found")
        return
    
    try:
        # Initialize detector
        detector = PersonDetector(args.model, args.conf, args.nms)
        
        # Process images
        detector.process_images(args.input, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
