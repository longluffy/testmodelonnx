#!/usr/bin/env python3
"""
Simple Person Detection Script
Runs person detection on all images in the 'splitted' folder using yolo11n.onnx
and saves detected person images to the 'detection' folder.
"""

from person_detector import PersonDetector
import os

def main():
    # Configuration
    model_path = "yolo11n.onnx"
    input_folder = "splitted"
    output_folder = "detection"
    confidence_threshold = 0.5
    nms_threshold = 0.4
    
    print("=" * 60)
    print("Person Detection with YOLO11n")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    print("=" * 60)
    
    # Check if required files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please make sure the YOLO11n ONNX model is in the current directory.")
        return
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        print("Please make sure the 'splitted' folder exists with images to process.")
        return
    
    try:
        # Initialize detector
        print("🔄 Loading YOLO11n model...")
        detector = PersonDetector(
            model_path=model_path,
            conf_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        print("✅ Model loaded successfully!")
        print("\n🔄 Starting person detection...")
        
        # Process images
        detector.process_images(input_folder, output_folder)
        
        print("\n✅ Person detection completed successfully!")
        print(f"Check the '{output_folder}' folder for detected person images.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
