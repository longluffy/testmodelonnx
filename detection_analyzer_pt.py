#!/usr/bin/env python3
"""
Detection Analysis with PyTorch YOLO Model (.pt)
This script analyzes the cropped person images in the 'detection' folder using the best.pt model
and generates a comprehensive markdown report with detection statistics and details.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Try to import torch and ultralytics
try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  Warning: PyTorch or Ultralytics not available. Please install them:")
    print("pip install torch ultralytics")

class DetectionAnalyzerPT:
    def __init__(self, model_path: str, conf_threshold: float = 0.6, nms_threshold: float = 0.4):
        """
        Initialize the Detection Analyzer with PyTorch YOLO model
        
        Args:
            model_path: Path to the PyTorch YOLO model (.pt)
            conf_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and Ultralytics are required. Install with: pip install torch ultralytics")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # Load PyTorch YOLO model
        self.model = YOLO(model_path)
        
        # Binary classification class names for best.pt model
        self.class_names = {
            0: "CA",  # Class 0 
            1: "PN",  # Class 1
        }
        
        print(f"Loaded PyTorch model from {model_path}")
        print(f"Model type: {type(self.model)}")
        
        # Get model info
        try:
            model_info = self.model.info()
            print(f"Model info: {model_info}")
        except:
            print("Could not retrieve detailed model info")

    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID"""
        return self.class_names.get(class_id, f"class_{class_id}")

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image and return detection results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image analysis results
        """
        start_time = time.time()
        
        try:
            # Run inference using YOLO model
            results = self.model(image_path, conf=self.conf_threshold, iou=self.nms_threshold, verbose=False)
            
            # Extract detections from results
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract box coordinates (xyxy format)
                        box_coords = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box_coords
                        
                        # Convert to x,y,w,h format
                        x = float(x1)
                        y = float(y1)
                        w = float(x2 - x1)
                        h = float(y2 - y1)
                        
                        # Extract confidence and class
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # For binary classification, you may need to adjust class interpretation
                        # Assuming the model outputs 0 and 1 directly
                        if class_id > 1:
                            # If model has more classes, map to binary
                            class_id = 1 if class_id > 0 else 0
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': self.get_class_name(class_id),
                            'confidence': confidence,
                            'bbox': [x, y, w, h],
                            'raw_class_id': int(boxes.cls[i].cpu().numpy())  # Original class ID
                        }
                        
                        detections.append(detection)
            
            process_time = time.time() - start_time
            
            return {
                'image_name': Path(image_path).name,
                'process_time': process_time,
                'detections': detections,
                'image_size': self._get_image_size(image_path)
            }
            
        except Exception as e:
            process_time = time.time() - start_time
            return {
                'image_name': Path(image_path).name,
                'error': f'Analysis failed: {str(e)}',
                'process_time': process_time,
                'detections': []
            }

    def _get_image_size(self, image_path: str) -> Tuple[int, int]:
        """Get image dimensions"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                h, w = image.shape[:2]
                return (w, h)
        except:
            pass
        return (0, 0)

    def analyze_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Analyze all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            Dictionary with complete analysis results
        """
        folder_path = Path(folder_path)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in folder_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No image files found in {folder_path}")
            return {'error': 'No images found'}
        
        print(f"Found {len(image_files)} images to analyze")
        
        results = []
        class_counts = {}
        total_process_time = 0
        
        for i, image_file in enumerate(sorted(image_files)):
            print(f"Analyzing {i+1}/{len(image_files)}: {image_file.name}")
            
            result = self.analyze_image(str(image_file))
            results.append(result)
            
            total_process_time += result['process_time']
            
            # Count classes
            for detection in result['detections']:
                class_id = detection['class_id']
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += 1
        
        # Calculate statistics
        total_images = len(image_files)
        avg_process_time = total_process_time / total_images if total_images > 0 else 0
        
        class_0_count = class_counts.get(0, 0)
        class_1_count = class_counts.get(1, 0)
        other_count = sum(count for class_id, count in class_counts.items() if class_id not in [0, 1])
        
        return {
            'summary': {
                'total_images': total_images,
                'total_class_0': class_0_count,
                'total_class_1': class_1_count,
                'total_other': other_count,
                'average_process_time': avg_process_time,
                'total_process_time': total_process_time,
                'class_counts': class_counts
            },
            'details': results
        }

    def generate_markdown_report(self, analysis_results: Dict[str, Any], output_path: str):
        """
        Generate a markdown report from analysis results
        
        Args:
            analysis_results: Results from analyze_folder
            output_path: Path to save the markdown report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md_content = f"""# Detection Analysis Report (PyTorch Model)

**Generated on:** {timestamp}  
**Model:** {self.model_path}  
**Model Type:** PyTorch YOLO (.pt)  
**Confidence Threshold:** {self.conf_threshold}  
**NMS Threshold:** {self.nms_threshold}  

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Images Processed | {analysis_results['summary']['total_images']} |
| Total Class 0 Detections | {analysis_results['summary']['total_class_0']} |
| Total Class 1 Detections | {analysis_results['summary']['total_class_1']} |
| Total Other/Unknown Classes | {analysis_results['summary']['total_other']} |
| Average Process Time | {analysis_results['summary']['average_process_time']:.4f}s |
| Total Process Time | {analysis_results['summary']['total_process_time']:.2f}s |

### Class Distribution

| Class ID | Class Name | Count |
|----------|------------|-------|
"""

        # Add class distribution
        for class_id, count in sorted(analysis_results['summary']['class_counts'].items()):
            class_name = self.get_class_name(class_id)
            md_content += f"| {class_id} | {class_name} | {count} |\n"

        md_content += f"""

## Detailed Results

| Image Name | Image Preview | Class ID | Class Name | Confidence | Raw Class ID | Bounding Box (x,y,w,h) | Process Time (s) |
|------------|---------------|----------|------------|------------|--------------|-------------------------|------------------|
"""

        # Add detailed results
        for result in analysis_results['details']:
            image_name = result['image_name']
            process_time = result['process_time']
            
            if result.get('error'):
                md_content += f"| {image_name} | - | ERROR | {result['error']} | - | - | - | {process_time:.4f} |\n"
            elif not result['detections']:
                # Include image preview for "No detections" entries
                image_path = f"detection/{image_name}"
                md_content += f"| {image_name} | ![{image_name}]({image_path}) | - | No detections | - | - | - | {process_time:.4f} |\n"
            else:
                for i, detection in enumerate(result['detections']):
                    class_id = detection['class_id']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    raw_class_id = detection.get('raw_class_id', 'N/A')
                    bbox = detection['bbox']
                    bbox_str = f"({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})"
                    
                    # Only show image name on first detection
                    img_name = image_name if i == 0 else ""
                    proc_time = f"{process_time:.4f}" if i == 0 else ""
                    
                    # No image preview for detections, just empty cell
                    md_content += f"| {img_name} | - | {class_id} | {class_name} | {confidence:.3f} | {raw_class_id} | {bbox_str} | {proc_time} |\n"

        md_content += f"""

## Analysis Notes

- Model used: `{self.model_path}` (PyTorch YOLO)
- Model type: .pt (PyTorch model)
- Total detections found: {sum(analysis_results['summary']['class_counts'].values())}
- Images with detections: {sum(1 for result in analysis_results['details'] if result['detections'])}
- Images without detections: {sum(1 for result in analysis_results['details'] if not result['detections'] and not result.get('error'))}

---
*Report generated by Detection Analyzer (PyTorch)*
"""

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdown report saved to: {output_path}")


def main():
    # Configuration
    model_path = "best.pt"
    input_folder = "detection"
    output_report = "detection_analysis_report_pt.md"
    confidence_threshold = 0.3
    nms_threshold = 0.4
    
    print("=" * 60)
    print("Detection Analysis with PyTorch YOLO Model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output report: {output_report}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    print("=" * 60)
    
    # Check if PyTorch is available
    if not TORCH_AVAILABLE:
        print("❌ Error: PyTorch and Ultralytics are required!")
        print("Install with: pip install torch ultralytics")
        return
    
    # Check if required files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        return
    
    try:
        # Initialize analyzer
        print("🔄 Loading PyTorch model...")
        analyzer = DetectionAnalyzerPT(
            model_path=model_path,
            conf_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        print("✅ Model loaded successfully!")
        print("\n🔄 Starting analysis...")
        
        # Analyze images
        results = analyzer.analyze_folder(input_folder)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("✅ Analysis completed!")
        print("\n🔄 Generating markdown report...")
        
        # Generate report
        analyzer.generate_markdown_report(results, output_report)
        
        print("\n✅ Detection analysis completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Images processed: {results['summary']['total_images']}")
        print(f"   - Class 0 detections: {results['summary']['total_class_0']}")
        print(f"   - Class 1 detections: {results['summary']['total_class_1']}")
        print(f"   - Other classes: {results['summary']['total_other']}")
        print(f"   - Average process time: {results['summary']['average_process_time']:.4f}s")
        print(f"📄 Report saved to: {output_report}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
