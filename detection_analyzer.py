#!/usr/bin/env python3
"""
Detection Analysis with Custom ONNX Model
This script analyzes the cropped person images in the 'detection' folder using the best.onnx model
and generates a comprehensive markdown report with detection statistics and details.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any
from datetime import datetime

class DetectionAnalyzer:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        Initialize the Detection Analyzer with custom ONNX model
        
        Args:
            model_path: Path to the custom ONNX model
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
        
        # Binary classification class names for best.onnx model
        self.class_names = {
            0: "CA",  # Class 0 
            1: "PN",  # Class 1
        }
        
        print(f"Loaded model from {model_path}")
        print(f"Input shape: {input_shape}")
        print(f"Input size: {self.input_width}x{self.input_height}")

    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID"""
        return self.class_names.get(class_id, f"class_{class_id}")

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Preprocess image for model inference
        
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
                             original_width: int, original_height: int) -> List[Dict[str, Any]]:
        """
        Post-process ONNX output with built-in NMS (nms=True export)
        Based on usage_onnx_refined.py approach
        
        Args:
            outputs: Raw model outputs
            scale_x, scale_y: Scale factors from preprocessing
            original_width, original_height: Original image dimensions
            
        Returns:
            List of detections with class_id, confidence, and bbox
        """
        output = outputs[0]  # Get the main output tensor
        
        if len(output.shape) == 3:  # [batch, detections, features]
            batch_size, num_detections, num_features = output.shape
            print(f"Processing NMS-enabled YOLO output: {output.shape}")
            
            # Remove batch dimension -> [num_detections, 6]
            detections = output[0]
            
            # Extract features: [x1, y1, x2, y2, confidence, class_prob]
            # Note: With NMS=True, output format uses corner coordinates
            x1 = detections[:, 0] / scale_x
            y1 = detections[:, 1] / scale_y
            x2 = detections[:, 2] / scale_x
            y2 = detections[:, 3] / scale_y
            confidence = detections[:, 4]  # Already processed confidence
            class_prob = detections[:, 5]   # Class probability/ID
            
            # Filter by confidence threshold
            valid_mask = confidence > self.conf_threshold
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return []
            
            # Extract valid detections
            final_detections = []
            for idx in valid_indices:
                # For binary classification, interpret class_prob
                if class_prob[idx] < 0.5:
                    class_id = 0  # Class 0 (CA)
                else:
                    class_id = 1  # Class 1 (PN)
                
                # Convert corner coordinates to x,y,w,h format
                x = float(x1[idx])
                y = float(y1[idx])
                w = float(x2[idx] - x1[idx])
                h = float(y2[idx] - y1[idx])
                
                # Clamp to image boundaries
                x = max(0, min(x, original_width))
                y = max(0, min(y, original_height))
                w = min(w, original_width - x)
                h = min(h, original_height - y)
                
                final_detections.append({
                    'class_id': class_id,
                    'class_name': self.get_class_name(class_id),
                    'confidence': float(confidence[idx]),
                    'bbox': [x, y, w, h],
                    'raw_class_prob': float(class_prob[idx])
                })
            
            return final_detections
        
        return []

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a single image and return detection results
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image analysis results
        """
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'image_name': Path(image_path).name,
                'error': 'Failed to load image',
                'process_time': 0,
                'detections': []
            }
        
        original_height, original_width = image.shape[:2]
        
        # Preprocess image
        input_tensor, scale_x, scale_y = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Post-process detections
        detections = self.postprocess_detections(outputs, scale_x, scale_y, original_width, original_height)
        
        process_time = time.time() - start_time
        
        return {
            'image_name': Path(image_path).name,
            'process_time': process_time,
            'detections': detections,
            'image_size': (original_width, original_height)
        }

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
        
        md_content = f"""# Detection Analysis Report

**Generated on:** {timestamp}  
**Model:** {self.model_path}  
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

| Image Name | Image Preview | Class ID | Class Name | Confidence | Raw Class Prob | Bounding Box (x,y,w,h) | Process Time (s) |
|------------|---------------|----------|------------|------------|----------------|-------------------------|------------------|
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
                    raw_class_prob = detection.get('raw_class_prob', 'N/A')
                    bbox = detection['bbox']
                    bbox_str = f"({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})"
                    
                    # Only show image name on first detection
                    img_name = image_name if i == 0 else ""
                    proc_time = f"{process_time:.4f}" if i == 0 else ""
                    
                    if isinstance(raw_class_prob, float):
                        raw_prob_str = f"{raw_class_prob:.3f}"
                    else:
                        raw_prob_str = str(raw_class_prob)
                    
                    # No image preview for detections, just empty cell
                    md_content += f"| {img_name} | - | {class_id} | {class_name} | {confidence:.3f} | {raw_prob_str} | {bbox_str} | {proc_time} |\n"

        md_content += f"""

## Analysis Notes

- Model used: `{self.model_path}`
- Input resolution: {self.input_width}x{self.input_height}
- Total detections found: {sum(analysis_results['summary']['class_counts'].values())}
- Images with detections: {sum(1 for result in analysis_results['details'] if result['detections'])}
- Images without detections: {sum(1 for result in analysis_results['details'] if not result['detections'] and not result.get('error'))}

---
*Report generated by Detection Analyzer*
"""

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdown report saved to: {output_path}")


def main():
    # Configuration
    model_path = "best.onnx"
    input_folder = "detection"
    output_report = "detection_analysis_report.md"
    confidence_threshold = 0.5
    nms_threshold = 0.4
    
    print("=" * 60)
    print("Detection Analysis with Custom ONNX Model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input folder: {input_folder}")
    print(f"Output report: {output_report}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    print("=" * 60)
    
    # Check if required files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        return
    
    try:
        # Initialize analyzer
        print("🔄 Loading model...")
        analyzer = DetectionAnalyzer(
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
