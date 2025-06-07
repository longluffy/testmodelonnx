#!/usr/bin/env python3
"""
Simple PyTorch Detection Analysis Script
Runs detection analysis on all images in the 'detection' folder using best.pt
and generates a comprehensive markdown report.
"""

from detection_analyzer_pt import DetectionAnalyzerPT
import os

def main():
    # Configuration
    model_path = "best.pt"
    input_folder = "detection"
    output_report = "detection_analysis_report_pt.md"
    confidence_threshold = 0.3
    nms_threshold = 0.4
    
    print("=" * 60)
    print("PyTorch YOLO Detection Analysis")
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
        print("Please make sure the PyTorch YOLO model is in the current directory.")
        return
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        print("Please make sure the 'detection' folder exists with images to process.")
        return
    
    try:
        # Initialize analyzer
        print("🔄 Loading PyTorch YOLO model...")
        analyzer = DetectionAnalyzerPT(
            model_path=model_path,
            conf_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        print("✅ Model loaded successfully!")
        print("\n🔄 Starting detection analysis...")
        
        # Process images
        results = analyzer.analyze_folder(input_folder)
        
        if 'error' in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("✅ Analysis completed!")
        print("\n🔄 Generating markdown report...")
        
        # Generate report
        analyzer.generate_markdown_report(results, output_report)
        
        print("\n✅ PyTorch detection analysis completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Images processed: {results['summary']['total_images']}")
        print(f"   - Class 0 detections: {results['summary']['total_class_0']}")
        print(f"   - Class 1 detections: {results['summary']['total_class_1']}")
        print(f"   - Other classes: {results['summary']['total_other']}")
        print(f"   - Average process time: {results['summary']['average_process_time']:.4f}s")
        print(f"📄 Report saved to: {output_report}")
        print(f"🌐 You can convert to HTML with: python3 convert_md_to_html.py")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
