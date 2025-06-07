#!/usr/bin/env python3
"""
Simple Detection Analysis Runner
Analyzes all images in the 'detection' folder using best.onnx model
and generates a comprehensive markdown report.
"""

from detection_analyzer import DetectionAnalyzer
import os

def main():
    # Configuration
    model_path = "best.onnx"
    input_folder = "detection"
    output_report = "detection_analysis_report.md"
    
    print("🔍 Detection Analysis with Custom Model")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return
    
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' not found!")
        return
    
    try:
        # Initialize and run analyzer
        analyzer = DetectionAnalyzer(model_path)
        results = analyzer.analyze_folder(input_folder)
        
        if 'error' not in results:
            analyzer.generate_markdown_report(results, output_report)
            print(f"✅ Analysis complete! Report: {output_report}")
        else:
            print(f"❌ Error: {results['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
