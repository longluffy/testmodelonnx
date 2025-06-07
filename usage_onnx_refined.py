import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def sigmoid(x):
    """Apply sigmoid activation"""
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow

def nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Get indices sorted by score (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the highest scoring box
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = current_area + remaining_areas - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with low IoU
        keep_mask = iou < iou_threshold
        indices = indices[1:][keep_mask]
    
    return keep

def postprocess_yolo_output(output, conf_threshold=0.5, nms_threshold=0.5):
    """Post-process ONNX output with built-in NMS (nms=True export)"""
    if len(output.shape) == 3:  # [batch, detections, features]
        batch_size, num_detections, num_features = output.shape
        logging.info(f"Processing NMS-enabled YOLO output: {output.shape}")
        
        # Remove batch dimension -> [num_detections, 6]
        detections = output[0]
        
        # Extract features: [x1, y1, x2, y2, confidence, class_id]
        # Note: With NMS=True, output format changes to corner coordinates
        x1 = detections[:, 0]
        y1 = detections[:, 1] 
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        confidence = detections[:, 4]  # Already processed confidence
        class_prob = detections[:, 5]   # Class probability/ID
        
        # Filter by confidence threshold
        valid_mask = confidence > conf_threshold
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Extract valid detections
        results = []
        for idx in valid_indices:
            # For binary classification, interpret class_prob
            if class_prob[idx] < 0.5:
                class_id = 0  # CA
            else:
                class_id = 1  # PN
                
            results.append({
                "bbox": [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                "confidence": float(confidence[idx]),
                "class": int(class_id),
                "raw_class_prob": float(class_prob[idx])
            })
        
        return results
    
    return []

def run_onnx_inference_refined(onnx_path, image_path):
    """Run inference with refined post-processing"""
    # Load ONNX model
    providers = ['CPUExecutionProvider']  # Use CPU for now
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Get model info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((input_shape[3], input_shape[2]), Image.LANCZOS)
    
    img_array = np.asarray(image_resized).astype(np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
    
    # Run inference
    start_time = time.time()
    outputs = session.run(None, {input_name: img_array})
    inference_time = time.time() - start_time
    
    # Post-process with proper NMS



    detections = postprocess_yolo_output(outputs[0], conf_threshold=0.3, nms_threshold=0.4)
    
    return {
        "image_path": str(image_path),
        "inference_time_seconds": round(inference_time, 4),
        "detections": detections,
        "num_detections": len(detections)
    }

def main():
    current_folder = Path.cwd()
    onnx_path = current_folder / "best.onnx"
    
    # Test with one image first
    test_image = current_folder / "testimg2.jpg"
    
    logging.info(f"Testing refined ONNX inference on {test_image.name}...")
    result = run_onnx_inference_refined(onnx_path, test_image)
    
    logging.info(f"Found {result['num_detections']} detections in {result['inference_time_seconds']}s")
    
    for i, det in enumerate(result['detections']):
        logging.info(f"  Detection {i+1}: class={det['class']}, conf={det['confidence']:.3f}, "
                    f"bbox=[{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]")
    
    # Save results
    output_file = current_folder / "onnx_refined_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": str(onnx_path),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "test_result": result
        }, f, indent=2)
    
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
