#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
from typing import List
from pathlib import Path

import cv2
import numpy as np

from onnx_predictor import YoloxOnnxPredictor


DEFAULT_CLASSES = (
    "character",
)


COLOR_PALETTE = np.array(
    [
        1.000, 0.500, 0.000,
    ]
).astype(np.float32).reshape(-1, 3)


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def parse_args():
    parser = argparse.ArgumentParser("YOLOX ONNX batch image detector")
    parser.add_argument(
        "input_dir",
        help="Path to the directory containing input images"
    )
    parser.add_argument(
        "--model",
        default="character.onnx",
        help="Path to the exported ONNX model (default: character.onnx)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where cropped character images are saved",
    )
    parser.add_argument(
        "--input-shape",
        default="640,640",
        help="Model input size as height,width",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold for filtering detections",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=0.45,
        help="IoU threshold used by NMS",
    )
    parser.add_argument(
        "--class-names",
        default=None,
        help="Optional file with custom class names, one per line",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional list of ONNX Runtime execution providers",
    )
    return parser.parse_args()


def load_class_names(path: str | None) -> List[str]:
    if not path:
        return list(DEFAULT_CLASSES)
    with open(path, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle.readlines()]
    names = [name for name in names if name]
    return names if names else list(DEFAULT_CLASSES)


def get_image_files(directory: str) -> List[str]:
    """Get list of image files from directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    path = Path(directory)
    
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")
    
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            image_files.append(str(file))
    
    return sorted(image_files)


def main():
    args = parse_args()
    class_names = load_class_names(args.class_names)
    input_shape = tuple(map(int, args.input_shape.split(",")))
    
    # Initialize predictor
    predictor = YoloxOnnxPredictor(
        model_path=args.model,
        input_shape=input_shape,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        class_names=class_names,
        providers=args.providers,
    )
    
    # Get image files
    image_files = get_image_files(args.input_dir)
    
    if not image_files:
        print(f"No image files found in: {args.input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    # Create output directory
    mkdir(args.output_dir)
    
    total_detections = 0
    
    # Process each image
    for image_path in image_files:
        print(f"\nProcessing: {image_path}")
        
        origin_img = cv2.imread(image_path)
        if origin_img is None:
            print(f"  Warning: Unable to read image, skipping")
            continue
        
        # Predict
        boxes, scores, cls_ids = predictor.predict(origin_img)
        
        if len(boxes) == 0:
            print(f"  No detections found")
            continue
        
        print(f"  Found {len(boxes)} detection(s)")
        
        # Crop and save detected regions
        base_name = Path(image_path).stem
        for idx, (box, score, cls_id) in enumerate(zip(boxes, scores, cls_ids)):
            x0, y0, x1, y1 = box.astype(int)
            
            # Clamp coordinates to image bounds
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(origin_img.shape[1], x1)
            y1 = min(origin_img.shape[0], y1)
            
            # Extract the region
            cropped = origin_img[y0:y1, x0:x1]
            
            if cropped.size == 0:
                print(f"  Warning: Empty crop at detection {idx+1}, skipping")
                continue
            
            # Create filename
            cls_idx = int(cls_id)
            label = class_names[cls_idx] if cls_idx < len(class_names) else f"cls_{cls_idx}"
            confidence = float(score)
            filename = f"{base_name}_{label}_{idx+1:03d}_{confidence:.2f}.jpg"
            output_path = os.path.join(args.output_dir, filename)
            
            # Save the cropped image
            cv2.imwrite(output_path, cropped)
            print(f"  Saved: {filename} (confidence: {confidence:.2f})")
            
            total_detections += 1
    
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Total detections: {total_detections}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
