import argparse
import json
import os
import math
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from onnx_predictor import YoloxOnnxPredictor

# --- CONFIGURATION ---
TARGET_SIZE = (640, 640)
GREY_COLOR = (128, 128, 128)
BATCH_SIZE = 64 
MODEL_PATH = "character.onnx"  
CLASS_NAMES = ["character"]
# ---------------------

def parse_args():
    parser = argparse.ArgumentParser("YOLOX ONNX batch image detector — COCO output")
    parser.add_argument("input_dir", help="Path to the directory containing input images")
    parser.add_argument(
        "--output-json",
        default="annotations.json",
        help="Filename for the COCO JSON file",
    )
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    parser.add_argument("--nms-thr", type=float, default=0.45, help="IoU threshold")
    parser.add_argument(
        "--providers",
        nargs="*",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution providers",
    )
    return parser.parse_args()

def get_image_files(directory: str) -> list[str]:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"Directory not found: {directory}")
    return sorted(
        str(f.resolve()) for f in path.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    )

def letterbox_pad(img, target_size, color):

    shape = img.shape[:2]  
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = target_size[1] - new_unpad[0], target_size[0] - new_unpad[1]
    
    dw, dh = dw / 2, dh / 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_CUBIC)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def main():
    args = parse_args()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found in the current directory.")
        return

    results_dir = Path("Results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = results_dir / args.output_json

    predictor = YoloxOnnxPredictor(
        model_path=MODEL_PATH,
        input_shape=TARGET_SIZE,
        score_thr=args.score_thr,
        nms_thr=args.nms_thr,
        class_names=CLASS_NAMES,
        providers=args.providers,
    )

    image_files = get_image_files(args.input_dir)
    if not image_files:
        print(f"No image files found in: {args.input_dir}")
        return

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "character"}],
    }

    annotation_id = 1
    total_detections = 0
    image_id = 1

    for batch_start in range(0, len(image_files), BATCH_SIZE):
        batch_paths = image_files[batch_start : batch_start + BATCH_SIZE]
        
        for image_path in batch_paths:
            origin_img = cv2.imread(image_path)
            if origin_img is None:
                continue

            processed_img = letterbox_pad(origin_img, TARGET_SIZE, GREY_COLOR)
            
            filename = Path(image_path).name
            output_image_path = results_dir / filename
            cv2.imwrite(str(output_image_path), processed_img)

            coco_format["images"].append({
                "id": image_id,
                "file_name": filename,
                "width": TARGET_SIZE[1],
                "height": TARGET_SIZE[0],
            })

            boxes, scores, cls_ids = predictor.predict(processed_img)

            for box, score, cls_id in zip(boxes, scores, cls_ids):
                x0, y0, x1, y1 = box.astype(float)
                w, h = x1 - x0, y1 - y0

                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": [x0, y0, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": round(float(score), 4),
                })
                annotation_id += 1
                total_detections += 1
            
            image_id += 1

        print(f"Processed batch {batch_start // BATCH_SIZE + 1}/{(len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_format, f, indent=2)

    print(f"\nExecution Complete.")
    print(f"Processed {len(image_files)} images.")
    print(f"Total detections: {total_detections}")
    print(f"Output Directory: {results_dir.resolve()}")

if __name__ == "__main__":
    main()