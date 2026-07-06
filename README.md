# anime-character-detector
Adapted the orignal YOLOX-based anime character detector by [ksasao](https://github.com/ksasao) to output the result in COCO based format for datasets

## Links

- **Orignal Repo**: [ksasao.github/anime-character-detector](https://github.com/ksasao/anime-character-detector)
- **Live Link**: [ksasao.github.io/anime-character-detector](https://ksasao.github.io/anime-character-detector/)
- **Result Dataset**:[muneeburrehman98/danbooru-annotated-images](https://www.kaggle.com/datasets/muneeburrehman98/danbooru-annotated-images)

## Model Information

| Property | Value |
|----------|-------|
| Base Model | YOLOX-s |
| Input Size | 640×640 |
| Classes | Single class: "character" |
| mAP@50 | 82.4% |
| mAP@50-95 | 57.7% |
| Training Images | 3,253 |
| Annotations | 15,901 |

## Features

- **Client-side Processing**: All inference runs in your browser using ONNX Runtime Web
- **Multiple Input Methods**: Drag & drop, file selection, or paste from clipboard
- **WebGPU/WASM Support**: Hardware acceleration when available
- **HEIC/HEIF Support**: Automatic conversion to PNG for iOS images
- **Adjustable Parameters**: Score threshold and NMS IoU threshold controls
