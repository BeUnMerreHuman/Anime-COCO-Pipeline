# anime-character-detector
YOLOX-based anime character detector running in browser with ONNX Runtime Web (WebGPU/WASM)

![Demo Screenshot](result.jpg)

## Links

- **Web Demo**: [https://ksasao.github.io/anime-character-detector/](https://ksasao.github.io/anime-character-detector/)
- **Model Download**: [docs/character.onnx](docs/character.onnx)
- **Python Code**: [python/](python/)

## Model Information

| Property | Value |
|----------|-------|
| Base Model | YOLOX-s |
| Input Size | 640×640 |
| Classes | Single class: "character" |
| mAP@50 | 79.4% |
| mAP@50-95 | 55.6% |
| Training Images | 2,656 |
| Annotations | 13,761 |

## Features

- **Client-side Processing**: All inference runs in your browser using ONNX Runtime Web
- **Multiple Input Methods**: Drag & drop, file selection, or paste from clipboard
- **WebGPU/WASM Support**: Hardware acceleration when available
- **HEIC/HEIF Support**: Automatic conversion to PNG for iOS images
- **Adjustable Parameters**: Score threshold and NMS IoU threshold controls

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.