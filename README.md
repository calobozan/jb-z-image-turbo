# jb-z-image-turbo

Fast image generation using [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), a 6B parameter diffusion model with 8-step inference.

## Features

- Sub-second inference on GPU
- Photorealistic quality
- Bilingual text rendering (English & Chinese)
- Robust instruction adherence

## Requirements

- CUDA GPU with 16GB+ VRAM (recommended)
- ~15GB disk space for model weights

## Installation

```bash
jb-serve install ~/projects/jb-z-image-turbo
```

## Usage

```bash
# Start the service
curl -X POST http://192.168.0.107:9800/v1/tools/z-image-turbo/start

# Generate an image
curl -X POST http://192.168.0.107:9800/v1/tools/z-image-turbo/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain lake at sunset with snow-capped peaks",
    "width": 1024,
    "height": 1024
  }'

# Response includes a file reference
# {
#   "image": {"ref": "abc123", "url": "/v1/files/abc123.png", ...},
#   "width": 1024,
#   "height": 1024,
#   "prompt": "..."
# }

# Fetch the image
curl http://192.168.0.107:9800/v1/files/abc123.png -o image.png
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prompt | string | required | Text description of the image |
| width | int | 1024 | Image width in pixels |
| height | int | 1024 | Image height in pixels |
| steps | int | 9 | Inference steps (9 gives 8 DiT forwards) |
| seed | int | random | Random seed for reproducibility |
| format | string | png | Output format (png, jpg, webp) |

## Model Info

- **Model**: Tongyi-MAI/Z-Image-Turbo
- **Parameters**: 6B
- **Architecture**: Scalable Single-Stream DiT (S3-DiT)
- **Inference**: 8 function evaluations
