# Tableizer Python Tooling

Python utilities for training YOLO ball detection models and working with pool table datasets.

## Setup

```bash
# Create virtual environment (from project root)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training New YOLO Models

### Overview

The training pipeline:
1. Prepare dataset with images and YOLO-format labels
2. Remap class labels to standard format (0=black, 1=cue, 2=solid, 3=stripe)
3. Split into train/val/test sets
4. Train with Ultralytics YOLO
5. Export to ONNX for mobile deployment

### Quick Start

```bash
cd python

# Train a new model with default settings
python model_table.py

# Train with custom config
python model_table.py --config path/to/config.json
```

### Step-by-Step Guide

#### 1. Prepare Your Dataset

Create a dataset directory with the following structure:

```
data/my_dataset/
├── images/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels/
    ├── image001.txt
    ├── image002.txt
    └── ...
```

**Label format (YOLO):** Each `.txt` file contains one line per object:
```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized (0-1). Example:
```
0 0.5123 0.3456 0.0234 0.0312
2 0.7891 0.6543 0.0198 0.0287
```

**Class IDs:**
| ID | Class |
|----|-------|
| 0 | Black (8-ball) |
| 1 | Cue (white ball) |
| 2 | Solid (1-7) |
| 3 | Stripe (9-15) |

#### 2. Transform Dataset (Optional)

If your images show full table views at angles, transform them to normalized top-down perspective:

```bash
python transform_dataset.py data/my_dataset -o data/my_dataset_transformed

# Options:
#   --visualize          Show each transformation
#   --require-valid      Skip images with bad orientation detection
```

This:
- Detects the table quadrilateral using C++ FFI
- Applies perspective correction to 864x1680 (ShotStudio dimensions)
- Transforms ball bounding boxes accordingly

#### 3. Configure Training

Edit `model_table.py` or create a JSON config file:

```python
CONFIG = {
    "srcImgDir": "data/my_dataset/images",
    "srcLblDir": "data/my_dataset/labels",
    "dstRoot": "/tmp/workdir",
    "oldToNewMap": {0: 0, 1: 1, 2: 2, 3: 3},  # Class remapping
    "classNames": ["black", "cue", "solid", "stripe"],
    "split": [0.8, 0.10, 0.10],  # train/val/test ratios
    "trainer": {
        "model": "yolov8n.pt",      # Base model
        "hyp": "data/hyps/hyp.custom.yaml",  # Hyperparameters
        "epochs": 40,
        "imgsz": 1280,              # Input image size
        "batch": 4,                 # Batch size (reduce if OOM)
        "device": "mps",            # "mps", "cuda:0", or "cpu"
        "workers": 8,
        "project": "tableizer",
        "name": "my_model",         # Output directory name
    },
}
```

#### 4. Train the Model

```bash
python model_table.py
```

Training outputs are saved to `tableizer/<name>/`:
```
tableizer/my_model/
├── weights/
│   ├── best.pt          # Best model weights
│   └── last.pt          # Final epoch weights
├── args.yaml            # Training configuration
├── results.csv          # Training metrics
└── *.png                # Training curves and confusion matrix
```

#### 5. Export to ONNX

For mobile deployment, export to ONNX format:

```bash
# From command line
yolo export model=tableizer/my_model/weights/best.pt \
    format=onnx \
    device=cpu \
    imgsz=1280 \
    simplify=True \
    dynamic=False \
    opset=17 \
    half=False
```

Copy the model to the Flutter app:
```bash
cp tableizer/my_model/weights/best.onnx ../app/assets/detection_model.onnx
```

### Hyperparameter Tuning

Custom hyperparameters are defined in `data/hyps/hyp.custom.yaml`:

```yaml
# Learning rate
lr0: 0.01           # Initial learning rate
lrf: 0.01           # Final learning rate (lr0 * lrf)

# Augmentation
hsv_h: 0.015        # Hue augmentation
hsv_s: 0.7          # Saturation augmentation
hsv_v: 0.4          # Value augmentation
degrees: 5.0        # Rotation (+/- degrees)
translate: 0.1      # Translation (+/- fraction)
scale: 0.25         # Scale (+/- gain)
flipud: 0.1         # Vertical flip probability
fliplr: 0.5         # Horizontal flip probability
mosaic: 0.45        # Mosaic augmentation probability

# Loss weights
box: 0.1            # Box loss gain
cls: 0.5            # Classification loss gain
dfl: 1.5            # Distribution focal loss gain

# Detection
conf: 0.4           # Confidence threshold
iou: 0.8            # IoU threshold
max_det: 20         # Maximum detections per image
```

### Model Versions

| Model | Description | Dataset |
|-------|-------------|---------|
| `baseline` | Initial pix2pockets only | pix2pockets |
| `combined` | + original ShotStudio | pix2pockets + shotstudio |
| `combined2` | + new images with ball sprites | Mixed |
| `combined3` | + rotated balls, varied backgrounds | Mixed |
| `combined4` | Current production model | All combined |

## Other Scripts

### detect_table.py

Test table and ball detection on images:

```bash
python detect_table.py path/to/image.jpg

# Options:
#   --model PATH         Path to YOLO model
#   --visualize          Show detection overlay
#   --rotation DEGREES   Rotate input image
```

### detect_transformed_table.py

Run detection on perspective-corrected images:

```bash
python detect_transformed_table.py path/to/image.jpg
```

### check_labels.py

Validate YOLO label files:

```bash
python check_labels.py data/my_dataset/labels/
```

### compare_models.py

Compare PyTorch and ONNX model outputs:

```bash
python compare_models.py
```

### tableizer_ffi.py

Python bindings for the C++ native library:

```python
from tableizer_ffi import detect_table_cpp, initialize_ball_detector

# Detect table quadrilateral
result = detect_table_cpp(image_bgr, rotation_degrees=0)
quad_points = result["quad_points"]
orientation = result["orientation"]

# Initialize ball detector with ONNX model
detector = initialize_ball_detector("path/to/model.onnx")
```

## Tips

### GPU Memory Issues

If training crashes with OOM errors:
- Reduce `batch` size (try 2 or 1)
- Reduce `imgsz` (try 640)
- Reduce `workers`
- Use `cache: disk` instead of `cache: True`

### Improving Detection

- **More data**: Combine multiple datasets
- **Data augmentation**: Adjust hyperparameters in `hyp.custom.yaml`
- **Larger model**: Try `yolov8s.pt` or `yolov8m.pt` instead of `yolov8n.pt`
- **More epochs**: Increase training epochs (monitor for overfitting)
- **Lower confidence**: Reduce `conf` threshold for more detections

### Dataset Quality

- Ensure consistent labeling across all images
- Include variety: different table colors, lighting, angles
- Balance classes: similar counts of each ball type
- Use `transform_dataset.py` to normalize perspective if images are at angles
