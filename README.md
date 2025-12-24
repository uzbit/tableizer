# Tableizer

A mobile application combining computer vision and machine learning to detect pool tables, locate ball positions, and map them onto ShotStudio-style overlays.

## Overview

Tableizer consists of three main components:

1. **Native C++ Vision Engine** (`lib/`) - Core detection and image processing with ONNX Runtime
2. **Flutter Mobile App** (`app/`) - iOS and Android real-time capture interface
3. **Python Tooling** (`python/`) - Model training, dataset transforms, and debugging utilities

## Project Structure

```
tableizer/
├── app/                    # Flutter mobile application
│   ├── lib/               # Dart source code
│   │   ├── controllers/   # Business logic controllers
│   │   ├── models/        # Data structures
│   │   ├── screens/       # UI screens
│   │   ├── services/      # FFI services and detection logic
│   │   └── widgets/       # Reusable UI components
│   ├── ios/               # iOS native configuration
│   ├── android/           # Android native configuration
│   └── assets/            # App assets (models, images)
├── lib/                    # C++ native library
│   ├── src/               # Source files
│   ├── include/           # Header files
│   ├── libs/              # Dependencies (OpenCV, ONNX Runtime)
│   ├── build_ios.sh       # iOS build script
│   └── build_android.sh   # Android build script
├── python/                 # Python tooling
├── tableizer/              # Trained YOLO models
└── data/                   # Datasets and training images
```

## Prerequisites

### General Requirements
- **Flutter SDK** >= 3.8.1
- **CMake** >= 3.10
- **Ninja** build system (for OpenCV builds)

### macOS Development
- **Xcode** >= 15.0 (with command line tools)
- **CocoaPods** (`sudo gem install cocoapods`)
- **Homebrew** with `opencv` and `onnxruntime` installed

### Android Development
- **Android Studio** with NDK installed
- **Android NDK** r29+ (set `ANDROID_NDK_HOME` environment variable)
- **Android SDK** with platform 24+

### Python Tooling
- **Python** 3.10+
- Virtual environment recommended

## Building the Flutter App

### 1. Build the Native Libraries

The Flutter app requires pre-built native libraries for table and ball detection.

#### For iOS

```bash
cd lib

# Build OpenCV and Tableizer for iOS (device + simulator)
./build_ios.sh
```

This script:
- Builds OpenCV static libraries for iOS device (arm64) and simulator
- Builds the `libtableizer_lib.dylib` shared library
- Copies libraries to `app/ios/`

**Output files:**
- `app/ios/libtableizer_lib.dylib` (device, default)
- `app/ios/libtableizer_lib_device.dylib`
- `app/ios/libtableizer_lib_sim.dylib`

#### For Android

```bash
# Set the NDK path (adjust version as needed)
export ANDROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/29.0.13599879

cd lib

# Build OpenCV and Tableizer for Android
./build_android.sh
```

This script:
- Builds OpenCV shared libraries for Android (arm64-v8a)
- Builds `libtableizer_lib.so`
- Copies all `.so` files to `app/android/app/src/main/jniLibs/arm64-v8a/`

**Required libraries in jniLibs:**
- `libtableizer_lib.so`
- `libopencv_*.so` (core, imgproc, dnn, etc.)
- `libonnxruntime.so`

### 2. Install Flutter Dependencies

```bash
cd app

# Get Flutter packages
flutter pub get
```

### 3. iOS Setup

```bash
cd app/ios

# Install CocoaPods dependencies (includes ONNX Runtime)
pod install
```

**Important:** The iOS build uses ONNX Runtime from CocoaPods (`pod 'onnxruntime-c'`).

### 4. Running the App

#### iOS Device
```bash
cd app
flutter run -d <ios-device-id>
```

Or build for release:
```bash
flutter build ios --release
```

#### iOS Simulator
For simulator, configure the app to use the simulator library:
1. The library loader in `app/lib/native/library_loader.dart` handles device vs simulator detection
2. Ensure `libtableizer_lib_sim.dylib` is available in `app/ios/`

```bash
flutter run -d <ios-simulator-id>
```

#### Android Device
```bash
cd app
flutter run -d <android-device-id>
```

Or build APK:
```bash
flutter build apk --debug
# or
flutter build apk --release
```

## Native Library Development

### Building for macOS (Local Testing)

```bash
cd lib
mkdir build && cd build
cmake ..
make
```

This creates:
- `tableizer_app` - CLI executable for testing
- `libtableizer_lib.a` - Static library

### Running Tests

```bash
cd lib/build
ctest
```

### FFI Entry Points

The native library exposes these functions for Flutter FFI:

| Function | Description |
|----------|-------------|
| `initialize_detector(modelPath)` | Load YOLO ONNX model |
| `detect_table_bgra(...)` | Detect table quadrilateral in image |
| `detect_balls_bgra(...)` | Detect balls within table region |
| `transform_points_using_quad(...)` | Transform coordinates to table space |
| `normalize_image_bgra(...)` | Apply perspective correction |

## Python Tooling

### Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Scripts

| Script | Purpose |
|--------|---------|
| `python/detect_table.py` | Table detection using C++ FFI or YOLO |
| `python/detect_transformed_table.py` | Detection on perspective-corrected images |
| `python/tableizer_ffi.py` | Python FFI bindings |
| `python/transform_dataset.py` | Dataset augmentation pipeline |

### Training Models

YOLO models are trained using Ultralytics:

```bash
cd python
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='path/to/data.yaml', epochs=100)"
```

Trained models are stored in `tableizer/combined4/` (current production model).

## App Architecture

### Data Flow

```
Camera Frame (CameraAwesome)
    ↓
TableDetectionController.processImage()
    ↓
TableDetectionService → Isolate
    ↓
[C++ FFI] detect_table_bgra()
    ├── Table Detection (color analysis, contours)
    ├── Quad Analysis (orientation detection)
    └── Image Normalization (perspective transform)
    ↓
BallDetectionController.detectBalls()
    ↓
[C++ FFI] detect_balls_bgra() + YOLO inference
    ↓
TableResultsScreen (visualization)
```

### Key Dependencies

**Flutter:**
- `camerawesome` - Live camera feed
- `permission_handler` - Camera permissions
- `wakelock_plus` - Keep screen on

**C++:**
- OpenCV - Image processing
- ONNX Runtime - YOLO inference

## Usage Notes

- Hold phone in **landscape orientation** (16:9 aspect ratio) for best detection
- Ensure adequate lighting on the pool table
- Position camera to capture entire table surface

## Known Issues

- Some modules contain hard-coded absolute paths (see `todo.md`)
- Frame rate depends on image resolution
- 45-degree camera angles may affect orientation detection

## License

Private project - not for redistribution.
