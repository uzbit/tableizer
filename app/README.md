# Tableizer Flutter App

Real-time pool table detection and ball tracking mobile application for iOS and Android.

## Quick Start

### Prerequisites

- Flutter SDK >= 3.8.1
- For iOS: Xcode >= 15.0, CocoaPods
- For Android: Android Studio with NDK, `ANDROID_NDK_HOME` set

### Build Native Libraries First

The app requires native C++ libraries. Build them before running:

```bash
# iOS
cd ../lib && ./build_ios.sh

# Android
export ANDROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/29.0.13599879
cd ../lib && ./build_android.sh
```

### Install Dependencies

```bash
flutter pub get

# iOS only
cd ios && pod install && cd ..
```

### Run

```bash
# List available devices
flutter devices

# Run on specific device
flutter run -d <device-id>

# Build release
flutter build ios --release
flutter build apk --release
```

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── controllers/              # Business logic
│   ├── table_detection_controller.dart
│   ├── ball_detection_controller.dart
│   └── camera_controller.dart
├── models/                   # Data classes
│   ├── table_detection_result.dart
│   └── ball_detection_result.dart
├── screens/                  # UI screens
│   ├── camera_screen.dart    # Main camera view
│   ├── table_results_screen.dart
│   └── settings_screen.dart
├── services/                 # Core services
│   ├── table_detection_service.dart
│   ├── table_detection_isolate.dart
│   └── ball_detection_service.dart
├── widgets/                  # UI components
│   ├── camera_preview_widget.dart
│   ├── table_painter.dart
│   └── table_ball_painter.dart
└── native/
    └── library_loader.dart   # FFI library loading
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `camerawesome` | Camera capture with live preview |
| `permission_handler` | Runtime permissions |
| `wakelock_plus` | Keep screen on during capture |
| `image` | Image processing utilities |
| `vector_math` | Vector/matrix operations |
| `ffi` | Native library bindings |

## Assets

Required assets in `assets/`:
- `detection_model.onnx` - YOLO ball detection model
- `images/shotstudio_table_felt_only.png` - Table overlay background

## Platform Configuration

### iOS (`ios/`)

- Minimum deployment target: iOS 18.5
- Native library: `libtableizer_lib.dylib`
- ONNX Runtime via CocoaPods: `pod 'onnxruntime-c'`

**Info.plist permissions:**
- `NSCameraUsageDescription` - Camera access for table detection

### Android (`android/`)

- Minimum SDK: 24 (Android 7.0)
- Target SDK: 34
- Native libraries in `app/src/main/jniLibs/arm64-v8a/`:
  - `libtableizer_lib.so`
  - `libopencv_*.so`
  - `libonnxruntime.so`

**AndroidManifest.xml permissions:**
- `android.permission.CAMERA`

## Development

### Hot Reload

```bash
flutter run  # Then press 'r' in terminal
```

### Debug Logging

Detection results and FFI calls are logged to console during debug builds.

### Testing

```bash
flutter test
flutter test integration_test/
```

## Troubleshooting

### iOS: Library not found
- Ensure `libtableizer_lib.dylib` exists in `ios/`
- Run `pod install` after adding/updating pods
- Clean build: `flutter clean && flutter pub get`

### Android: UnsatisfiedLinkError
- Verify all `.so` files exist in `jniLibs/arm64-v8a/`
- Check that `ANDROID_NDK_HOME` was set correctly during build
- Rebuild native libraries: `cd ../lib && ./build_android.sh`

### Camera permission denied
- Check device settings for app permissions
- On Android 13+, may need to request permissions manually

### Slow detection
- Reduce camera resolution in settings
- Detection runs in isolate to avoid blocking UI
- Frame rate depends on image size and device performance
