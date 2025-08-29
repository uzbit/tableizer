# Tableizer Detection

A Flutter package for pool table and ball detection using computer vision.

## Features

- Real-time table detection with quad overlay
- Ball detection and classification  
- Alpha-filtered smooth overlays
- Cross-platform support (Android, iOS, Desktop)
- Optimized performance with isolates and caching

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  tableizer_detection:
    git:
      url: https://github.com/yourusername/tableizer_detection.git
```

## Binary Dependencies

This package requires native libraries to be built separately. The package expects these files to be available:

### Android
- `libtableizer_lib.so` in `android/app/src/main/jniLibs/arm64-v8a/`

### iOS  
- `libtableizer_lib.dylib` in the iOS app bundle

### Building Native Libraries

1. Clone the native library project:
```bash
git clone https://github.com/yourusername/tableizer-native.git
```

2. Build for your target platform:
```bash
# Android
cd tableizer-native && ./build-android.sh

# iOS Simulator  
cd tableizer-native && ./build-ios-sim.sh

# iOS Device
cd tableizer-native && ./build-ios-device.sh
```

3. Copy the resulting binaries to your Flutter project

## Usage

```dart
import 'package:tableizer_detection/tableizer_detection.dart';

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late TableDetectionController _tableController;
  late BallDetectionController _ballController;
  
  @override
  void initState() {
    super.initState();
    
    // Configure detection parameters
    final config = DetectionConfig(
      quadAlpha: 0.3,  // Smoothing factor for quad overlay
      confidenceThreshold: 0.6,
      cellSize: 10,
    );
    
    _tableController = TableDetectionController();
    _ballController = BallDetectionController();
  }
  
  @override
  void dispose() {
    _tableController.dispose();
    _ballController.dispose();
    super.dispose();
  }
}
```

## API Reference

See the [API documentation](https://pub.dev/documentation/tableizer_detection/latest/) for detailed usage.