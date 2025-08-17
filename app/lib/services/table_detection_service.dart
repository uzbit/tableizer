import 'dart:async';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

// --- FFI Structs ---
// Mirrors the FFI_Point struct in C++.
final class FFI_Point extends Struct {
  @Float()
  external double x;

  @Float()
  external double y;
}

// Mirrors the DetectionResult struct in C++.
final class DetectionResult extends Struct {
  @Array(4)
  external Array<FFI_Point> quad_points;

  @Int32()
  external int quad_points_count;

  external Pointer<Uint8> image_bytes;

  @Int32()
  external int image_width;

  @Int32()
  external int image_height;
}

// --- FFI Function Signatures ---
typedef DetectTableRawC = Pointer<DetectionResult> Function(
    Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 stride);
typedef DetectTableRawDart = Pointer<DetectionResult> Function(
    Pointer<Uint8> imageBytes, int width, int height, int stride);

typedef FreeDetectionResultC = Void Function(Pointer<DetectionResult> result);
typedef FreeDetectionResultDart = void Function(Pointer<DetectionResult> result);

// --- Dart Data Class ---
// A pure Dart class to hold the detection data safely.
class TableDetection {
  final List<Offset> quadPoints;
  final Uint8List? debugImage;
  final int imageWidth;
  final int imageHeight;

  TableDetection({
    required this.quadPoints,
    this.debugImage,
    required this.imageWidth,
    required this.imageHeight,
  });
}

class TableDetectionService {
  late final DetectTableRawDart _detectTableRaw;
  late final FreeDetectionResultDart _freeDetectionResult;

  final StreamController<TableDetection> _detectionsController =
      StreamController<TableDetection>.broadcast();

  bool _isDetecting = false;
  bool _isInitialized = false;

  Stream<TableDetection> get detections => _detectionsController.stream;

  Future<void> initialize() async {
    if (_isInitialized) return;
    await _loadLibrary();
    _isInitialized = true;
  }

  void dispose() {
    _detectionsController.close();
  }

  void processImage(img.Image image) {
    if (!_isInitialized || _isDetecting) return;
    _isDetecting = true;

    final rgbaImage = image.convert(numChannels: 4);
    final bytes = rgbaImage.getBytes(order: img.ChannelOrder.rgba);

    final Pointer<Uint8> imagePtr = calloc<Uint8>(bytes.length);
    imagePtr.asTypedList(bytes.length).setAll(0, bytes);

    Pointer<DetectionResult> resultPtr = nullptr;
    try {
      resultPtr = _detectTableRaw(
        imagePtr,
        rgbaImage.width,
        rgbaImage.height,
        rgbaImage.width * 4,
      );

      if (resultPtr != nullptr) {
        // Access the struct's fields.
        final result = resultPtr.ref;

        // Copy the data into safe Dart objects.
        final quadPoints = <Offset>[];
        for (var i = 0; i < result.quad_points_count; i++) {
          quadPoints.add(Offset(
            result.quad_points[i].x,
            result.quad_points[i].y,
          ));
        }

        final imageSize = result.image_width * result.image_height * 4;
        final imageBytes = Uint8List.fromList(
          result.image_bytes.asTypedList(imageSize),
        );

        // Create the final Dart object.
        final detection = TableDetection(
          quadPoints: quadPoints,
          debugImage: imageBytes,
          imageWidth: result.image_width,
          imageHeight: result.image_height,
        );

        // Add to the stream for the UI to consume.
        _detectionsController.add(detection);
      }
    } finally {
      // CRITICAL: Free the native memory.
      if (resultPtr != nullptr) {
        _freeDetectionResult(resultPtr);
      }
      calloc.free(imagePtr);
      _isDetecting = false;
    }
  }

  Future<void> _loadLibrary() async {
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open('libtableizer_lib.so')
        : DynamicLibrary.process();

    _detectTableRaw = dylib
        .lookup<NativeFunction<DetectTableRawC>>('detect_table_raw')
        .asFunction();
    _freeDetectionResult = dylib
        .lookup<NativeFunction<FreeDetectionResultC>>('free_detection_result')
        .asFunction();
  }
}

