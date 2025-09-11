import 'dart:async';
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';

import '../native/library_loader.dart';
import '../models/table_detection_result.dart';
import 'table_detection_isolate.dart';

// --- FFI Structs ---
final class FFI_Point extends Struct {
  @Float()
  external double x;
  @Float()
  external double y;
}

final class DetectionResult extends Struct {
  @Array(4)
  external Array<FFI_Point> quad_points;
  @Int32()
  external int quad_points_count;
  @Int32()
  external int image_width;
  @Int32()
  external int image_height;
}

// --- FFI Function Signatures ---
typedef DetectTableBgraC = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, Int32 width,
    Int32 height, Int32 stride, Int32 rotationDegrees, Pointer<Utf8> debugImagePath);
typedef DetectTableBgraDart = Pointer<Utf8> Function(Pointer<Uint8> imageBytes,
    int width, int height, int stride, int rotationDegrees, Pointer<Utf8> debugImagePath);


typedef FreeBgraDetectionResultC = Void Function(Pointer<DetectionResult> result);
typedef FreeBgraDetectionResultDart = void Function(Pointer<DetectionResult> result);

typedef TransformPointsUsingQuadC = Pointer<Utf8> Function(
    Pointer<Float> pointsData, Int32 pointsCount,
    Pointer<Float> quadData, Int32 quadCount,
    Int32 imageWidth, Int32 imageHeight,
    Int32 displayWidth, Int32 displayHeight);
typedef TransformPointsUsingQuadDart = Pointer<Utf8> Function(
    Pointer<Float> pointsData, int pointsCount,
    Pointer<Float> quadData, int quadCount,
    int imageWidth, int imageHeight,
    int displayWidth, int displayHeight);

class TableDetectionService {
  late final DetectTableBgraDart detectTableBgra;
  late final FreeBgraDetectionResultDart freeBgraDetectionResult;
  late final TransformPointsUsingQuadDart transformPointsUsingQuad;

  final StreamController<TableDetectionResult> _detectionsController =
      StreamController<TableDetectionResult>.broadcast();
  Stream<TableDetectionResult> get detections => _detectionsController.stream;

  Isolate? _isolate;
  SendPort? _sendPort;
  bool _isReady = true; // Gate to control frame processing
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;
    _isInitialized = true;

    _loadLibrary();

    final receivePort = ReceivePort();
    print('[MAIN] Spawning isolate...');
    final rootToken = RootIsolateToken.instance!;
    _isolate = await Isolate.spawn(
        tableDetectionIsolateEntry, [receivePort.sendPort, rootToken]);

    receivePort.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        print('[MAIN] Received SendPort from isolate.');
      } else if (message is TableDetectionResult) {
        _detectionsController.add(message);
      } else if (message is bool && message == true) {
        // This is the "ready" signal from the isolate
        _isReady = true;
      }
    });
  }

  void _loadLibrary() {
    final dylib = LibraryLoader.library;

    detectTableBgra = dylib
        .lookup<NativeFunction<DetectTableBgraC>>('detect_table_bgra')
        .asFunction();
    freeBgraDetectionResult = dylib
        .lookup<NativeFunction<FreeBgraDetectionResultC>>('free_bgra_detection_result')
        .asFunction();
    transformPointsUsingQuad = dylib
        .lookup<NativeFunction<TransformPointsUsingQuadC>>('transform_points_using_quad')
        .asFunction();
  }

  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _detectionsController.close();
  }

  Future<void> processImage(AnalysisImage image) async {
    if (_sendPort == null || !_isReady) {
      // Drop frame if the isolate is busy or not ready
      return;
    }
    _isReady = false; // Close the gate
    _sendPort!.send(image);
  }

  Future<TableDetectionResult?> detectTableFromBytes(Uint8List imageBytes, int width, int height) async {
    if (!_isInitialized) {
      await initialize();
    }

    try {
      // Allocate native memory
      final Pointer<Uint8> nativeBytes = malloc<Uint8>(imageBytes.length);
      final Uint8List nativeBytesList = nativeBytes.asTypedList(imageBytes.length);
      nativeBytesList.setAll(0, imageBytes);

      // Call native function (BGRA format, stride = width * 4)
      final Pointer<Utf8> resultPtr = detectTableBgra(nativeBytes, width, height, width * 4, 0, nullptr);
      
      // Free native memory immediately
      malloc.free(nativeBytes);

      if (resultPtr == nullptr) {
        return null;
      }

      final String jsonResult = resultPtr.toDartString();
      
      // Parse JSON result
      final Map<String, dynamic> parsed = json.decode(jsonResult);
      if (parsed.containsKey('error')) {
        print('Table detection error: ${parsed['error']}');
        return null;
      }

      final List<dynamic> quadPointsJson = parsed['quad_points'] ?? [];
      final List<Offset> quadPoints = quadPointsJson.map<Offset>((point) {
        return Offset(point[0].toDouble(), point[1].toDouble());
      }).toList();

      return TableDetectionResult(
        quadPoints,
        Size(width.toDouble(), height.toDouble()),
      );
    } catch (e) {
      print('Error in detectTableFromBytes: $e');
      return null;
    }
  }

  /// Transform points using quad-to-rectangle perspective transformation via C++ FFI
  List<Offset>? transformPoints(
    List<Offset> points,
    List<Offset> quadPoints,
    Size imageSize,
    Size displaySize,
  ) {
    if (!_isInitialized || quadPoints.length != 4 || points.isEmpty) {
      return null;
    }

    try {
      // Allocate native memory for points
      final Pointer<Float> pointsPtr = malloc<Float>(points.length * 2);
      for (int i = 0; i < points.length; i++) {
        pointsPtr[i * 2] = points[i].dx;
        pointsPtr[i * 2 + 1] = points[i].dy;
      }

      // Allocate native memory for quad points
      final Pointer<Float> quadPtr = malloc<Float>(8); // 4 points * 2 coordinates
      for (int i = 0; i < 4; i++) {
        quadPtr[i * 2] = quadPoints[i].dx;
        quadPtr[i * 2 + 1] = quadPoints[i].dy;
      }

      // Call native function
      final Pointer<Utf8> resultPtr = transformPointsUsingQuad(
        pointsPtr, points.length,
        quadPtr, 4,
        imageSize.width.toInt(), imageSize.height.toInt(),
        displaySize.width.toInt(), displaySize.height.toInt(),
      );

      // Free native memory
      malloc.free(pointsPtr);
      malloc.free(quadPtr);

      if (resultPtr == nullptr) {
        return null;
      }

      final String jsonResult = resultPtr.toDartString();
      final Map<String, dynamic> parsed = json.decode(jsonResult);
      
      if (parsed.containsKey('error')) {
        print('Transform points error: ${parsed['error']}');
        return null;
      }

      final List<dynamic> transformedPointsJson = parsed['transformed_points'] ?? [];
      return transformedPointsJson.map<Offset>((point) {
        return Offset(point['x'].toDouble(), point['y'].toDouble());
      }).toList();

    } catch (e) {
      print('Error in transformPoints: $e');
      return null;
    }
  }
}