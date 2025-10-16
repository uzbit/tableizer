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

// --- FFI Function Signatures ---
typedef TransformPointsUsingQuadC = Pointer<Utf8> Function(
    Pointer<Float> pointsData, Int32 pointsCount,
    Pointer<Float> quadData, Int32 quadCount,
    Int32 imageWidth, Int32 imageHeight,
    Int32 displayWidth, Int32 displayHeight,
    Int32 inputRotationDegrees);
typedef TransformPointsUsingQuadDart = Pointer<Utf8> Function(
    Pointer<Float> pointsData, int pointsCount,
    Pointer<Float> quadData, int quadCount,
    int imageWidth, int imageHeight,
    int displayWidth, int displayHeight,
    int inputRotationDegrees);

class TableDetectionService {
  late final TransformPointsUsingQuadDart transformPointsUsingQuad;

  final StreamController<TableDetectionResult> _detectionsController =
      StreamController<TableDetectionResult>.broadcast();
  Stream<TableDetectionResult> get detections => _detectionsController.stream;

  Isolate? _isolate;
  SendPort? _sendPort;
  bool _isReady = true; // Gate to control frame processing
  bool _isInitialized = false;
  bool _isPaused = false; // Control whether to process frames

  Future<void> initialize() async {
    if (_isInitialized) return;
    _isInitialized = true;

    _loadLibrary();

    final receivePort = ReceivePort();
    final rootToken = RootIsolateToken.instance!;
    _isolate = await Isolate.spawn(
        tableDetectionIsolateEntry, [receivePort.sendPort, rootToken]);

    receivePort.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
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

    transformPointsUsingQuad = dylib
        .lookup<NativeFunction<TransformPointsUsingQuadC>>('transform_points_using_quad')
        .asFunction();
  }

  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _detectionsController.close();
  }

  Future<void> processImage(AnalysisImage image) async {
    if (_sendPort == null || !_isReady || _isPaused) {
      // Drop frame if paused, isolate is busy, or not ready
      return;
    }
    _isReady = false; // Close the gate
    _sendPort!.send(image);
  }

  void pause() {
    _isPaused = true;
    print('[TABLE_SERVICE] Table detection paused');
  }

  void resume() {
    _isPaused = false;
    print('[TABLE_SERVICE] Table detection resumed');
  }

  /// Transform points using quad-to-rectangle perspective transformation via C++ FFI
  List<Offset>? transformPoints(
    List<Offset> points,
    List<Offset> quadPoints,
    Size imageSize,
    Size displaySize,
    int inputRotationDegrees,
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

      // Call native function with rotation parameter
      final Pointer<Utf8> resultPtr = transformPointsUsingQuad(
        pointsPtr, points.length,
        quadPtr, 4,
        imageSize.width.toInt(), imageSize.height.toInt(),
        displaySize.width.toInt(), displaySize.height.toInt(),
        inputRotationDegrees,
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