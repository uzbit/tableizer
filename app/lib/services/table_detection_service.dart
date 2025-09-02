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

typedef DetectTableRgbaC = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, Int32 width,
    Int32 height, Int32 channels, Int32 stride);
typedef DetectTableRgbaDart = Pointer<Utf8> Function(Pointer<Uint8> imageBytes,
    int width, int height, int channels, int stride);

typedef FreeBgraDetectionResultC = Void Function(Pointer<DetectionResult> result);
typedef FreeBgraDetectionResultDart = void Function(Pointer<DetectionResult> result);

class TableDetectionService {
  late final DetectTableBgraDart detectTableBgra;
  late final DetectTableRgbaDart detectTableRgba;
  late final FreeBgraDetectionResultDart freeBgraDetectionResult;

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
    detectTableRgba = dylib
        .lookup<NativeFunction<DetectTableRgbaC>>('detect_table_rgba')
        .asFunction();
    freeBgraDetectionResult = dylib
        .lookup<NativeFunction<FreeBgraDetectionResultC>>('free_bgra_detection_result')
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

      // Call native function (4 channels for RGBA, stride = width * 4)
      final Pointer<Utf8> resultPtr = detectTableRgba(nativeBytes, width, height, 4, width * 4);
      
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
        return Offset(point['x'].toDouble(), point['y'].toDouble());
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
}