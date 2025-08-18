import 'dart:async';
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'detection_isolate.dart';

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
}

// --- FFI Function Signatures ---
typedef DetectTableBgraC = Pointer<DetectionResult> Function(Pointer<Uint8> imageBytes, Int32 width,
    Int32 height, Int32 stride, Pointer<Utf8> debugImagePath);
typedef DetectTableBgraDart = Pointer<DetectionResult> Function(Pointer<Uint8> imageBytes,
    int width, int height, int stride, Pointer<Utf8> debugImagePath);

typedef FreeBgraDetectionResultC = Void Function(Pointer<DetectionResult> result);
typedef FreeBgraDetectionResultDart = void Function(Pointer<DetectionResult> result);

class TableDetectionService {
  late final DetectTableBgraDart detectTableBgra;
  late final FreeBgraDetectionResultDart freeBgraDetectionResult;

  final StreamController<List<Offset>> _detectionsController =
      StreamController<List<Offset>>.broadcast();
  Stream<List<Offset>> get detections => _detectionsController.stream;

  final StreamController<Uint8List> _debugImageController =
      StreamController<Uint8List>.broadcast();
  Stream<Uint8List> get debugImages => _debugImageController.stream;

  Isolate? _isolate;
  SendPort? _sendPort;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;
    _isInitialized = true;

    _loadLibrary();

    final receivePort = ReceivePort();
    print('[MAIN] Spawning isolate...');
    final rootToken = RootIsolateToken.instance!;
    _isolate = await Isolate.spawn(
        detectionIsolateEntry, [receivePort.sendPort, rootToken]);

    receivePort.listen((message) {
      // print('[MAIN] Received message from isolate: ${message.runtimeType}');
      if (message is SendPort) {
        _sendPort = message;
        print('[MAIN] Received SendPort from isolate.');
      } else if (message is List<Offset>) {
        _detectionsController.add(message);
      } else if (message is Map && message['type'] == 'debug_image') {
        _debugImageController.add(message['data']);
      }
    });
  }

  void _loadLibrary() {
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open('libtableizer_lib.so')
        : DynamicLibrary.process();

    detectTableBgra = dylib
        .lookup<NativeFunction<DetectTableBgraC>>('detect_table_bgra')
        .asFunction();
    freeBgraDetectionResult = dylib
        .lookup<NativeFunction<FreeBgraDetectionResultC>>('free_bgra_detection_result')
        .asFunction();
  }

  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _detectionsController.close();
    _debugImageController.close();
  }

  Future<void> processImage(AnalysisImage image) async {
    if (_sendPort == null) return;
    _sendPort!.send(image);
  }
}

