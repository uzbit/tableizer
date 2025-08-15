import 'dart:async';
import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'detection_isolate.dart';

// --- FFI Signatures ---
typedef DetectTableRGBAC = Pointer<Utf8> Function(
    Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectTableRGBADart = Pointer<Utf8> Function(
    Pointer<Uint8> imageBytes, int width, int height, int channels);

class TableDetectionService {
  late final DetectTableRGBADart _detectTableRGBA;
  final Completer<void> _isolateReady = Completer<void>();
  final StreamController<Map<String, dynamic>> _detectionsController =
      StreamController<Map<String, dynamic>>.broadcast();

  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;
  bool _isDetecting = false;
  bool _isInitialized = false;

  Stream<Map<String, dynamic>> get detections => _detectionsController.stream;

  // ────────── PUBLIC API ──────────
  Future<void> initialize() async {
    if (_isInitialized) return;
    await _loadLibrary();
    await _initIsolate();
    _isInitialized = true;
  }

  void dispose() {
    _receivePort?.close();
    _isolate?.kill(priority: Isolate.immediate);
    _detectionsController.close();
  }

  Future<Map<String, dynamic>> detectTableFromByteBuffer(
      Uint8List bytes, int width, int height) async {
    if (!_isInitialized) return const {};
    await _isolateReady.future;

    if (_isDetecting) return const {};
    _isDetecting = true;

    _sendPort?.send({
      'bytes': bytes,
      'width': width,
      'height': height,
    });
    return const {};
  }

  // ────────── PRIVATE HELPERS ──────────
  Future<void> _initIsolate() async {
    _receivePort = ReceivePort();
    _isolate = await Isolate.spawn(
      DetectionIsolate.init,
      _receivePort!.sendPort,
    );

    _receivePort!.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        _isolateReady.complete();
      } else {
        _detectionsController.add(message as Map<String, dynamic>);
        _isDetecting = false;
      }
    });
  }

  Future<void> _loadLibrary() async {
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open('libtableizer_lib.so')
        : DynamicLibrary.process();

    _detectTableRGBA = dylib
        .lookup<NativeFunction<DetectTableRGBAC>>('detect_table_rgba')
        .asFunction();
  }
}

