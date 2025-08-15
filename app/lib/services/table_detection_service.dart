import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:convert';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

// --- FFI Signatures ---
typedef DetectTableRGBAC = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectTableRGBADart = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, int width, int height, int channels);

class TableDetectionService {
  late final DetectTableRGBADart _detectTableRGBA;

  bool _isDetecting = false;
  bool _isInitialized = false;

  // ────────── PUBLIC API ──────────
  Future<void> initialize() async {
    if (_isInitialized) return;
    await _loadLibrary();
    _isInitialized = true;
  }

  Future<Map<String, dynamic>> detectTableFromRGBImage(img.Image src) async {
    if (_isDetecting || !_isInitialized) return const {};
    _isDetecting = true;

    Pointer<Uint8>? pixelPtr;
    Pointer<Utf8>?  jsonPtr;

    try {
      final img.Image rgba =
      src.numChannels == 4 ? src : src.convert(numChannels: 4);

      final bytes = rgba.getBytes(order: img.ChannelOrder.rgba);
      pixelPtr = calloc<Uint8>(bytes.length)
        ..asTypedList(bytes.length).setAll(0, bytes);

      jsonPtr = _detectTableRGBA(
          pixelPtr, rgba.width, rgba.height, 4);

      final result = jsonDecode(jsonPtr.toDartString());
      print('JSON RGBA Table Detection: ${result["quad_points"]}');
      return result;

    } catch (e, st) {
      print('detectFromImage error: $e');
      print(st);
      return const {};
    } finally {
      if (pixelPtr != null) calloc.free(pixelPtr);
      _isDetecting = false;
    }
  }

  // ────────── PRIVATE HELPERS ──────────
  Future<void> _loadLibrary() async {
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open('libtableizer_lib.so')
        : DynamicLibrary.process();

    _detectTableRGBA = dylib
        .lookup<NativeFunction<DetectTableRGBAC>>('detect_table_rgba')
        .asFunction();
  }
}

