import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

// --- FFI Signatures ---
typedef DetectTableYUVC = Pointer<Utf8> Function(
    Pointer<Uint8> yPlane,
    Pointer<Uint8> uPlane,
    Pointer<Uint8> vPlane,
    Int32 width,
    Int32 height,
    Int32 yStride,
    Int32 uStride,
    Int32 vStride);
typedef DetectTableYUVDart = Pointer<Utf8> Function(
    Pointer<Uint8> yPlane,
    Pointer<Uint8> uPlane,
    Pointer<Uint8> vPlane,
    int width,
    int height,
    int yStride,
    int uStride,
    int vStride);

typedef DetectTableRGBAC = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectTableRGBADart = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, int width, int height, int channels);

class TableDetectionService {
  late final DetectTableYUVDart _detectTableYUV;
  late final DetectTableRGBADart _detectTableRGBA;

  bool _isDetecting = false;
  bool _isInitialized = false;

  // ────────── PUBLIC API ──────────
  Future<void> initialize() async {
    if (_isInitialized) return;
    await _loadLibrary();
    _isInitialized = true;
  }

  Future<Map<String, dynamic>> detectTableFromYUV(CameraImage img) async {
    if (_isDetecting || !_isInitialized) return const {};
    _isDetecting = true;

    Pointer<Uint8>? yPtr;
    Pointer<Uint8>? uPtr;
    Pointer<Uint8>? vPtr;

    try {
      final y = img.planes[0];
      final u = img.planes[1];
      final v = img.planes[2];

      yPtr = calloc<Uint8>(y.bytes.length)
        ..asTypedList(y.bytes.length).setAll(0, y.bytes);
      uPtr = calloc<Uint8>(u.bytes.length)
        ..asTypedList(u.bytes.length).setAll(0, u.bytes);
      vPtr = calloc<Uint8>(v.bytes.length)
        ..asTypedList(v.bytes.length).setAll(0, v.bytes);

      final jsonPtr = _detectTableYUV(
        yPtr,
        uPtr,
        vPtr,
        img.width,
        img.height,
        y.bytesPerRow,
        u.bytesPerRow,
        v.bytesPerRow,
      );

      final json = jsonPtr.toDartString();
      print('JSON YUV Table Detection: $json');

      return json.isEmpty ? const {} : jsonDecode(json);
    } finally {
      if (yPtr != null) calloc.free(yPtr);
      if (uPtr != null) calloc.free(uPtr);
      if (vPtr != null) calloc.free(vPtr);
      _isDetecting = false;
    }
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

      final jsonStr = jsonPtr.toDartString();
      print('JSON RGBA Table Detection: $jsonStr');

      if (jsonStr.isEmpty) return const {};
      return jsonDecode(jsonStr);
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

    _detectTableYUV = dylib
        .lookup<NativeFunction<DetectTableYUVC>>('detect_table_yuv')
        .asFunction();
    _detectTableRGBA = dylib
        .lookup<NativeFunction<DetectTableRGBAC>>('detect_table_rgba')
        .asFunction();
  }
}
