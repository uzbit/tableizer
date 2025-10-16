import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:ui';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import '../models/ball_detection_result.dart';
import '../native/library_loader.dart';

// --- FFI Signatures ---
typedef InitializeDetectorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitializeDetectorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef DetectBallsBGRAC = Pointer<Utf8> Function(
    Pointer<Void> detector,
    Pointer<Uint8> imageBytes,
    Int32 width,
    Int32 height,
    Int32 stride,
    Pointer<Float> quadPoints,
    Int32 quadPointsLength,
    Int32 channelFormat);
typedef DetectBallsBGRADart = Pointer<Utf8> Function(
    Pointer<Void> detector,
    Pointer<Uint8> imageBytes,
    int width,
    int height,
    int stride,
    Pointer<Float> quadPoints,
    int quadPointsLength,
    int channelFormat);

typedef ReleaseDetectorC = Void Function(Pointer<Void> detector);
typedef ReleaseDetectorDart = void Function(Pointer<Void> detector);

typedef NormalizeImageBgraC = Pointer<Utf8> Function(
    Pointer<Uint8> inputBytes,
    Int32 inputWidth,
    Int32 inputHeight,
    Int32 inputStride,
    Int32 rotationDegrees,
    Pointer<Uint8> outputBytes,
    Int32 outputBufferSize,
    Int32 channelFormat);
typedef NormalizeImageBgraDart = Pointer<Utf8> Function(
    Pointer<Uint8> inputBytes,
    int inputWidth,
    int inputHeight,
    int inputStride,
    int rotationDegrees,
    Pointer<Uint8> outputBytes,
    int outputBufferSize,
    int channelFormat);

class BallDetectionService {
  Pointer<Void> _detector = nullptr;
  late final InitializeDetectorDart _initializeDetector;
  late final DetectBallsBGRADart _detectBallsBGRA;
  late final ReleaseDetectorDart _releaseDetector;
  late final NormalizeImageBgraDart _normalizeImageBgra;

  bool _isDetecting = false;

  // Channel format: 0=BGRA (Android), 1=RGBA (iOS)
  int get _channelFormat => Platform.isIOS ? 1 : 0;

  // ────────── PUBLIC API ──────────
  Future<void> initialize() async {
    if (_detector != nullptr) return;                       // idempotent
    await _loadLibrary();

    final modelPath = await _copyAssetToFile('assets/detection_model.onnx');
    print('Ball detector model path: $modelPath');
    final pathPtr = modelPath.toNativeUtf8();
    _detector = _initializeDetector(pathPtr);
    calloc.free(pathPtr);

    if (_detector == nullptr) {
      print('ERROR: initialize_detector returned nullptr');
      throw StateError('Failed to create detector');
    }
    print('Ball detector initialized successfully');
  }

  Future<void> dispose() async {
    if (_detector != nullptr) {
      _releaseDetector(_detector);
      _detector = nullptr;
    }
  }

  Future<List<BallDetectionResult>> detectBallsFromBytes(
      Uint8List bytes, int width, int height, {List<Offset>? quadPoints, int rotationDegrees = 0}) async {
    // Skip if detector busy or not initialised.
    if (_isDetecting || _detector == nullptr) return const [];

    _isDetecting = true;

    Pointer<Uint8>? inputPtr;
    Pointer<Uint8>? outputPtr;
    Pointer<Float>? quadPtr;
    Pointer<Utf8>? jsonPtr;

    try {
      // Allocate buffers for normalization
      inputPtr = calloc<Uint8>(bytes.length);
      inputPtr.asTypedList(bytes.length).setAll(0, bytes);

      // Allocate output buffer (big enough for 16:9 canvas)
      const int maxOutputSize = 3840 * 2160 * 4;
      outputPtr = calloc<Uint8>(maxOutputSize);

      // Call C++ normalize function
      final normalizeResult = _normalizeImageBgra(
        inputPtr,
        width,
        height,
        width * 4, // stride
        rotationDegrees,
        outputPtr,
        maxOutputSize,
        _channelFormat,
      );

      if (normalizeResult == nullptr) {
        print('[BALL_SERVICE] Normalization failed: returned nullptr');
        return const [];
      }

      // Parse normalization result
      final normalizeJson = normalizeResult.toDartString();
      final Map<String, dynamic> normalizeData = json.decode(normalizeJson);

      if (normalizeData.containsKey('error')) {
        print('[BALL_SERVICE] Normalization error: ${normalizeData['error']}');
        return const [];
      }

      final int normalizedWidth = normalizeData['width'] ?? 0;
      final int normalizedHeight = normalizeData['height'] ?? 0;
      final int normalizedStride = normalizeData['stride'] ?? 0;

      print('[BALL_SERVICE] Image normalized: ${normalizedWidth}x${normalizedHeight}, rotation: $rotationDegrees');

      // ── Copy quad points into native buffer if provided ────────────────────
      if (quadPoints != null && quadPoints.length == 4) {
        quadPtr = calloc<Float>(8); // 4 points * 2 coordinates
        for (int i = 0; i < 4; i++) {
          quadPtr[i * 2] = quadPoints[i].dx;
          quadPtr[i * 2 + 1] = quadPoints[i].dy;
        }
        print('[BALL_SERVICE] Passing quad points to C++: $quadPoints');
      }

      // ── Native inference call ──────────────────────────────────────────────
      print('[BALL_SERVICE] Calling C++ detect_balls_bgra: ${normalizedWidth}x${normalizedHeight}, stride: ${normalizedStride}');
      jsonPtr = _detectBallsBGRA(
        _detector,
        outputPtr,
        normalizedWidth,
        normalizedHeight,
        normalizedStride,
        quadPtr ?? nullptr,
        quadPoints?.length ?? 0,
        1, // After normalization, output is always RGBA
      );

      final jsonStr = jsonPtr.toDartString();
      print('[BALL_SERVICE] C++ returned JSON: $jsonStr');

      if (jsonStr.isEmpty) return const [];
      return BallDetectionResults.fromJson(jsonStr).detections;
    } catch (e, st) {
      print('detectFromImage error: $e');
      print(st);
      return const [];
    } finally {
      if (inputPtr != null) calloc.free(inputPtr);
      if (outputPtr != null) calloc.free(outputPtr);
      if (quadPtr != null) calloc.free(quadPtr);
      _isDetecting = false;
    }
  }

  /// Detect balls from already-normalized image bytes (skips normalization)
  /// Use this when you already have normalized buffer from table detection
  Future<List<BallDetectionResult>> detectBallsFromNormalizedBytes(
      Uint8List normalizedBytes, int width, int height, int stride, {List<Offset>? quadPoints}) async {
    // Skip if detector busy or not initialised.
    if (_isDetecting || _detector == nullptr) return const [];

    _isDetecting = true;

    Pointer<Uint8>? pixelPtr;
    Pointer<Float>? quadPtr;
    Pointer<Utf8>? jsonPtr;

    try {
      print('[BALL_SERVICE] Using pre-normalized buffer: ${width}x${height}, stride: $stride');

      // ── Copy normalized pixels into native buffer ──────────────────────────
      pixelPtr = calloc<Uint8>(normalizedBytes.length);
      pixelPtr.asTypedList(normalizedBytes.length).setAll(0, normalizedBytes);

      // ── Copy quad points into native buffer if provided ────────────────────
      if (quadPoints != null && quadPoints.length == 4) {
        quadPtr = calloc<Float>(8); // 4 points * 2 coordinates
        for (int i = 0; i < 4; i++) {
          quadPtr[i * 2] = quadPoints[i].dx;
          quadPtr[i * 2 + 1] = quadPoints[i].dy;
        }
        print('[BALL_SERVICE] Passing quad points to C++: $quadPoints');
      }

      // ── Native inference call ──────────────────────────────────────────────
      print('[BALL_SERVICE] Calling C++ detect_balls_bgra (pre-normalized): ${width}x${height}, stride: $stride');
      jsonPtr = _detectBallsBGRA(
        _detector,
        pixelPtr,
        width,
        height,
        stride,
        quadPtr ?? nullptr,
        quadPoints?.length ?? 0,
        1, // Pre-normalized bytes are now RGBA
      );

      final jsonStr = jsonPtr.toDartString();
      print('[BALL_SERVICE] C++ returned JSON: $jsonStr');

      if (jsonStr.isEmpty) return const [];
      return BallDetectionResults.fromJson(jsonStr).detections;
    } catch (e, st) {
      print('[BALL_SERVICE] detectBallsFromNormalizedBytes error: $e');
      print(st);
      return const [];
    } finally {
      if (pixelPtr != null) calloc.free(pixelPtr);
      if (quadPtr != null) calloc.free(quadPtr);
      _isDetecting = false;
    }
  }

  // ────────── PRIVATE HELPERS ──────────
  Future<void> _loadLibrary() async {
    final dylib = LibraryLoader.library;

    _initializeDetector = dylib
        .lookup<NativeFunction<InitializeDetectorC>>('initialize_detector')
        .asFunction();
    _detectBallsBGRA = dylib
        .lookup<NativeFunction<DetectBallsBGRAC>>('detect_balls_bgra')
        .asFunction();
    _releaseDetector = dylib
        .lookup<NativeFunction<ReleaseDetectorC>>('release_detector')
        .asFunction();
    _normalizeImageBgra = dylib
        .lookup<NativeFunction<NormalizeImageBgraC>>('normalize_image_bgra')
        .asFunction();
  }

  Future<String> _copyAssetToFile(String asset) async {
    final appDir = await getApplicationSupportDirectory();
    final dst = p.join(appDir.path, asset);
    final file = File(dst);
    if (!await file.exists()) {
      await file.create(recursive: true);
      final data = await rootBundle.load(asset);
      await file.writeAsBytes(
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
    }
    return dst;
  }
}

