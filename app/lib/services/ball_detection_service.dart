import 'dart:ffi' hide Size;
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import '../models/ball_detection_result.dart';
import '../native/library_loader.dart';

// --- FFI Signatures ---
typedef InitializeDetectorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitializeDetectorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef DetectObjectsBGRAC = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 stride);
typedef DetectObjectsBGRADart = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, int width, int height, int stride);

typedef ReleaseDetectorC = Void Function(Pointer<Void> detector);
typedef ReleaseDetectorDart = void Function(Pointer<Void> detector);

class BallDetectionService {
  Pointer<Void> _detector = nullptr;
  late final InitializeDetectorDart _initializeDetector;
  late final DetectObjectsBGRADart _detectObjectsBGRA;
  late final ReleaseDetectorDart _releaseDetector;
  // late final void Function(Pointer<Utf8>) _freeCString;

  bool _isDetecting = false;

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
      Uint8List bytes, int width, int height) async {
    // Skip if detector busy or not initialised.
    if (_isDetecting || _detector == nullptr) return const [];

    _isDetecting = true;

    Pointer<Uint8>? pixelPtr;
    Pointer<Utf8>? jsonPtr;

    try {
      // ── Copy pixels into native buffer ─────────────────────────────────────
      pixelPtr = calloc<Uint8>(bytes.length);
      pixelPtr.asTypedList(bytes.length).setAll(0, bytes);

      // ── Native inference call ─────────────────────────────────────────────
      jsonPtr = _detectObjectsBGRA(_detector, pixelPtr, width, height, width * 4);

      final jsonStr = jsonPtr.toDartString();
      print('JSON Detection: $jsonStr'); // always log result

      if (jsonStr.isEmpty) return const [];
      return BallDetectionResults.fromJson(jsonStr).detections;
    } catch (e, st) {
      print('detectFromImage error: $e');
      print(st);
      return const [];
    } finally {
      if (pixelPtr != null) calloc.free(pixelPtr);
      //if (jsonPtr  != null && jsonPtr != nullptr) _freeCString(jsonPtr);
      _isDetecting = false;
    }
  }

  // ────────── PRIVATE HELPERS ──────────
  Future<void> _loadLibrary() async {
    final dylib = LibraryLoader.library;

    _initializeDetector = dylib
        .lookup<NativeFunction<InitializeDetectorC>>('initialize_detector')
        .asFunction();
    _detectObjectsBGRA = dylib
        .lookup<NativeFunction<DetectObjectsBGRAC>>('detect_objects_bgra')
        .asFunction();
    _releaseDetector = dylib
        .lookup<NativeFunction<ReleaseDetectorC>>('release_detector')
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

