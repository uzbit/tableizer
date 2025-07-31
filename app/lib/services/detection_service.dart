import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:isolate';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import '../detection_box.dart';
import '../utils/image_converter.dart';

// --- FFI Signatures ---
typedef InitializeDetectorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitializeDetectorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef DetectObjectsC = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectObjectsDart = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, int width, int height, int channels);

typedef ReleaseDetectorC = Void Function(Pointer<Void> detector);
typedef ReleaseDetectorDart = void Function(Pointer<Void> detector);

class DetectionService {
  Pointer<Void> _detector = nullptr;
  late final InitializeDetectorDart _initializeDetector;
  late final DetectObjectsDart _detectObjects;
  late final ReleaseDetectorDart _releaseDetector;
  // late final void Function(Pointer<Utf8>) _freeCString;

  bool _isDetecting = false;

  // ────────── PUBLIC API ──────────
  Future<void> initialize() async {
    if (_detector != nullptr) return;                       // idempotent
    await _loadLibrary();

    final modelPath = await _copyAssetToFile('assets/detection_model.onnx');
    final pathPtr = modelPath.toNativeUtf8();
    _detector = _initializeDetector(pathPtr);
    calloc.free(pathPtr);

    if (_detector == nullptr) {
      throw StateError('Failed to create detector');
    }
  }

  Future<void> dispose() async {
    if (_detector != nullptr) {
      _releaseDetector(_detector);
      _detector = nullptr;
    }
  }

  /// Runs native detection on an *already decoded* `package:image` image.
  ///
  /// * `src` can be RGB (3 channels) or RGBA (4 channels).
  /// * Returns an empty list on failure or when no objects are detected.
  Future<List<Detection>> detectFromImage(img.Image src) async {
    // Skip if detector busy or not initialised.
    if (_isDetecting || _detector == nullptr) return const [];

    _isDetecting = true;

    Pointer<Uint8>? pixelPtr;
    Pointer<Utf8>?  jsonPtr;

    try {
      // ── Ensure 4-channel RGBA (conversion is a no-op if already RGBA) ───────
      final img.Image rgba =
      src.numChannels == 4 ? src : src.convert(numChannels: 4);

      // ── Copy pixels into native buffer ─────────────────────────────────────
      final bytes = rgba.getBytes(order: img.ChannelOrder.rgba);
      pixelPtr = calloc<Uint8>(bytes.length)
        ..asTypedList(bytes.length).setAll(0, bytes);

      // ── Native inference call ─────────────────────────────────────────────
      jsonPtr = _detectObjects(
          _detector, pixelPtr, rgba.width, rgba.height, 4);

      final jsonStr = jsonPtr.toDartString();
      print('JSON Detection: $jsonStr');          // always log result

      if (jsonStr.isEmpty) return const [];
      return Detections.fromJson(jsonStr).detections;
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
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open('libtableizer_lib.so')
        : DynamicLibrary.process();

    _initializeDetector = dylib
        .lookup<NativeFunction<InitializeDetectorC>>('initialize_detector')
        .asFunction();
    _detectObjects = dylib
        .lookup<NativeFunction<DetectObjectsC>>('detect_objects')
        .asFunction();
    _releaseDetector = dylib
        .lookup<NativeFunction<ReleaseDetectorC>>('release_detector')
        .asFunction();

    // C helper: void free_cstring(char*)
    // _freeCString = dylib
    //     .lookup<NativeFunction<Void Function(Pointer<Utf8>)>>('free_cstring')
    //     .asFunction();
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


//
// void _detectionIsolate(Map<String, dynamic> context) {
//   final detectorPtr = Pointer<Void>.fromAddress(context['detector_ptr']);
//   final CameraImage image = context['camera_image'];
//   final SendPort sendPort = context['send_port'];
//
//   final convertedImage = convertCameraImage(image);
//
//   final imageBytes = convertedImage.getBytes(order: img.ChannelOrder.rgba);
//   if (convertedImage.lengthInBytes != imageBytes.lengthInBytes) {
//     sendPort.send('{"error": "Image buffer size mismatch"}');
//     return;
//   }
//
//   final imageBytesPtr = calloc<Uint8>(convertedImage.lengthInBytes);
//   imageBytesPtr.asTypedList(convertedImage.lengthInBytes).setAll(0, imageBytes);
//
//   final dylib = Platform.isAndroid
//       ? DynamicLibrary.open("libtableizer_lib.so")
//       : DynamicLibrary.process();
//   final detectObjects = dylib
//       .lookup<NativeFunction<DetectObjectsC>>('detect_objects')
//       .asFunction<DetectObjectsDart>();
//
//   final resultPtr = detectObjects(
//       detectorPtr, imageBytesPtr, convertedImage.width, convertedImage.height, 4);
//   final resultJson = resultPtr.toDartString();
//
//   calloc.free(imageBytesPtr);
//
//   sendPort.send(resultJson);
// }
