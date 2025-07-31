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

typedef DetectObjectsRGBAC = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectObjectsRGBADart = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, int width, int height, int channels);

typedef DetectObjectsYUVC = Pointer<Utf8> Function(
    Pointer<Void> detector,
    Pointer<Uint8> yPlane,
    Pointer<Uint8> uPlane,
    Pointer<Uint8> vPlane,
    Int32 width,
    Int32 height,
    Int32 yStride,
    Int32 uStride,
    Int32 vStride);
typedef DetectObjectsYUVDart = Pointer<Utf8> Function(
    Pointer<Void> detector,
    Pointer<Uint8> yPlane,
    Pointer<Uint8> uPlane,
    Pointer<Uint8> vPlane,
    int width,
    int height,
    int yStride,
    int uStride,
    int vStride);

typedef ReleaseDetectorC = Void Function(Pointer<Void> detector);
typedef ReleaseDetectorDart = void Function(Pointer<Void> detector);

class DetectionService {
  Pointer<Void> _detector = nullptr;
  late final InitializeDetectorDart _initializeDetector;
  late final DetectObjectsRGBADart _detectObjectsRGBA;
  late final DetectObjectsYUVDart _detectObjectsYUV;
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
  Future<List<Detection>> detectFromRGBImage(img.Image src) async {
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
      jsonPtr = _detectObjectsRGBA(
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

  Future<List<Detection>> detectFromYUV(CameraImage img) async {
    if (_isDetecting || _detector == nullptr) return const [];
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

      final jsonPtr = _detectObjectsYUV(
        _detector,
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
      print('JSON YUV Detection: $json');

      return json.isEmpty ? const [] : Detections.fromJson(json).detections;
    } finally {
      if (yPtr != null) calloc.free(yPtr);
      if (uPtr != null) calloc.free(uPtr);
      if (vPtr != null) calloc.free(vPtr);
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
    _detectObjectsRGBA = dylib
        .lookup<NativeFunction<DetectObjectsRGBAC>>('detect_objects_rgba')
        .asFunction();
    _detectObjectsYUV = dylib
        .lookup<NativeFunction<DetectObjectsYUVC>>('detect_objects_yuv')
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
