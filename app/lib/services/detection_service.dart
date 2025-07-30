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
  late InitializeDetectorDart _initializeDetector;
  late DetectObjectsDart _detectObjects;
  late ReleaseDetectorDart _releaseDetector;

  bool _isDetecting = false;

  Future<void> initialize() async {
    await _loadLibrary();
    final modelPath = await _getAssetPath('assets/detection_model.onnx');
    _detector = _initializeDetector(modelPath.toNativeUtf8());
  }

  Future<void> _loadLibrary() async {
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open("libtableizer_lib.so")
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
  }

  Future<String> _getAssetPath(String asset) async {
    final path = p.join((await getApplicationSupportDirectory()).path, asset);
    await Directory(p.dirname(path)).create(recursive: true);
    final file = File(path);
    if (!await file.exists()) {
      final byteData = await rootBundle.load(asset);
      await file.writeAsBytes(
          byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }
    return path;
  }

  Future<List<Detection>> detect(CameraImage image) async {
    if (_isDetecting || _detector == nullptr) return [];

    _isDetecting = true;

    final port = ReceivePort();
    await Isolate.spawn(_detectionIsolate, {
      'detector_ptr': _detector.address,
      'camera_image': image,
      'send_port': port.sendPort,
    });

    final results = await port.first;
    _isDetecting = false;

    if (results != null && results is String && results.isNotEmpty) {
      final detections = Detections.fromJson(results);
      return detections.detections;
    } else {
      return [];
    }
  }

  void dispose() {
    if (_detector != nullptr) {
      _releaseDetector(_detector);
    }
  }

  Future<List<Detection>> detectFromBytes(Uint8List imageBytes) async {
    final img.Image? decodedImage = img.decodeImage(imageBytes);

    if (decodedImage == null) {
      print("Failed to decode image from assets");
      return [];
    }

    final img.Image rgbaImage = img.Image(width: decodedImage.width, height: decodedImage.height);
    for (int y = 0; y < decodedImage.height; ++y) {
        for (int x = 0; x < decodedImage.width; ++x) {
            final pixel = decodedImage.getPixel(x, y);
            rgbaImage.setPixelRgba(x, y, pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt(), pixel.a.toInt());
        }
    }

    final imageBytesPtr = calloc<Uint8>(rgbaImage.lengthInBytes);
    imageBytesPtr.asTypedList(rgbaImage.lengthInBytes).setAll(0, rgbaImage.getBytes(order: img.ChannelOrder.rgba));

    final resultPtr = _detectObjects(_detector, imageBytesPtr, rgbaImage.width, rgbaImage.height, 4);
    final resultJson = resultPtr.toDartString();

    calloc.free(imageBytesPtr);

    if (resultJson.isNotEmpty) {
      final detections = Detections.fromJson(resultJson);
      return detections.detections;
    } else {
      return [];
    }
  }
}

void _detectionIsolate(Map<String, dynamic> context) {
  final detectorPtr = Pointer<Void>.fromAddress(context['detector_ptr']);
  final CameraImage image = context['camera_image'];
  final SendPort sendPort = context['send_port'];

  final convertedImage = convertCameraImage(image);

  final imageBytes = convertedImage.getBytes(order: img.ChannelOrder.rgba);
  if (convertedImage.lengthInBytes != imageBytes.lengthInBytes) {
    sendPort.send('{"error": "Image buffer size mismatch"}');
    return;
  }

  final imageBytesPtr = calloc<Uint8>(convertedImage.lengthInBytes);
  imageBytesPtr.asTypedList(convertedImage.lengthInBytes).setAll(0, imageBytes);

  final dylib = Platform.isAndroid
      ? DynamicLibrary.open("libtableizer_lib.so")
      : DynamicLibrary.process();
  final detectObjects = dylib
      .lookup<NativeFunction<DetectObjectsC>>('detect_objects')
      .asFunction<DetectObjectsDart>();

  final resultPtr = detectObjects(
      detectorPtr, imageBytesPtr, convertedImage.width, convertedImage.height, 4);
  final resultJson = resultPtr.toDartString();

  calloc.free(imageBytesPtr);

  sendPort.send(resultJson);
}
