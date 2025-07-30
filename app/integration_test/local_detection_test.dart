import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:integration_test/integration_test.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

// --- FFI Signatures ---
typedef InitializeDetectorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitializeDetectorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef DetectObjectsC = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectObjectsDart = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, int width, int height, int channels);

typedef ReleaseDetectorC = Void Function(Pointer<Void> detector);
typedef ReleaseDetectorDart = void Function(Pointer<Void> detector);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late InitializeDetectorDart initializeDetector;
  late DetectObjectsDart detectObjects;
  late ReleaseDetectorDart releaseDetector;
  late Pointer<Void> detector;

  setUpAll(() async {
    // When running on a device, we can load the library by its name.
    final dylib = DynamicLibrary.open("libtableizer_lib.so");

    initializeDetector = dylib
        .lookup<NativeFunction<InitializeDetectorC>>('initialize_detector')
        .asFunction();
    detectObjects = dylib
        .lookup<NativeFunction<DetectObjectsC>>('detect_objects')
        .asFunction();
    releaseDetector = dylib
        .lookup<NativeFunction<ReleaseDetectorC>>('release_detector')
        .asFunction();

    final modelPath = await _getTestAssetPath('assets/detection_model.onnx');
    detector = initializeDetector(modelPath.toNativeUtf8());
  });

  tearDownAll(() {
    if (detector != nullptr) {
      releaseDetector(detector);
    }
  });

  testWidgets('Detect objects in a local image on-device', (WidgetTester tester) async {
    final imagePath = await _getTestAssetPath('assets/images/P_20250718_203819.jpg');
    final imageBytes = await File(imagePath).readAsBytes();
    final img.Image? decodedImage = img.decodeImage(imageBytes);

    expect(decodedImage, isNotNull, reason: "Failed to decode image");

    final img.Image rgbaImage = img.Image(width: decodedImage!.width, height: decodedImage.height);
    for (int y = 0; y < decodedImage.height; ++y) {
        for (int x = 0; x < decodedImage.width; ++x) {
            final pixel = decodedImage.getPixel(x, y);
            rgbaImage.setPixelRgba(x, y, pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt(), pixel.a.toInt());
        }
    }

    final bytes = rgbaImage.getBytes(order: img.ChannelOrder.rgba);
    final imageBytesPtr = calloc<Uint8>(bytes.length);
    imageBytesPtr.asTypedList(bytes.length).setAll(0, bytes);

    final resultPtr = detectObjects(detector, imageBytesPtr, rgbaImage.width, rgbaImage.height, 4);
    final resultJson = resultPtr.toDartString();

    calloc.free(imageBytesPtr);

    print("On-device test detection results: $resultJson");
    expect(resultJson, isNotEmpty);
    expect(resultJson, startsWith('{"detections"'));
  });
}

// Helper to get asset paths in a test environment
Future<String> _getTestAssetPath(String assetName) async {
  final tempDir = await getApplicationSupportDirectory();
  final tempPath = p.join(tempDir.path, assetName);
  await Directory(p.dirname(tempPath)).create(recursive: true);
  
  final byteData = await rootBundle.load(assetName);
  final file = File(tempPath);
  await file.writeAsBytes(byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
  
  return file.path;
}
