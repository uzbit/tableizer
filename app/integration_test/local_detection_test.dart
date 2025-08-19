import 'dart:ffi';
import 'dart:io';
import 'package:app/services/ball_detection_service.dart';
import 'package:app/services/table_detection_service.dart';
import 'package:ffi/ffi.dart';
import 'package:image/image.dart' as img;

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

void main() {
  testWidgets('Detect balls in a local image on-device',
      (WidgetTester tester) async {
    final imagePath =
        await _getTestAssetPath('assets/images/P_20250718_203819.jpg');
    final imageBytes = await File(imagePath).readAsBytes();
    final img.Image? testImage = img.decodeImage(imageBytes);
    assert(testImage != null, 'Failed to decode test asset');

    final img.Image rgbaImage = testImage!.convert(numChannels: 4);
    final bytes = rgbaImage.getBytes(order: img.ChannelOrder.rgba);

    final detectionService = BallDetectionService();
    await detectionService.initialize();

    final detections = await detectionService.detectFromByteBuffer(
        bytes, rgbaImage.width, rgbaImage.height);
    final int count = detections.length;
    print('Number of detections: $count');
    expect(count, 16);
  });

  testWidgets('Detect table in a local image on-device',
      (WidgetTester tester) async {
    final imagePath =
        await _getTestAssetPath('assets/images/P_20250718_203819.jpg');
    final imageBytes = await File(imagePath).readAsBytes();
    final img.Image? testImage = img.decodeImage(imageBytes);
    assert(testImage != null, 'Failed to decode test asset');

    final img.Image rgbaImage = testImage!.convert(numChannels: 4);
    final bytes = rgbaImage.getBytes(order: img.ChannelOrder.rgba);

    final detectionService = TableDetectionService();
    await detectionService.initialize();

    // --- Direct FFI call for testing ---
    final Pointer<Uint8> imagePtr = calloc<Uint8>(bytes.length);
    imagePtr.asTypedList(bytes.length).setAll(0, bytes);
    Pointer<DetectionResult> resultPtr = nullptr;

    try {
      resultPtr = detectionService.detectTableBgra(
        imagePtr,
        rgbaImage.width,
        rgbaImage.height,
        rgbaImage.width * 4,
          0, // rotation deg
          nullptr // debug image location string
      );

      // --- Assertions ---
      expect(resultPtr, isNot(nullptr));
      final result = resultPtr.ref;
      expect(result.quad_points_count, 4);

      // EXPECTED QUAD (in resized detection image coordinates):
      // x, y: 1045.0, 1719.0
      // x, y: 2146.0, 1719.0
      // x, y: 3224.0, 3176.0
      // x, y: 44.0, 3176.0
      final expectedPoints = [
        img.Point(1045.0, 1719.0),
        img.Point(2146.0, 1719.0),
        img.Point(3224.0, 3176.0),
        img.Point(44.0, 3176.0),
      ];

      for (int i = 0; i < 4; i++) {
        expect(result.quad_points[i].x, closeTo(expectedPoints[i].x, 5));
        expect(result.quad_points[i].y, closeTo(expectedPoints[i].y, 5));
      }

    } finally {
      // --- CRITICAL: Memory cleanup ---
      if (resultPtr != nullptr) {
        detectionService.freeBgraDetectionResult(resultPtr);
      }
      calloc.free(imagePtr);
    }
  });
}

// Helper to get asset paths in a test environment
Future<String> _getTestAssetPath(String assetName) async {
  final tempDir = await getApplicationSupportDirectory();
  final tempPath = p.join(tempDir.path, assetName);
  await Directory(p.dirname(tempPath)).create(recursive: true);

  final byteData = await rootBundle.load(assetName);
  final file = File(tempPath);
  await file.writeAsBytes(
      byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));

  return file.path;
}
