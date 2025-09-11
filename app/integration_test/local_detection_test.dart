import 'dart:convert';
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

    final img.Image bgraImage = testImage!.convert(numChannels: 4);
    final bytes = bgraImage.getBytes(order: img.ChannelOrder.bgra);

    final detectionService = BallDetectionService();
    await detectionService.initialize();

    final detections = await detectionService.detectBallsFromBytes(
        bytes, bgraImage.width, bgraImage.height);
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

    final img.Image bgraImage = testImage!.convert(numChannels: 4);
    final bytes = bgraImage.getBytes(order: img.ChannelOrder.bgra);

    final detectionService = TableDetectionService();
    await detectionService.initialize();

    // --- Direct FFI call for testing ---
    final Pointer<Uint8> imagePtr = calloc<Uint8>(bytes.length);
    imagePtr.asTypedList(bytes.length).setAll(0, bytes);
    Pointer<Utf8> resultPtr = nullptr;

    try {
      resultPtr = detectionService.detectTableBgra(
        imagePtr,
        bgraImage.width,
        bgraImage.height,
        bgraImage.width * 4,
          0, // rotation deg
          nullptr // debug image location string
      );

      // --- Assertions ---
      expect(resultPtr, isNot(nullptr));
      final jsonResult = resultPtr.toDartString();
      final Map<String, dynamic> parsed = json.decode(jsonResult);
      
      expect(parsed.containsKey('error'), false);
      final List<dynamic> quadPointsJson = parsed['quad_points'] ?? [];
      expect(quadPointsJson.length, 4);

      // EXPECTED QUAD (in resized detection image coordinates):
      final expectedPoints = [
        img.Point(1045.0, 1729.0),
        img.Point(2129.0, 1709.0),
        img.Point(3163.0, 3109.0),
        img.Point(42.0, 3205.0),
      ];

      for (int i = 0; i < 4; i++) {
        expect(quadPointsJson[i][0], closeTo(expectedPoints[i].x, 5));
        expect(quadPointsJson[i][1], closeTo(expectedPoints[i].y, 5));
      }

    } finally {
      // --- CRITICAL: Memory cleanup ---
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
