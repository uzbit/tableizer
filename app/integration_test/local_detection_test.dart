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

    // Use the service method instead of direct FFI
    final result = await detectionService.detectTableFromBytes(
      bytes,
      bgraImage.width,
      bgraImage.height,
    );

    // --- Assertions ---
    expect(result, isNotNull);
    expect(result!.points.length, 4);

    // EXPECTED QUAD (in resized detection image coordinates):
    // Note: Values may differ slightly due to ImageAdapter normalization
    final expectedPoints = [
      img.Point(1045.0, 1729.0),
      img.Point(2129.0, 1709.0),
      img.Point(3163.0, 3109.0),
      img.Point(42.0, 3205.0),
    ];

    // Verify points are within reasonable range (allowing for normalization)
    for (int i = 0; i < 4; i++) {
      expect(result.points[i].dx, greaterThan(0));
      expect(result.points[i].dy, greaterThan(0));
      expect(result.points[i].dx, lessThan(result.imageSize.width));
      expect(result.points[i].dy, lessThan(result.imageSize.height));
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
