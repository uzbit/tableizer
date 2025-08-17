import 'dart:io';
import 'package:app/services/ball_detection_service.dart';
import 'package:app/services/table_detection_service.dart';
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

    final detections = await detectionService.detectTableFromRawRgba(
        bytes, rgbaImage.width, rgbaImage.height);
    print('Table detections: $detections');
    expect(detections.containsKey('quad_points'), isTrue);
    expect(detections['quad_points'].length, 4);
    expect(detections.containsKey('image'), isTrue);
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
