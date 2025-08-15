import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'table_detection_service.dart';

class DetectionIsolate {
  static Future<void> init(SendPort sendPort) async {
    final ReceivePort receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    final TableDetectionService tableDetectionService = TableDetectionService();
    await tableDetectionService.initialize();

    receivePort.listen((message) async {
      if (message is Map<String, dynamic>) {
        final Uint8List bytes = message['bytes'];
        final int width = message['width'];
        final int height = message['height'];

        final result = await tableDetectionService.detectTableFromByteBuffer(
            bytes, width, height);
        sendPort.send(result);
      }
    });
  }
}
