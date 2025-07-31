import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../detection_box.dart';

class BoxPainter extends CustomPainter {
  final List<Detection> detections;
  final ui.Size imageSize;
  final ui.Size screenSize;

  BoxPainter(
      {required this.detections,
      required this.imageSize,
      required this.screenSize});

  @override
  void paint(Canvas canvas, ui.Size size) {
    if (imageSize.isEmpty) return;

    final double scaleX = screenSize.width / imageSize.width;
    final double scaleY = screenSize.height / imageSize.height;

    final Paint paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    for (final detection in detections) {
      final rect = Rect.fromLTWH(
        detection.box.x * scaleX,
        detection.box.y * scaleY,
        detection.box.width * scaleX,
        detection.box.height * scaleY,
      );

      paint.color = _getColorForClass(detection.classId);
      canvas.drawRect(rect, paint);

      final TextPainter textPainter = TextPainter(
        text: TextSpan(
          text:
              '${detection.classId} (${(detection.confidence * 100).toStringAsFixed(2)}%)',
          style: TextStyle(
            color: paint.color,
            fontSize: 12.0,
            backgroundColor: Colors.black.withOpacity(0.5),
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);
      textPainter.paint(canvas, rect.topLeft.translate(0, -14));
    }
  }

  Color _getColorForClass(int classId) {
    switch (classId) {
      case 0:
        return Colors.red;
      case 1:
        return Colors.green;
      case 2:
        return Colors.blue;
      case 3:
        return Colors.yellow;
      default:
        return Colors.purple;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
