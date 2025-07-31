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
    if (imageSize.isEmpty || size.isEmpty) return;

    // --- DEBUG STEP 1: Draw a border around the entire canvas ---
    final canvasBorderPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.redAccent;
    canvas.drawRect(Offset.zero & size, canvasBorderPaint);

    // --- Calculate Transformation ---
    final rotatedImageSize = ui.Size(imageSize.height, imageSize.width);
    final fittedSizes = applyBoxFit(BoxFit.cover, rotatedImageSize, size);
    final sourceRect = Alignment.center
        .inscribe(fittedSizes.source, Offset.zero & rotatedImageSize);
    final destRect =
        Alignment.center.inscribe(fittedSizes.destination, Offset.zero & size);

    // --- DEBUG STEP 2: Draw the calculated area of the camera preview ---
    final previewAreaPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.blueAccent;
    canvas.drawRect(destRect, previewAreaPaint);

    final double scaleX = destRect.width / sourceRect.width;
    final double scaleY = destRect.height / sourceRect.height;
    final Offset offset = destRect.topLeft;

    // --- DEBUG STEP 3: Draw the final transformed bounding boxes ---
    for (final detection in detections) {
      final rotatedRect = Rect.fromLTWH(
        detection.box.y.toDouble(),
        imageSize.width - detection.box.x - detection.box.width,
        detection.box.height.toDouble(),
        detection.box.width.toDouble(),
      );

      final finalRect = Rect.fromLTWH(
        rotatedRect.left * scaleX + offset.dx,
        rotatedRect.top * scaleY + offset.dy,
        rotatedRect.width * scaleX,
        rotatedRect.height * scaleY,
      );

      final boxPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = _getColorForClass(detection.classId);
      canvas.drawRect(finalRect, boxPaint);

      final TextPainter textPainter = TextPainter(
        text: TextSpan(
          text:
              '${detection.classId} (${(detection.confidence * 100).toStringAsFixed(2)}%)',
          style: TextStyle(
            color: boxPaint.color,
            fontSize: 12.0,
            backgroundColor: Colors.black.withOpacity(0.5),
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout(minWidth: 0, maxWidth: size.width);
      textPainter.paint(canvas, finalRect.topLeft.translate(0, -14));
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
