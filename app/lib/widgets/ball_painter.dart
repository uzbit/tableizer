import 'dart:ui';
import 'package:flutter/material.dart';
import '../models/ball_detection_result.dart';

class BallPainter extends CustomPainter {
  final List<BallDetectionResult> detections;
  final Size imageSize;
  final Size displaySize;

  BallPainter({
    required this.detections,
    required this.imageSize,
    required this.displaySize,
  });

  String _getClassLabel(int classId) {
    switch (classId) {
      case 0:
        return 'Black';
      case 1:
        return 'Cue';
      case 2:
        return 'Solid';
      case 3:
        return 'Stripe';
      default:
        return 'Ball';
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    // Calculate scale factors to transform detection coordinates to display coordinates
    final double scaleX = displaySize.width / imageSize.width;
    final double scaleY = displaySize.height / imageSize.height;

    // Use the smaller scale to maintain aspect ratio (like BoxFit.contain)
    final double scale = scaleX < scaleY ? scaleX : scaleY;
    
    // Calculate offset to center the scaled image
    final double offsetX = (displaySize.width - (imageSize.width * scale)) / 2;
    final double offsetY = (displaySize.height - (imageSize.height * scale)) / 2;

    // Paint for bounding boxes
    final Paint boxPaint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // Paint for confidence text background
    final Paint textBgPaint = Paint()
      ..color = Colors.black54;

    for (final detection in detections) {
      // Transform bounding box coordinates
      final transformedBox = Rect.fromLTWH(
        (detection.box.x * scale) + offsetX,
        (detection.box.y * scale) + offsetY,
        detection.box.width * scale,
        detection.box.height * scale,
      );

      // Draw bounding box
      canvas.drawRect(transformedBox, boxPaint);

      // Draw class and confidence text
      final classLabel = _getClassLabel(detection.classId);
      final confidencePercent = (detection.confidence * 100).toStringAsFixed(1);
      final labelText = '$classLabel ${confidencePercent}%';
      
      final textSpan = TextSpan(
        text: labelText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      );
      
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      // Background for text
      final textBgRect = Rect.fromLTWH(
        transformedBox.left,
        transformedBox.top - 20,
        textPainter.width + 6,
        18,
      );
      canvas.drawRect(textBgRect, textBgPaint);

      // Draw text
      textPainter.paint(
        canvas,
        Offset(transformedBox.left + 3, transformedBox.top - 17),
      );
    }
  }

  @override
  bool shouldRepaint(BallPainter oldDelegate) {
    return oldDelegate.detections != detections ||
           oldDelegate.imageSize != imageSize ||
           oldDelegate.displaySize != displaySize;
  }
}