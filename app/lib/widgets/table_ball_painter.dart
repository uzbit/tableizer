import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../detection_box.dart';

class TableBallPainter extends CustomPainter {
  final List<Detection> detections;
  final ui.Size? capturedImageSize;
  final ui.Size tableDisplaySize;

  TableBallPainter({
    required this.detections,
    required this.capturedImageSize,
    required this.tableDisplaySize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (capturedImageSize == null || detections.isEmpty) return;

    // Transform coordinates from captured image to table display
    for (final detection in detections) {
      _drawBall(canvas, detection);
    }
  }

  void _drawBall(Canvas canvas, Detection detection) {
    // Transform coordinates from image space to table space
    // Since table is rotated 90 degrees, we need to transform coordinates accordingly
    final normalizedX = detection.centerX / capturedImageSize!.width;
    final normalizedY = detection.centerY / capturedImageSize!.height;
    
    // For 90-degree rotation: x' = y, y' = 1-x
    final rotatedX = normalizedY * tableDisplaySize.width;
    final rotatedY = (1 - normalizedX) * tableDisplaySize.height;
    
    final x = rotatedX;
    final y = rotatedY;

    // Ball colors based on class
    final ballColor = _getBallColor(detection.classId);
    final ballRadius = 12.0;

    // Draw ball shadow
    final shadowPaint = Paint()
      ..color = Colors.black.withOpacity(0.3)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2.0);
    
    canvas.drawCircle(
      Offset(x + 2, y + 2),
      ballRadius,
      shadowPaint,
    );

    // Draw ball
    final ballPaint = Paint()
      ..color = ballColor
      ..style = PaintingStyle.fill;

    canvas.drawCircle(
      Offset(x, y),
      ballRadius,
      ballPaint,
    );

    // Draw ball outline
    final outlinePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawCircle(
      Offset(x, y),
      ballRadius,
      outlinePaint,
    );

    // Draw class label
    final textStyle = TextStyle(
      color: Colors.white,
      fontSize: 10,
      fontWeight: FontWeight.bold,
      shadows: [
        Shadow(
          offset: const Offset(1, 1),
          blurRadius: 2,
          color: Colors.black.withOpacity(0.8),
        ),
      ],
    );

    final textSpan = TextSpan(
      text: _getClassLabel(detection.classId),
      style: textStyle,
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(x - textPainter.width / 2, y + ballRadius + 4),
    );
  }

  Color _getBallColor(int classId) {
    switch (classId) {
      case 0: // Black
        return Colors.black;
      case 1: // Cue
        return Colors.white;
      case 2: // Solid
        return Colors.red.shade700;
      case 3: // Stripe
        return Colors.yellow.shade600;
      default:
        return Colors.grey;
    }
  }

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
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return oldDelegate != this;
  }
}