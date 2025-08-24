import 'dart:ui';
import 'package:flutter/material.dart';

class BullseyePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    // Draw concentric circles for bullseye
    canvas.drawCircle(center, 4, paint);   // Inner circle
    canvas.drawCircle(center, 12, paint);  // Middle circle
    canvas.drawCircle(center, 20, paint);  // Outer circle

    // Draw crosshairs
    const crosshairLength = 8.0;
    
    // Horizontal line
    canvas.drawLine(
      Offset(center.dx - crosshairLength, center.dy),
      Offset(center.dx + crosshairLength, center.dy),
      paint,
    );
    
    // Vertical line
    canvas.drawLine(
      Offset(center.dx, center.dy - crosshairLength),
      Offset(center.dx, center.dy + crosshairLength),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}