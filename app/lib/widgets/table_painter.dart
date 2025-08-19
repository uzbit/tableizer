import 'dart:ui' as ui;

import 'package:flutter/material.dart';

class TablePainter extends CustomPainter {
  TablePainter({
    required this.imageSize,
    required this.quadPoints,
  });

  final ui.Size? imageSize;
  final List<Offset> quadPoints;

  @override
  void paint(Canvas canvas, ui.Size size) {
    if (quadPoints.isEmpty || imageSize == null) return;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 10
      ..color = Colors.red;

    // The Positioned widget in the parent has already aligned our canvas
    // with the camera preview. We just need to scale the points from the
    // image coordinate space to our canvas's coordinate space.
    final double scaleX = size.width / imageSize!.width;
    final double scaleY = size.height / imageSize!.height;

    final path = Path();
    for (int i = 0; i < quadPoints.length; i++) {
      final point = quadPoints[i];
      final Offset screenPoint = Offset(point.dx * scaleX, point.dy * scaleY);

      if (i == 0) {
        path.moveTo(screenPoint.dx, screenPoint.dy);
      } else {
        path.lineTo(screenPoint.dx, screenPoint.dy);
      }
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant TablePainter old) =>
      old.quadPoints != quadPoints || old.imageSize != imageSize;
}
