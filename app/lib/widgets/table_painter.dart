import 'package:flutter/material.dart';

class TablePainter extends CustomPainter {
  TablePainter({
    required this.quadPoints,
    this.orientation,
  });

  final List<Offset> quadPoints;
  final String? orientation;

  @override
  void paint(Canvas canvas, Size size) {
    if (quadPoints.isEmpty) return;

    // Green for SHORT_SIDE orientation, red for all others
    final Color quadColor = orientation == 'SHORT_SIDE'
        ? Colors.green
        : Colors.red;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..color = quadColor;

    final path = Path();
    if (quadPoints.isNotEmpty) {
      path.moveTo(quadPoints.first.dx, quadPoints.first.dy);
      for (int i = 1; i < quadPoints.length; i++) {
        path.lineTo(quadPoints[i].dx, quadPoints[i].dy);
      }
      path.close();
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant TablePainter old) =>
      old.quadPoints != quadPoints || old.orientation != orientation;
}
