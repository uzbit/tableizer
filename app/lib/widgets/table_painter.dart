import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:vector_math/vector_math_64.dart' hide Colors;

class TablePainter extends CustomPainter {
  TablePainter({
    required this.sensorSize,
    required this.quadPoints,
  });

  final ui.Size sensorSize;
  final List<Offset> quadPoints;

  Rect _sensorToScreen(Rect r, ui.Size screen, BoxFit fit) {
    final ui.Size bufSize = ui.Size(sensorSize.height, sensorSize.width);
    final fitted = applyBoxFit(fit, bufSize, screen);
    final src = Alignment.center.inscribe(fitted.source, Offset.zero & bufSize);
    final dst = Alignment.center.inscribe(fitted.destination, Offset.zero & screen);
    final srcCropped = Rect.fromLTWH(0, 0, src.width, src.height);

    final matrix = Matrix4.identity()
      ..translate(dst.left, dst.top)
      ..scale(dst.width / srcCropped.width, dst.height / srcCropped.height)
      ..translate(-src.left, -src.top);

    final tl = matrix.transform3(Vector3(r.left, r.top, 0));
    final br = matrix.transform3(Vector3(r.right, r.bottom, 0));

    return Rect.fromLTRB(tl.x, tl.y, br.x, br.y);
  }

  @override
  void paint(Canvas canvas, ui.Size size) {
    if (quadPoints.isEmpty) return;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.red;

    final path = Path();
    for (int i = 0; i < quadPoints.length; i++) {
      final point = quadPoints[i];
      final rotated = Rect.fromLTWH(
        sensorSize.height - point.dy,
        point.dx,
        1,
        1,
      );
      final onScreen = _sensorToScreen(rotated, size, BoxFit.cover);
      if (i == 0) {
        path.moveTo(onScreen.left, onScreen.top);
      } else {
        path.lineTo(onScreen.left, onScreen.top);
      }
    }
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant TablePainter old) =>
      old.quadPoints != quadPoints || old.sensorSize != sensorSize;
}
