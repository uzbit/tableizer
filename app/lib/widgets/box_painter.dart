import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../detection_box.dart';

/// Paints YOLO / detector boxes on top of a portrait camera preview.
///
/// Call from a `CustomPaint( painter: BoxPainter( … ) )`.
class BoxPainter extends CustomPainter {
  BoxPainter({
    required this.sensorSize,             // e.g. Size(1280, 720)
    required this.detections,
  });

  final ui.Size sensorSize;               // landscape sensor resolution
  final List<Detection> detections;

  // ───────────────────  core transform  ───────────────────
  Rect _sensorToScreen(Rect r, ui.Size screen, BoxFit fit) {
    // 1. rotate sensor 90° CW → portrait buffer size
    final ui.Size bufSize = ui.Size(sensorSize.height, sensorSize.width);

    // 2. same maths Flutter uses for CameraPreview (BoxFit.cover)
    final fitted = applyBoxFit(fit, bufSize, screen);
    final src = Alignment.center.inscribe(fitted.source, Offset.zero & bufSize);
    final dst = Alignment.center.inscribe(fitted.destination, Offset.zero & screen);

    final double scale = dst.width / src.width; // X and Y scale are equal for cover
    final Offset shift = dst.topLeft;

    // shift into cropped src space, then scale & shift to screen
    final Rect cropped = r.shift(-src.topLeft);
    return Rect.fromLTWH(
      cropped.left   * scale + shift.dx,
      cropped.top    * scale + shift.dy,
      cropped.width  * scale,
      cropped.height * scale,
    );
  }

  // ───────────────────  painter  ───────────────────
  @override
  void paint(Canvas canvas, ui.Size size) {
    if (detections.isEmpty) return;

    for (final d in detections) {
      // 1. rotate raw bbox 90° CCW from sensor-landscape → buffer-portrait
      final Rect rotated = Rect.fromLTWH(
        sensorSize.height - d.box.y.toDouble() - d.box.height.toDouble(),
        d.box.x.toDouble(),
        d.box.height.toDouble(),
        d.box.width.toDouble(),
      );

      // 2. map into the on-screen preview rectangle
      final Rect onScreen = _sensorToScreen(rotated, size, BoxFit.cover);

      // 3. draw
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..color = _colorForClass(d.classId);

      canvas.drawRect(onScreen, paint);

      // optional label
      final tp = TextPainter(
        text: TextSpan(
          text: '${d.classId} ${(d.confidence * 100).toStringAsFixed(1)}%',
          style: TextStyle(color: paint.color, fontSize: 12),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, onScreen.topLeft.translate(0, -tp.height));
    }
  }

  @override
  bool shouldRepaint(covariant BoxPainter old) =>
      old.detections != detections || old.sensorSize != sensorSize;

  // simple deterministic colour wheel
  Color _colorForClass(int id) {
    const colors = [
      Colors.black, Colors.white, Colors.yellow,
      Colors.red, //Colors.purple, Colors.cyan,
    ];
    return colors[id % colors.length];
  }
}