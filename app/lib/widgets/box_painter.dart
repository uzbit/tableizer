import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:vector_math/vector_math_64.dart' hide Colors;
import 'package:tableizer_detection/tableizer_detection.dart';

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

    // 2. get the transform from bufSize to screen
    final fitted = applyBoxFit(fit, bufSize, screen);
    final src = Alignment.center.inscribe(fitted.source, Offset.zero & bufSize);
    final dst = Alignment.center.inscribe(fitted.destination, Offset.zero & screen);

    // 3. same as `src` but with zero origin
    final srcCropped = Rect.fromLTWH(0, 0, src.width, src.height);

    // 4. transform r from sensor-full → sensor-cropped → screen
    final matrix = Matrix4.identity()
      ..translate(dst.left, dst.top)
      ..scale(dst.width / srcCropped.width, dst.height / srcCropped.height)
      ..translate(-src.left, -src.top);

    final tl = matrix.transform3(Vector3(r.left, r.top, 0));
    final br = matrix.transform3(Vector3(r.right, r.bottom, 0));

    return Rect.fromLTRB(tl.x, tl.y, br.x, br.y);
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
    final colors = [
      Colors.black, Colors.white, Colors.yellow,
      Colors.red, //Colors.purple, Colors.cyan,
    ];
    return colors[id % colors.length];
  }
}