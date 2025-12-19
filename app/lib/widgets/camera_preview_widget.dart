import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../widgets/table_painter.dart';
import '../widgets/bullseye_painter.dart';
import '../models/table_detection_result.dart';

class CameraPreviewWidget extends StatelessWidget {
  final TableDetectionResult? tableDetectionResult;
  final List<Offset> filteredQuadPoints;
  final double fps;

  const CameraPreviewWidget({
    super.key,
    required this.tableDetectionResult,
    required this.filteredQuadPoints,
    required this.fps,
  });

  List<Offset> _scalePoints({
    required List<Offset> points,
    required Size sourceSize,
    required Size targetSize,
    BoxFit fit = BoxFit.contain,
  }) {
    final double scaleX = targetSize.width / sourceSize.width;
    final double scaleY = targetSize.height / sourceSize.height;

    double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (fit == BoxFit.contain) {
      scale = scaleX < scaleY ? scaleX : scaleY;
      final scaledWidth = sourceSize.width * scale;
      final scaledHeight = sourceSize.height * scale;
      offsetX = (targetSize.width - scaledWidth) / 2;
      offsetY = (targetSize.height - scaledHeight) / 2;
    } else {
      scale = 1.0;
    }

    return points.map((point) {
      return Offset(
        point.dx * scale + offsetX,
        point.dy * scale + offsetY,
      );
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Table detection overlay
        if (tableDetectionResult != null && filteredQuadPoints.isNotEmpty)
          LayoutBuilder(builder: (context, constraints) {
            final previewSize = constraints.biggest;
            final result = tableDetectionResult!;

            // Use filtered quad points (alpha-smoothed) instead of raw points
            // Points are in rotated image space (before padding)
            // Scale directly from original image size to display size
            final scaledPoints = _scalePoints(
              points: filteredQuadPoints,
              sourceSize: result.originalImageSize ?? result.imageSize,
              targetSize: previewSize,
              fit: BoxFit.contain,
            );

            return CustomPaint(
              size: previewSize,
              painter: TablePainter(
                quadPoints: scaledPoints,
                orientation: result.orientation,
              ),
            );
          }),

        // Bullseye Reticule (Center)
        Center(
          child: CustomPaint(
            size: const Size(40, 40),
            painter: BullseyePainter(),
          ),
        ),

        // FPS Counter
        Align(
          alignment: Alignment.topLeft,
          child: Container(
            color: Colors.black.withValues(alpha: 0.5),
            child: Text(
              'FPS: ${fps.toStringAsFixed(1)}',
              style: const TextStyle(color: Colors.white, fontSize: 20),
            ),
          ),
        ),
      ],
    );
  }
}