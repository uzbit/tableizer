import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../widgets/table_painter.dart';
import '../widgets/bullseye_painter.dart';

class CameraPreviewWidget extends StatelessWidget {
  final List<Offset> quadPoints;
  final ui.Size? imageSize;
  final double fps;

  const CameraPreviewWidget({
    super.key,
    required this.quadPoints,
    required this.imageSize,
    required this.fps,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Table detection overlay
        if (imageSize != null)
          LayoutBuilder(builder: (context, constraints) {
            final previewSize = constraints.biggest;
            final imageSize = this.imageSize!;

            final imageAspectRatio = imageSize.width / imageSize.height;
            final previewAspectRatio = previewSize.width / previewSize.height;

            double scale;
            // This logic mimics BoxFit.contain.
            if (previewAspectRatio > imageAspectRatio) {
              // Preview is wider than the image -> letterbox
              scale = previewSize.height / imageSize.height;
            } else {
              // Preview is taller than the image -> pillarbox
              scale = previewSize.width / imageSize.width;
            }

            final scaledWidth = imageSize.width * scale;
            final scaledHeight = imageSize.height * scale;

            // Center the scaled image within the preview.
            final dx = (previewSize.width - scaledWidth) / 2.0;
            final dy = (previewSize.height - scaledHeight) / 2.0;

            // Transform points from image-space to screen-space.
            final scaledPoints = quadPoints.map((p) {
              return Offset(p.dx * scale + dx, p.dy * scale + dy);
            }).toList();

            return CustomPaint(
              size: previewSize,
              painter: TablePainter(quadPoints: scaledPoints),
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