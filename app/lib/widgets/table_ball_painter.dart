import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';
import '../models/ball_detection_result.dart';
import '../models/table_detection_result.dart';
import '../services/settings_service.dart';

class TableBallPainter extends CustomPainter {
  final List<BallDetectionResult> detections;
  final ui.Size? capturedImageSize;
  final ui.Size tableDisplaySize;
  final TableDetectionResult? tableDetectionResult;
  final InputAnalysisImageRotation? capturedRotation;
  final List<Offset>? Function(List<Offset>, List<Offset>, ui.Size, ui.Size, int)? transformPointsCallback;

  TableBallPainter({
    required this.detections,
    required this.capturedImageSize,
    required this.tableDisplaySize,
    this.tableDetectionResult,
    this.capturedRotation,
    this.transformPointsCallback,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (capturedImageSize == null || detections.isEmpty) return;

    // If we have table detection results, use proper coordinate transformation
    if (tableDetectionResult != null && tableDetectionResult!.points.length == 4) {
      // Get ball positions. These are in the padded "canvas" space from C++.
      // We need to convert them back to original image space before transforming.
      final canvasSize = tableDetectionResult!.imageSize;
      final originalSize = tableDetectionResult!.originalImageSize ?? capturedImageSize!;
      final offsetX = (canvasSize.width - originalSize.width) / 2;
      final offsetY = (canvasSize.height - originalSize.height) / 2;

      print('[TABLE_BALL_PAINTER] Canvas size: $canvasSize, Original size: $originalSize');
      print('[TABLE_BALL_PAINTER] Un-padding offset: ($offsetX, $offsetY)');
      print('[TABLE_BALL_PAINTER] Detected orientation: ${tableDetectionResult!.orientation}');

      // Log ball positions in canvas space (before un-padding)
      print('[TABLE_BALL_PAINTER] Ball positions IN CANVAS SPACE (${canvasSize.width}x${canvasSize.height}, before un-padding):');
      for (int i = 0; i < detections.length; i++) {
        print('  Ball $i: (${detections[i].centerX}, ${detections[i].centerY})');
      }

      var ballPositions = detections.map((d) {
        return Offset(d.centerX - offsetX, d.centerY - offsetY);
      }).toList();

      // Log ball positions after un-padding (in original image space)
      print('[TABLE_BALL_PAINTER] Ball positions AFTER UN-PADDING (in original ${originalSize.width}x${originalSize.height} space):');
      for (int i = 0; i < ballPositions.length; i++) {
        print('  Ball $i: ${ballPositions[i]}');
      }

      // Apply 90° rotation to ball positions if orientation is LONG_SIDE
      // ShotStudio displays table in portrait orientation, but LONG_SIDE captures
      // have the table appearing more horizontal, so we rotate coordinates to match
      if (tableDetectionResult!.orientation == 'LONG_SIDE') {
        print('[TABLE_BALL_PAINTER] Applying 90° clockwise rotation for LONG_SIDE orientation');
        ballPositions = ballPositions.map((pos) {
          // Rotate 90° clockwise: (x, y) -> (y, width - x)
          return Offset(pos.dy, originalSize.width - pos.dx);
        }).toList();
        print('[TABLE_BALL_PAINTER] Ball positions after rotation: $ballPositions');
      }

      // Rotation was already applied during normalization, so use rotation=0
      // and use the post-rotation original image size
      final int rotationDegrees = 0;
      final ui.Size transformImageSize = originalSize;

      print('[TABLE_BALL_PAINTER] transformPointsCallback is ${transformPointsCallback == null ? "NULL" : "available"}');

      // Log quad points being sent to C++ transform
      print('[TABLE_BALL_PAINTER] Quad points being sent to C++ (in original ${originalSize.width}x${originalSize.height} space):');
      for (int i = 0; i < tableDetectionResult!.points.length; i++) {
        print('  Quad[$i]: ${tableDetectionResult!.points[i]}');
      }

      print('[TABLE_BALL_PAINTER] Calling transformation: ballPositions=${ballPositions.length}, quadPoints=${tableDetectionResult!.points.length}, imageSize=$transformImageSize (post-rotation original), displaySize=$tableDisplaySize, rotation=$rotationDegrees (already rotated)');

      // Use C++ FFI transformation (rotation already applied during normalization)
      final transformedPositions = transformPointsCallback?.call(
        ballPositions,
        tableDetectionResult!.points,
        transformImageSize,
        tableDisplaySize,
        rotationDegrees,
      ) ?? ballPositions; // fallback to original positions if callback is null

      print('[TABLE_BALL_PAINTER] Transformation result: ${transformedPositions == ballPositions ? "UNCHANGED (callback was null or failed)" : "SUCCESS"}');

      // Draw balls at transformed positions
      print('[TABLE_BALL_PAINTER] Drawing ${transformedPositions.length} balls:');
      for (int i = 0; i < detections.length; i++) {
        if (i < transformedPositions.length) {
          print('[TABLE_BALL_PAINTER] Ball $i at ${transformedPositions[i]} (canvas size: ${tableDisplaySize.width}x${tableDisplaySize.height})');
          _drawBallAtPosition(canvas, detections[i], transformedPositions[i]);
        }
      }
    } else {
      // Fallback to simple scaling if no table detection available
      for (final detection in detections) {
        _drawBall(canvas, detection);
      }
    }
  }

  void _drawBall(Canvas canvas, BallDetectionResult detection) {
    // Transform coordinates from image space to table space (simple scaling fallback)
    final x = (detection.centerX / capturedImageSize!.width) * tableDisplaySize.width;
    final y = (detection.centerY / capturedImageSize!.height) * tableDisplaySize.height;

    _drawBallAtPosition(canvas, detection, Offset(x, y));
  }

  void _drawBallAtPosition(Canvas canvas, BallDetectionResult detection, Offset position) {
    // Ball colors based on class
    final ballColor = _getBallColor(detection.classId);
    
    // Calculate ball radius based on table size setting
    final settingsService = SettingsService();
    final ballRadius = BallScaling.calculateBallRadius(
      tableDisplaySize.width, 
      tableDisplaySize.height, 
      settingsService.tableSizeInches
    );

    // Draw ball shadow
    final shadowPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.3)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 2.0);
    
    canvas.drawCircle(
      Offset(position.dx + 2, position.dy + 2),
      ballRadius,
      shadowPaint,
    );

    // Draw ball
    final ballPaint = Paint()
      ..color = ballColor
      ..style = PaintingStyle.fill;

    canvas.drawCircle(
      position,
      ballRadius,
      ballPaint,
    );

    // Draw ball outline
    final outlinePaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawCircle(
      position,
      ballRadius,
      outlinePaint,
    );

    // Draw class label with scaled text size
    final textSize = BallScaling.calculateTextSize(ballRadius);
    final textStyle = TextStyle(
      color: Colors.white,
      fontSize: math.max(textSize, 8.0), // Minimum readable text size
      fontWeight: FontWeight.bold,
      shadows: [
        Shadow(
          offset: const Offset(1, 1),
          blurRadius: 2,
          color: Colors.black.withValues(alpha: 0.8),
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
      Offset(position.dx - textPainter.width / 2, position.dy + ballRadius + 4),
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