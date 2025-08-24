import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../detection_box.dart';
import '../services/coordinate_transformer.dart';
import '../services/table_detection_result.dart';

class TableBallPainter extends CustomPainter {
  final List<Detection> detections;
  final ui.Size? capturedImageSize;
  final ui.Size tableDisplaySize;
  final TableDetectionResult? tableDetectionResult;

  TableBallPainter({
    required this.detections,
    required this.capturedImageSize,
    required this.tableDisplaySize,
    this.tableDetectionResult,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (capturedImageSize == null || detections.isEmpty) return;

    // If we have table detection results, use proper coordinate transformation
    if (tableDetectionResult != null && tableDetectionResult!.points.length == 4) {
      // Use simple coordinate mapping that matches the table orientation
      // The captured image quad points map to the display table rectangle
      final transformedPositions = _transformPointsUsingQuad(
        detections.map((d) => Offset(d.centerX, d.centerY)).toList(),
        tableDetectionResult!.points,
        capturedImageSize!,
        tableDisplaySize,
      );

      // Draw balls at transformed positions  
      for (int i = 0; i < detections.length; i++) {
        if (i < transformedPositions.length) {
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

  void _drawBall(Canvas canvas, Detection detection) {
    // Transform coordinates from image space to table space (simple scaling fallback)
    final x = (detection.centerX / capturedImageSize!.width) * tableDisplaySize.width;
    final y = (detection.centerY / capturedImageSize!.height) * tableDisplaySize.height;
    
    _drawBallAtPosition(canvas, detection, Offset(x, y));
  }

  /// Simple perspective transformation from quad to display rectangle
  List<Offset> _transformPointsUsingQuad(
    List<Offset> points,
    List<Offset> quadPoints,
    ui.Size imageSize,
    ui.Size displaySize,
  ) {
    if (quadPoints.length != 4 || points.isEmpty) return points;
    
    // Order quad points counter-clockwise starting from top-left
    final orderedQuad = _orderQuadPoints(quadPoints);
    
    // Calculate edge lengths to determine if rotation is needed
    final topLength = _distance(orderedQuad[0], orderedQuad[1]);     // top edge
    final rightLength = _distance(orderedQuad[1], orderedQuad[2]);   // right edge
    final needsRotation = topLength > rightLength * 1.25;  // If top > 1.25x right, table is landscape in camera, needs rotation
    
    // Apply rotation if needed
    List<Offset> workingQuad;
    if (needsRotation) {
      // Rotate quad points 90° CCW
      workingQuad = orderedQuad.map((point) {
        final centerX = imageSize.width / 2;
        final centerY = imageSize.height / 2;
        final translatedX = point.dx - centerX;
        final translatedY = point.dy - centerY;
        
        // 90° CCW: (x,y) → (-y, x)
        final rotatedX = -translatedY + centerX;
        final rotatedY = translatedX + centerY;
        
        return Offset(rotatedX, rotatedY);
      }).toList();
      print('ROTATION APPLIED: topLength=$topLength > rightLength=$rightLength');
    } else {
      // Use original quad
      workingQuad = orderedQuad;
      print('NO ROTATION: topLength=$topLength <= rightLength=$rightLength');
    }
    
    print('Table orientation (ROTATED): topLength=$topLength, rightLength=$rightLength, needsRotation=$needsRotation');
    print('Working quad points: $workingQuad');
    
    // Define the canonical table corners based on our display orientation
    // Key insight: Our DISPLAY table has long axis on Y (portrait: width=411, height=823)
    // But DETECTED table is landscape (top=752 > right=359)
    // We need to map the landscape quad to the portrait display
    
    List<Offset> canonicalCorners;
    
    // After rotation (if applied), always use direct mapping to portrait display
    canonicalCorners = [
      Offset(0, 0),                               // top-left → top-left
      Offset(displaySize.width, 0),               // top-right → top-right
      Offset(displaySize.width, displaySize.height), // bottom-right → bottom-right
      Offset(0, displaySize.height),              // bottom-left → bottom-left
    ];
    print('DIRECT MAPPING TO PORTRAIT DISPLAY');
    
    print('Canonical corners: $canonicalCorners');
    
    // Transform each ball point (apply same rotation as quad if needed)
    return points.map((point) {
      Offset workingBallPoint;
      
      if (needsRotation) {
        // Apply same 90° CCW rotation to ball point
        final centerX = imageSize.width / 2;
        final centerY = imageSize.height / 2;
        final translatedX = point.dx - centerX;
        final translatedY = point.dy - centerY;
        
        // 90° CCW rotation
        final rotatedBallX = -translatedY + centerX;
        final rotatedBallY = translatedX + centerY;
        workingBallPoint = Offset(rotatedBallX, rotatedBallY);
        
        print('Ball ${point} → rotated: ${workingBallPoint}');
      } else {
        // Use original ball point
        workingBallPoint = point;
        print('Ball ${point} → no rotation');
      }
      
      // Transform using working quad and working ball point
      final transformed = _transformPoint(workingBallPoint, workingQuad, canonicalCorners);
      print('Final: ${workingBallPoint} → ${transformed}');
      return transformed;
    }).toList();
  }
  
  /// Order quad points counter-clockwise starting from top-left
  List<Offset> _orderQuadPoints(List<Offset> points) {
    if (points.length != 4) return points;

    // Find centroid
    final centroid = Offset(
      points.map((p) => p.dx).reduce((a, b) => a + b) / 4,
      points.map((p) => p.dy).reduce((a, b) => a + b) / 4,
    );

    // Sort by angle from centroid (counter-clockwise)
    final sortedPoints = List<Offset>.from(points);
    sortedPoints.sort((a, b) {
      final angleA = math.atan2(a.dy - centroid.dy, a.dx - centroid.dx);
      final angleB = math.atan2(b.dy - centroid.dy, b.dx - centroid.dx);
      return angleA.compareTo(angleB);
    });

    return sortedPoints;
  }
  
  /// Calculate distance between two points
  double _distance(Offset p1, Offset p2) {
    final dx = p2.dx - p1.dx;
    final dy = p2.dy - p1.dy;
    return math.sqrt(dx * dx + dy * dy);
  }
  
  /// Transform a point using bilinear interpolation within the quad
  Offset _transformPoint(Offset point, List<Offset> srcQuad, List<Offset> dstCorners) {
    // Simplified bilinear transformation
    // For a proper implementation, we'd use perspective transform math
    // For now, use bounding box approximation
    
    final srcMinX = srcQuad.map((p) => p.dx).reduce(math.min);
    final srcMaxX = srcQuad.map((p) => p.dx).reduce(math.max);
    final srcMinY = srcQuad.map((p) => p.dy).reduce(math.min);
    final srcMaxY = srcQuad.map((p) => p.dy).reduce(math.max);
    
    // Normalize within source quad bounding box
    final normalizedX = (point.dx - srcMinX) / (srcMaxX - srcMinX);
    final normalizedY = (point.dy - srcMinY) / (srcMaxY - srcMinY);
    
    // Map to destination 
    final dstMinX = dstCorners.map((p) => p.dx).reduce(math.min);
    final dstMaxX = dstCorners.map((p) => p.dx).reduce(math.max);
    final dstMinY = dstCorners.map((p) => p.dy).reduce(math.min);
    final dstMaxY = dstCorners.map((p) => p.dy).reduce(math.max);
    
    return Offset(
      dstMinX + normalizedX * (dstMaxX - dstMinX),
      dstMinY + normalizedY * (dstMaxY - dstMinY),
    );
  }

  void _drawBallAtPosition(Canvas canvas, Detection detection, Offset position) {
    // Ball colors based on class
    final ballColor = _getBallColor(detection.classId);
    final ballRadius = 12.0;

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

    // Draw class label
    final textStyle = TextStyle(
      color: Colors.white,
      fontSize: 10,
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