import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import '../detection_box.dart';
import '../services/table_detection_result.dart';
import '../services/settings_service.dart';

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

  /// Simple approach: rotate the input data if needed, then do perspective transform
  List<Offset> _transformPointsUsingQuad(
    List<Offset> points,
    List<Offset> quadPoints,
    ui.Size imageSize,
    ui.Size displaySize,
  ) {
    if (quadPoints.length != 4 || points.isEmpty) return points;
    
    // Order quad points counter-clockwise 
    final orderedQuad = _orderQuadPoints(quadPoints);
    
    print('Original quad points: $quadPoints');
    print('Ordered quad points: $orderedQuad');
    print('Quad[0] (top-left): ${orderedQuad[0]}');
    print('Quad[1] (top-right): ${orderedQuad[1]}'); 
    print('Quad[2] (bottom-right): ${orderedQuad[2]}');
    print('Quad[3] (bottom-left): ${orderedQuad[3]}');
    
    // Check if we need rotation based on quad orientation
    final topLength = _distance(orderedQuad[0], orderedQuad[1]);
    final rightLength = _distance(orderedQuad[1], orderedQuad[2]);
    final needsRotation = topLength > rightLength * 1.5;
    
    print('Quad analysis: topLength=$topLength, rightLength=$rightLength, needsRotation=$needsRotation');
    
    // When needsRotation=true, we need to map the landscape quad to portrait display
    // The key insight: don't rotate the input data, just map the quad corners differently
    
    List<Offset> dstCorners;
    
    if (needsRotation) {
      // Map landscape quad to portrait display by rotating the destination mapping 90° CCW
      // The quad is landscape (wider than tall), display is portrait (taller than wide)
      dstCorners = [
        Offset(0, displaySize.height),                       // top-left of quad → bottom-left of display
        Offset(0, 0),                                        // top-right of quad → top-left of display
        Offset(displaySize.width, 0),                        // bottom-right of quad → top-right of display  
        Offset(displaySize.width, displaySize.height),       // bottom-left of quad → bottom-right of display
      ];
      print('Using rotated destination mapping for landscape quad (90° CCW)');
    } else {
      // Direct mapping for portrait quad
      dstCorners = [
        Offset(0, 0),
        Offset(displaySize.width, 0),
        Offset(displaySize.width, displaySize.height),
        Offset(0, displaySize.height),
      ];
      print('Using direct mapping for portrait quad');
    }
    
    return points.map((point) {
      final transformed = _transformPoint(point, orderedQuad, dstCorners);
      print('Ball ${point} → ${transformed}');
      return transformed;
    }).toList();
  }
  
  /// Build combined transformation matrix: Perspective * Rotation (if needed)
  List<List<double>> _buildTransformMatrix(
    List<Offset> quadPoints, 
    ui.Size displaySize, 
    bool needsRotation
  ) {
    // Step 1: Create destination corners for perspective transform
    List<Offset> dstCorners;
    
    // Always map to the final display rectangle 
    dstCorners = [
      Offset(0, 0),
      Offset(displaySize.width, 0),
      Offset(displaySize.width, displaySize.height),
      Offset(0, displaySize.height),
    ];
    
    // Step 2: Build perspective transform matrix (simplified)
    final perspectiveMatrix = _buildSimplePerspectiveMatrix(quadPoints, dstCorners);
    
    // Step 3: Apply rotation matrix if needed 
    if (needsRotation) {
      // Rotate around center of display rectangle
      final centerX = displaySize.width / 2;
      final centerY = displaySize.height / 2;
      
      // Combined rotation matrix: Translate to center + Rotate 90° CCW + Translate back
      // Final transformation: (x,y) → (-y + centerX + centerY, x - centerX + centerY)
      final rotationMatrix = [
        [0.0, -1.0, centerX + centerY],
        [1.0, 0.0, centerY - centerX],  
        [0.0, 0.0, 1.0],
      ];
      
      return _multiplyMatrix3x3(rotationMatrix, perspectiveMatrix);
    }
    
    return perspectiveMatrix;
  }
  
  /// Build simple perspective transform matrix from quad to rectangle
  List<List<double>> _buildSimplePerspectiveMatrix(List<Offset> src, List<Offset> dst) {
    // Simplified affine approximation
    final srcRect = _boundingRect(src);
    final dstRect = _boundingRect(dst);
    
    final scaleX = dstRect.width / srcRect.width;
    final scaleY = dstRect.height / srcRect.height;
    final translateX = dstRect.left - srcRect.left * scaleX;
    final translateY = dstRect.top - srcRect.top * scaleY;
    
    return [
      [scaleX, 0.0, translateX],
      [0.0, scaleY, translateY],
      [0.0, 0.0, 1.0],
    ];
  }
  
  /// Multiply two 3x3 matrices
  List<List<double>> _multiplyMatrix3x3(List<List<double>> a, List<List<double>> b) {
    final result = List.generate(3, (_) => List<double>.filled(3, 0.0));
    
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    
    return result;
  }
  
  /// Apply 3x3 transformation matrix to a point
  Offset _applyMatrix3Transform(Offset point, List<List<double>> matrix) {
    final x = point.dx;
    final y = point.dy;
    
    final transformedX = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2];
    final transformedY = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2];
    final w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
    
    // Handle homogeneous coordinates
    if (w != 0 && w != 1.0) {
      return Offset(transformedX / w, transformedY / w);
    }
    
    return Offset(transformedX, transformedY);
  }
  
  /// Get bounding rectangle from points
  Rect _boundingRect(List<Offset> points) {
    if (points.isEmpty) return Rect.zero;
    
    double minX = points.first.dx, maxX = points.first.dx;
    double minY = points.first.dy, maxY = points.first.dy;
    
    for (final point in points) {
      if (point.dx < minX) minX = point.dx;
      if (point.dx > maxX) maxX = point.dx;
      if (point.dy < minY) minY = point.dy;
      if (point.dy > maxY) maxY = point.dy;
    }
    
    return Rect.fromLTRB(minX, minY, maxX, maxY);
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
  
  /// Transform a point using proper perspective transformation from quad to rectangle
  Offset _transformPoint(Offset point, List<Offset> srcQuad, List<Offset> dstCorners) {
    // Use bilinear interpolation within the quadrilateral
    // This approximates perspective transformation by treating the quad as two triangles
    
    // Find which triangle the point is in and use barycentric coordinates
    return _perspectiveTransform(point, srcQuad, dstCorners);
  }
  
  /// Proper perspective transformation using inverse bilinear interpolation
  Offset _perspectiveTransform(Offset point, List<Offset> srcQuad, List<Offset> dstRect) {
    // Ensure we have exactly 4 points
    if (srcQuad.length != 4 || dstRect.length != 4) {
      return point;
    }
    
    // Source quad points (should be ordered counter-clockwise)
    final p0 = srcQuad[0]; // top-left
    final p1 = srcQuad[1]; // top-right  
    final p2 = srcQuad[2]; // bottom-right
    final p3 = srcQuad[3]; // bottom-left
    
    // Destination rectangle corners
    final q0 = dstRect[0]; // top-left
    final q1 = dstRect[1]; // top-right
    final q2 = dstRect[2]; // bottom-right
    final q3 = dstRect[3]; // bottom-left
    
    // Find the normalized coordinates (u,v) within the source quadrilateral
    // This uses iterative solution to the bilinear equation
    final uv = _findUVInQuad(point, p0, p1, p2, p3);
    final u = uv.dx;
    final v = uv.dy;
    
    // Apply bilinear interpolation to destination rectangle
    final top = Offset.lerp(q0, q1, u)!;
    final bottom = Offset.lerp(q3, q2, u)!;
    return Offset.lerp(top, bottom, v)!;
  }
  
  /// Find normalized coordinates (u,v) of point P within quadrilateral defined by p0,p1,p2,p3
  Offset _findUVInQuad(Offset P, Offset p0, Offset p1, Offset p2, Offset p3) {
    // Use Newton's method to solve the bilinear equation:
    // P = (1-u)(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
    
    double u = 0.5, v = 0.5; // Initial guess
    
    for (int i = 0; i < 10; i++) { // Max 10 iterations
      // Current point using bilinear interpolation
      final currentP = Offset(
        (1-u)*(1-v)*p0.dx + u*(1-v)*p1.dx + u*v*p2.dx + (1-u)*v*p3.dx,
        (1-u)*(1-v)*p0.dy + u*(1-v)*p1.dy + u*v*p2.dy + (1-u)*v*p3.dy,
      );
      
      // Error vector
      final dx = P.dx - currentP.dx;
      final dy = P.dy - currentP.dy;
      
      // If close enough, break
      if (dx.abs() < 0.1 && dy.abs() < 0.1) break;
      
      // Jacobian matrix partial derivatives
      final dPdu_x = -(1-v)*p0.dx + (1-v)*p1.dx + v*p2.dx - v*p3.dx;
      final dPdu_y = -(1-v)*p0.dy + (1-v)*p1.dy + v*p2.dy - v*p3.dy;
      final dPdv_x = -(1-u)*p0.dx - u*p1.dx + u*p2.dx + (1-u)*p3.dx;
      final dPdv_y = -(1-u)*p0.dy - u*p1.dy + u*p2.dy + (1-u)*p3.dy;
      
      // Inverse Jacobian determinant
      final det = dPdu_x * dPdv_y - dPdu_y * dPdv_x;
      if (det.abs() < 1e-10) break; // Avoid division by zero
      
      // Newton step
      final du = (dPdv_y * dx - dPdv_x * dy) / det;
      final dv = (-dPdu_y * dx + dPdu_x * dy) / det;
      
      u += du;
      v += dv;
      
      // Clamp to [0,1]
      u = u.clamp(0.0, 1.0);
      v = v.clamp(0.0, 1.0);
    }
    
    return Offset(u, v);
  }

  void _drawBallAtPosition(Canvas canvas, Detection detection, Offset position) {
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