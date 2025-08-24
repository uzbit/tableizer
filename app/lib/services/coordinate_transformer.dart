import 'dart:math' as math;
import 'dart:ui';

import 'package:vector_math/vector_math_64.dart';
import '../detection_box.dart';

class CoordinateTransformer {
  /// Transform ball detection coordinates from captured image space to canonical table space
  /// using the detected table quad points.
  /// 
  /// This mirrors the transformation logic from tableizer.cpp lines 114-145:
  /// 1. Scale from original image to table detection image
  /// 2. Apply perspective transform from quad points to canonical table space
  static List<Offset> transformBallsToCanonicalTable({
    required List<Detection> ballDetections,
    required Size capturedImageSize,
    required List<Offset> quadPoints,
    required Size tableDetectionImageSize,
    required Size canonicalTableSize,
  }) {
    if (ballDetections.isEmpty || quadPoints.length != 4) {
      return [];
    }

    // Step 1: Build scale transform (original image -> table detection image)
    final scaleX = tableDetectionImageSize.width / capturedImageSize.width;
    final scaleY = tableDetectionImageSize.height / capturedImageSize.height;

    // Step 2: Get ball centers in original image space
    final List<Offset> ballCenters = ballDetections
        .map((detection) => Offset(detection.centerX, detection.centerY))
        .toList();

    // Step 3: Apply scale transform to ball centers
    final List<Offset> scaledBallCenters = ballCenters.map((center) {
      return Offset(center.dx * scaleX, center.dy * scaleY);
    }).toList();

    // Step 4: Build perspective transform matrix from quad to canonical table
    final Matrix4 perspectiveMatrix = _buildPerspectiveTransform(
      quadPoints,
      canonicalTableSize,
    );

    // Step 5: Apply perspective transform to scaled ball centers
    final List<Offset> canonicalBallCenters = scaledBallCenters.map((center) {
      return _applyPerspectiveTransform(center, perspectiveMatrix);
    }).toList();

    return canonicalBallCenters;
  }

  /// Build a perspective transformation matrix from quad points to canonical table rectangle
  static Matrix4 _buildPerspectiveTransform(List<Offset> quadPoints, Size canonicalSize) {
    if (quadPoints.length != 4) {
      return Matrix4.identity();
    }

    // Define canonical table corners (destination points)
    // Since table long axis is on y-axis, canonicalSize has width < height (portrait orientation)
    final List<Offset> canonicalCorners = [
      const Offset(0, 0),                                    // Top-left
      Offset(canonicalSize.width, 0),                        // Top-right  
      Offset(canonicalSize.width, canonicalSize.height),     // Bottom-right
      Offset(0, canonicalSize.height),                       // Bottom-left
    ];

    // Order quad points to match canonical corners (assuming counter-clockwise)
    final orderedQuad = _orderQuadPoints(quadPoints);

    // Build perspective transform matrix using homography
    return _computeHomography(orderedQuad, canonicalCorners);
  }

  /// Order quad points counter-clockwise starting from top-left
  static List<Offset> _orderQuadPoints(List<Offset> points) {
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

  /// Compute homography matrix from source points to destination points
  /// This mirrors the C++ warpTable logic with rotation
  static Matrix4 _computeHomography(List<Offset> src, List<Offset> dst) {
    if (src.length != 4 || dst.length != 4) {
      return Matrix4.identity();
    }

    // Step 1: Compute basic perspective transform (like cv::getPerspectiveTransform)
    final perspectiveMatrix = _computePerspectiveTransform(src, dst);
    
    // Step 2: Apply 90° CCW rotation (matching C++ rot matrix)
    // C++ rotation matrix: [0, 1, 0; -1, 0, canvasW-1; 0, 0, 1]
    final canvasW = dst.map((p) => p.dx).reduce(math.max);
    final rotationMatrix = Matrix4.identity()
      ..setEntry(0, 0, 0)    // First row: [0, 1, 0, 0]
      ..setEntry(0, 1, 1)
      ..setEntry(0, 2, 0)
      ..setEntry(0, 3, 0)
      ..setEntry(1, 0, -1)   // Second row: [-1, 0, 0, canvasW-1]  
      ..setEntry(1, 1, 0)
      ..setEntry(1, 2, 0)
      ..setEntry(1, 3, canvasW - 1);
    
    // Step 3: Combine rotation * perspective (matching C++ finalH = rot * Hpersp)
    return rotationMatrix * perspectiveMatrix;
  }

  /// Basic perspective transform computation
  static Matrix4 _computePerspectiveTransform(List<Offset> src, List<Offset> dst) {
    // Simplified affine approximation for now
    // In a full implementation, this would solve the 8-parameter homography system
    final srcRect = _boundingRect(src);
    final dstRect = _boundingRect(dst);

    final scaleX = dstRect.width / srcRect.width;
    final scaleY = dstRect.height / srcRect.height;
    final translateX = dstRect.left - srcRect.left * scaleX;
    final translateY = dstRect.top - srcRect.top * scaleY;

    return Matrix4.identity()
      ..setEntry(0, 0, scaleX)
      ..setEntry(1, 1, scaleY)
      ..setEntry(0, 3, translateX)
      ..setEntry(1, 3, translateY);
  }

  /// Get bounding rectangle from a list of points
  static Rect _boundingRect(List<Offset> points) {
    if (points.isEmpty) return Rect.zero;

    double minX = points.first.dx;
    double maxX = points.first.dx;
    double minY = points.first.dy;
    double maxY = points.first.dy;

    for (final point in points) {
      minX = math.min(minX, point.dx);
      maxX = math.max(maxX, point.dx);
      minY = math.min(minY, point.dy);
      maxY = math.max(maxY, point.dy);
    }

    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  /// Apply perspective transformation to a point
  static Offset _applyPerspectiveTransform(Offset point, Matrix4 matrix) {
    final Vector4 homogeneousPoint = Vector4(point.dx, point.dy, 0, 1);
    final Vector4 transformed = matrix.transform(homogeneousPoint);

    // Convert back from homogeneous coordinates
    if (transformed.w != 0) {
      return Offset(transformed.x / transformed.w, transformed.y / transformed.w);
    }
    
    return Offset(transformed.x, transformed.y);
  }
}