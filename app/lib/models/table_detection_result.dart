import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart';

/// A data class to hold the results of a table detection operation.
@immutable
class TableDetectionResult {
  const TableDetectionResult(
    this.points,
    this.imageSize, {
    this.maskBytes,
    this.canvasOffsetX = 0,
    this.canvasOffsetY = 0,
    this.originalImageSize,
    this.normalizedBytes,
    this.normalizedWidth = 0,
    this.normalizedHeight = 0,
    this.normalizedStride = 0,
    this.orientation,
  });

  final List<Offset> points; // Points in original image coordinates (not canvas)
  final Size imageSize; // Canvas size (16:9)
  final Uint8List? maskBytes;
  final int canvasOffsetX; // Offset where original image starts in canvas
  final int canvasOffsetY; // Offset where original image starts in canvas
  final Size? originalImageSize; // Original image size before canvas padding

  // Normalized image buffer (for reuse in ball detection)
  final Uint8List? normalizedBytes; // The normalized BGRA buffer from C++
  final int normalizedWidth; // Width of normalized image
  final int normalizedHeight; // Height of normalized image
  final int normalizedStride; // Stride of normalized image

  // Table orientation from quad analysis
  final String? orientation; // SHORT_SIDE, LONG_SIDE, TOP_DOWN, or OTHER
}