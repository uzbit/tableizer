import 'dart:ui';

import 'package:flutter/foundation.dart';

/// A data class to hold the results of a table detection operation.
@immutable
class TableDetectionResult {
  const TableDetectionResult(this.points, this.imageSize);

  final List<Offset> points;
  final Size imageSize;
}