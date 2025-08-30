import 'dart:convert';
import 'dart:ui';

class BallDetectionResult {
  final int classId;
  final double confidence;
  final double centerX;
  final double centerY;
  final BoundingBox box;

  BallDetectionResult({
    required this.classId,
    required this.confidence,
    required this.centerX,
    required this.centerY,
    required this.box,
  });

  factory BallDetectionResult.fromJson(Map<String, dynamic> json) {
    return BallDetectionResult(
      classId: json['class_id'],
      confidence: json['confidence'],
      centerX: json['center_x'],
      centerY: json['center_y'],
      box: BoundingBox.fromJson(json['box']),
    );
  }
}

class BoundingBox {
  final int x;
  final int y;
  final int width;
  final int height;

  BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x: json['x'],
      y: json['y'],
      width: json['width'],
      height: json['height'],
    );
  }

  Rect toRect() {
    return Rect.fromLTWH(
      x.toDouble(),
      y.toDouble(),
      width.toDouble(),
      height.toDouble(),
    );
  }
}

class BallDetectionResults {
  final List<BallDetectionResult> detections;

  BallDetectionResults({required this.detections});

  factory BallDetectionResults.fromJson(String jsonString) {
    final parsed = json.decode(jsonString);
    var list = parsed['detections'] as List;
    List<BallDetectionResult> detectionsList =
        list.map((i) => BallDetectionResult.fromJson(i)).toList();
    return BallDetectionResults(detections: detectionsList);
  }
}
