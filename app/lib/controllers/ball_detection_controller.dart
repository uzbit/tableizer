import 'dart:typed_data';

import 'package:flutter/material.dart';
import '../models/ball_detection_result.dart';
import '../services/ball_detection_service.dart';
import '../models/table_detection_result.dart';

class BallDetectionController extends ChangeNotifier {
  final BallDetectionService _ballDetectionService = BallDetectionService();
  static const double _confidenceThreshold = 0.6; // Match native CONF_THRESH from tableizer.cpp

  // Ball detection state
  List<BallDetectionResult> _ballDetections = [];
  bool _isProcessingBalls = false;

  // Getters
  List<BallDetectionResult> get ballDetections => _ballDetections;
  bool get isProcessingBalls => _isProcessingBalls;

  Future<void> initialize() async {
    await _ballDetectionService.initialize();
  }

  Future<void> processBallDetection(
    Uint8List bgraBytes,
    int width,
    int height,
    TableDetectionResult? tableDetectionResult,
    int rotationDegrees,
  ) async {
    if (_isProcessingBalls) return;

    _isProcessingBalls = true;
    _ballDetections.clear();
    notifyListeners();

    try {
      print('====================================');
      print('[BALL_DETECTION] BGRA frame: ${width}x$height (${bgraBytes.length} bytes)');
      print('[BALL_DETECTION] First 16 bytes: ${bgraBytes.sublist(0, 16)}');
      print('[BALL_DETECTION] Table detected: ${tableDetectionResult != null}');
      print('[BALL_DETECTION] Quad points available: ${tableDetectionResult?.points != null}');

      // Transform quad points to canvas coordinates if available
      List<Offset>? canvasQuadPoints;
      if (tableDetectionResult?.points != null) {
        // Table result contains points in original image coordinates
        // We need to transform them to canvas coordinates for ball detection
        canvasQuadPoints = tableDetectionResult!.points.map((point) {
          return Offset(
            point.dx + tableDetectionResult.canvasOffsetX,
            point.dy + tableDetectionResult.canvasOffsetY,
          );
        }).toList();
        print('[BALL_DETECTION] Original quad points: ${tableDetectionResult.points}');
        print('[BALL_DETECTION] Canvas quad points: $canvasQuadPoints');
        print('[BALL_DETECTION] Canvas offset: (${tableDetectionResult.canvasOffsetX}, ${tableDetectionResult.canvasOffsetY})');
      }
      print('====================================');

      // Run ball detection - use pre-normalized buffer if available
      List<BallDetectionResult> detections;
      if (tableDetectionResult?.normalizedBytes != null &&
          tableDetectionResult!.normalizedWidth > 0 &&
          tableDetectionResult.normalizedHeight > 0) {
        print('[BALL_DETECTION] ✓ Using pre-normalized buffer from table detection (no re-normalization)');
        print('[BALL_DETECTION] Normalized size: ${tableDetectionResult.normalizedWidth}x${tableDetectionResult.normalizedHeight}');
        detections = await _ballDetectionService.detectBallsFromNormalizedBytes(
          tableDetectionResult.normalizedBytes!,
          tableDetectionResult.normalizedWidth,
          tableDetectionResult.normalizedHeight,
          tableDetectionResult.normalizedStride,
          quadPoints: canvasQuadPoints,
        );
      } else {
        print('[BALL_DETECTION] ⚠ No pre-normalized buffer available, normalizing image...');
        detections = await _ballDetectionService.detectBallsFromBytes(
          bgraBytes,
          width,
          height,
          quadPoints: canvasQuadPoints,
          rotationDegrees: rotationDegrees,
        );
      }

      // Filter detections by confidence (extra safety, native code should already filter)
      final filteredDetections = detections
          .where((detection) => detection.confidence >= _confidenceThreshold)
          .toList();

      // Apply ball class corrections
      final correctedDetections = _applyBallClassCorrections(filteredDetections);

      _ballDetections = correctedDetections;
      _isProcessingBalls = false;
      notifyListeners();

      print('Ball detection completed: ${filteredDetections.length} balls detected (${detections.length} total, filtered by confidence >= $_confidenceThreshold)');
    } catch (e) {
      print('Ball detection failed: $e');
      _isProcessingBalls = false;
      notifyListeners();
    }
  }

  void clearDetections() {
    _ballDetections.clear();
    notifyListeners();
  }

  String buildCaptureStatusText(Uint8List? capturedImageBytes) {
    if (capturedImageBytes == null) {
      return 'Frame Captured!\n(Ready for ball detection)';
    }

    final sizeKB = (capturedImageBytes.length / 1024).toStringAsFixed(1);

    if (_isProcessingBalls) {
      return 'Processing Frame...\n(${sizeKB}KB - Detecting balls)';
    } else if (_ballDetections.isNotEmpty) {
      return 'Detection Complete!\n(${_ballDetections.length} balls found - ${sizeKB}KB)';
    } else {
      return 'Frame Captured!\n(${sizeKB}KB - Ready for analysis)';
    }
  }

  /// Apply ball class corrections:
  /// - Keep only highest confidence cue ball (classId 1), convert others to stripes (classId 3)
  /// - Keep only highest confidence black ball (classId 0), convert others to solids (classId 2)
  List<BallDetectionResult> _applyBallClassCorrections(List<BallDetectionResult> detections) {
    final correctedDetections = <BallDetectionResult>[];
    
    // Find highest confidence cue ball (classId 1)
    BallDetectionResult? bestCue;
    final cueDetections = detections.where((d) => d.classId == 1).toList();
    if (cueDetections.isNotEmpty) {
      bestCue = cueDetections.reduce((a, b) => a.confidence > b.confidence ? a : b);
    }
    
    // Find highest confidence black ball (classId 0)  
    BallDetectionResult? bestBlack;
    final blackDetections = detections.where((d) => d.classId == 0).toList();
    if (blackDetections.isNotEmpty) {
      bestBlack = blackDetections.reduce((a, b) => a.confidence > b.confidence ? a : b);
    }
    
    // Process all detections
    for (final detection in detections) {
      if (detection.classId == 1) { // Cue ball
        if (detection == bestCue) {
          correctedDetections.add(detection); // Keep best cue
        } else {
          // Convert other cues to stripes
          correctedDetections.add(BallDetectionResult(
            centerX: detection.centerX,
            centerY: detection.centerY,
            box: detection.box,
            confidence: detection.confidence,
            classId: 3, // Change to stripe
          ));
        }
      } else if (detection.classId == 0) { // Black ball
        if (detection == bestBlack) {
          correctedDetections.add(detection); // Keep best black
        } else {
          // Convert other blacks to solids
          correctedDetections.add(BallDetectionResult(
            centerX: detection.centerX,
            centerY: detection.centerY,
            box: detection.box,
            confidence: detection.confidence,
            classId: 2, // Change to solid
          ));
        }
      } else {
        // Keep other ball types unchanged
        correctedDetections.add(detection);
      }
    }
    
    return correctedDetections;
  }

  @override
  void dispose() {
    _ballDetectionService.dispose();
    super.dispose();
  }
}