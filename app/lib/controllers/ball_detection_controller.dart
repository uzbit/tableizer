import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import '../models/ball_detection_result.dart';
import '../services/ball_detection_service.dart';
import '../services/table_detection_service.dart';
import '../models/table_detection_result.dart';

class BallDetectionController extends ChangeNotifier {
  final BallDetectionService _ballDetectionService = BallDetectionService();
  final TableDetectionService _tableDetectionService = TableDetectionService();
  static const double _confidenceThreshold = 0.6; // Match native CONF_THRESH from tableizer.cpp

  // Ball detection state
  List<BallDetectionResult> _ballDetections = [];
  bool _isProcessingBalls = false;
  ui.Size? _capturedImageSize;
  
  // Table detection state
  TableDetectionResult? _tableDetectionResult;

  // Getters
  List<BallDetectionResult> get ballDetections => _ballDetections;
  bool get isProcessingBalls => _isProcessingBalls;
  ui.Size? get capturedImageSize => _capturedImageSize;
  TableDetectionResult? get tableDetectionResult => _tableDetectionResult;

  Future<void> initialize() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
  }

  Future<void> processBallDetection(Uint8List imageBytes) async {
    if (_isProcessingBalls) return;
    
    _isProcessingBalls = true;
    _ballDetections.clear();
    notifyListeners();

    try {
      // Step 1: Decode JPEG bytes to Image
      final img.Image? decodedImage = img.decodeImage(imageBytes);
      if (decodedImage == null) {
        throw Exception('Failed to decode captured image');
      }

      // Step 2: Convert to BGRA format (for C++ BGRA functions)
      final img.Image bgraImage = decodedImage.convert(numChannels: 4);
      final Uint8List bgraBytes = bgraImage.getBytes(order: img.ChannelOrder.bgra);

      // Step 3: Store image dimensions for coordinate transformation
      _capturedImageSize = ui.Size(
        bgraImage.width.toDouble(),
        bgraImage.height.toDouble(),
      );

      // Step 4: Run ball detection (following test pattern)
      final detections = await _ballDetectionService.detectBallsFromBytes(
        bgraBytes,
        bgraImage.width,
        bgraImage.height,
      );

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

  Future<void> processTableDetection(Uint8List imageBytes) async {
    try {
      // Step 1: Decode JPEG bytes to Image
      final img.Image? decodedImage = img.decodeImage(imageBytes);
      if (decodedImage == null) {
        throw Exception('Failed to decode captured image');
      }

      // Step 2: Convert to BGRA format (for C++ BGRA functions)
      final img.Image bgraImage = decodedImage.convert(numChannels: 4);
      final Uint8List bgraBytes = bgraImage.getBytes(order: img.ChannelOrder.bgra);

      // Step 3: Run table detection on the image
      final tableResult = await _tableDetectionService.detectTableFromBytes(
        bgraBytes,
        bgraImage.width,
        bgraImage.height,
      );

      _tableDetectionResult = tableResult;
      notifyListeners();

      if (tableResult != null) {
        print('Table detection completed: ${tableResult.points.length} quad points found');
      } else {
        print('Table detection failed');
      }
    } catch (e) {
      print('Table detection failed: $e');
      notifyListeners();
    }
  }

  void clearDetections() {
    _ballDetections.clear();
    _capturedImageSize = null;
    _tableDetectionResult = null;
    notifyListeners();
  }

  String buildCaptureStatusText(Uint8List? capturedImageBytes) {
    if (capturedImageBytes == null) {
      return 'Image Captured!\n(Ready for ball detection)';
    }
    
    final sizeKB = (capturedImageBytes.length / 1024).toStringAsFixed(1);
    
    if (_isProcessingBalls) {
      return 'Processing Image...\n(${sizeKB}KB - Detecting balls)';
    } else if (_ballDetections.isNotEmpty) {
      return 'Detection Complete!\n(${_ballDetections.length} balls found - ${sizeKB}KB)';
    } else {
      return 'Image Captured!\n(${sizeKB}KB - Ready for analysis)';
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
    _tableDetectionService.dispose();
    super.dispose();
  }
}