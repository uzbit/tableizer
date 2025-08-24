import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import '../detection_box.dart';
import '../services/ball_detection_service.dart';

class BallDetectionController extends ChangeNotifier {
  final BallDetectionService _ballDetectionService = BallDetectionService();
  static const double _confidenceThreshold = 0.6; // Match native CONF_THRESH from tableizer.cpp

  // Ball detection state
  List<Detection> _ballDetections = [];
  bool _isProcessingBalls = false;
  ui.Size? _capturedImageSize;

  // Getters
  List<Detection> get ballDetections => _ballDetections;
  bool get isProcessingBalls => _isProcessingBalls;
  ui.Size? get capturedImageSize => _capturedImageSize;

  Future<void> initialize() async {
    await _ballDetectionService.initialize();
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

      // Step 2: Convert to RGBA format (following test pattern)
      final img.Image rgbaImage = decodedImage.convert(numChannels: 4);
      final Uint8List rgbaBytes = rgbaImage.getBytes(order: img.ChannelOrder.rgba);

      // Step 3: Store image dimensions for coordinate transformation
      _capturedImageSize = ui.Size(
        rgbaImage.width.toDouble(),
        rgbaImage.height.toDouble(),
      );

      // Step 4: Run ball detection (following test pattern)
      final detections = await _ballDetectionService.detectFromByteBuffer(
        rgbaBytes,
        rgbaImage.width,
        rgbaImage.height,
      );

      // Filter detections by confidence (extra safety, native code should already filter)
      final filteredDetections = detections
          .where((detection) => detection.confidence >= _confidenceThreshold)
          .toList();

      _ballDetections = filteredDetections;
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
    _capturedImageSize = null;
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

  @override
  void dispose() {
    _ballDetectionService.dispose();
    super.dispose();
  }
}