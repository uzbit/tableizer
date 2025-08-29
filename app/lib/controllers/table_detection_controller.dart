import 'dart:async';
import 'dart:ui' as ui;

import 'package:app/services/table_detection_result.dart';
import 'package:app/services/table_detection_service.dart';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';

class TableDetectionController extends ChangeNotifier {
  final TableDetectionService _tableDetectionService = TableDetectionService();
  StreamSubscription<TableDetectionResult>? _tableDetectionsSubscription;
  
  // Table detection state
  List<Offset> _quadPoints = [];
  List<Offset> _filteredQuadPoints = [];  // Alpha-filtered smooth quad points
  ui.Size? _imageSize;
  double _fps = 0.0;
  int _frameCounter = 0;
  DateTime _lastFrameTime = DateTime.now();
  bool _isEnabled = true;  // Control whether table detection is active
  
  // Alpha filter settings for quad point smoothing
  static const double _quadAlpha = 0.3;  // 30% new, 70% previous (smooth)

  // Getters
  List<Offset> get quadPoints => _filteredQuadPoints.isNotEmpty ? _filteredQuadPoints : _quadPoints;
  ui.Size? get imageSize => _imageSize;
  double get fps => _fps;
  bool get isEnabled => _isEnabled;

  Future<void> initialize() async {
    await _tableDetectionService.initialize();
    _lastFrameTime = DateTime.now();

    _tableDetectionsSubscription =
        _tableDetectionService.detections.listen((result) {
      _frameCounter++;
      final now = DateTime.now();
      final difference = now.difference(_lastFrameTime);
      double newFps = _fps;

      if (difference.inSeconds >= 1) {
        newFps = _frameCounter / difference.inSeconds;
        _frameCounter = 0;
        _lastFrameTime = now;
      }

      _quadPoints = result.points;
      _imageSize = result.imageSize;
      _fps = newFps;
      
      // Apply alpha filter to quad points for smooth display
      _applyQuadPointFiltering(result.points);
      
      notifyListeners();
    });
  }

  Future<void> processImage(AnalysisImage image) async {
    // Skip processing if table detection is disabled
    if (!_isEnabled) return;
    
    await _tableDetectionService.processImage(image);
  }
  
  // Enable/disable table detection
  void setEnabled(bool enabled) {
    if (_isEnabled != enabled) {
      _isEnabled = enabled;
      notifyListeners();
    }
  }
  
  // Apply alpha filter to smooth quad point jitter
  void _applyQuadPointFiltering(List<Offset> newPoints) {
    // Initialize filtered points if empty or size mismatch
    if (_filteredQuadPoints.isEmpty || _filteredQuadPoints.length != newPoints.length) {
      _filteredQuadPoints = List<Offset>.from(newPoints);
      return;
    }
    
    // Apply alpha filter: filtered = α * new + (1-α) * previous
    for (int i = 0; i < newPoints.length; i++) {
      final newPoint = newPoints[i];
      final prevPoint = _filteredQuadPoints[i];
      
      final filteredX = _quadAlpha * newPoint.dx + (1 - _quadAlpha) * prevPoint.dx;
      final filteredY = _quadAlpha * newPoint.dy + (1 - _quadAlpha) * prevPoint.dy;
      
      _filteredQuadPoints[i] = Offset(filteredX, filteredY);
    }
  }

  @override
  void dispose() {
    _tableDetectionsSubscription?.cancel();
    _tableDetectionService.dispose();
    super.dispose();
  }
}