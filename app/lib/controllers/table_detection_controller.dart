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
  ui.Size? _imageSize;
  double _fps = 0.0;
  int _frameCounter = 0;
  DateTime _lastFrameTime = DateTime.now();

  // Getters
  List<Offset> get quadPoints => _quadPoints;
  ui.Size? get imageSize => _imageSize;
  double get fps => _fps;

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
      notifyListeners();
    });
  }

  Future<void> processImage(AnalysisImage image) async {
    await _tableDetectionService.processImage(image);
  }

  @override
  void dispose() {
    _tableDetectionsSubscription?.cancel();
    _tableDetectionService.dispose();
    super.dispose();
  }
}