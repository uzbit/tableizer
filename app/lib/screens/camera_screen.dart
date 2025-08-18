import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart' hide BoxPainter;
import 'package:image/image.dart' as img;
import 'package:image/image.dart' as img;
import '../services/ball_detection_service.dart';
import '../services/table_detection_service.dart';

import '../detection_box.dart';
import '../widgets/box_painter.dart';
import '../widgets/table_painter.dart';
import 'table_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  final BallDetectionService _ballDetectionService = BallDetectionService();
  final TableDetectionService _tableDetectionService = TableDetectionService();
  StreamSubscription<List<Offset>>? _tableDetectionsSubscription;
  StreamSubscription<Uint8List>? _debugImageSubscription;
  List<Detection> _ballDetections = [];
  List<Offset> _quadPoints = [];
  double _fps = 0.0;
  int _frameCounter = 0;
  DateTime _lastFrameTime = DateTime.now();
  bool _isDialogShowing = false;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  @override
  void dispose() {
    _tableDetectionsSubscription?.cancel();
    _debugImageSubscription?.cancel();
    _tableDetectionService.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
    _lastFrameTime = DateTime.now();

    _tableDetectionsSubscription =
        _tableDetectionService.detections.listen((points) {
      if (!mounted) return;

      _frameCounter++;
      final now = DateTime.now();
      final difference = now.difference(_lastFrameTime);
      double newFps = _fps;

      if (difference.inSeconds >= 1) {
        newFps = _frameCounter / difference.inSeconds;
        _frameCounter = 0;
        _lastFrameTime = now;
      }

      setState(() {
        _quadPoints = points;
        _fps = newFps;
      });
    });

    _debugImageSubscription =
        _tableDetectionService.debugImages.listen((imageBytes) {
      if (!mounted || _isDialogShowing) return;
      _isDialogShowing = true;
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          content: Image.memory(imageBytes),
          actions: <Widget>[
            TextButton(
              child: const Text('Close'),
              onPressed: () {
                Navigator.of(context).pop();
              },
            ),
          ],
        ),
      ).then((_) => _isDialogShowing = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tableizer')),
      body: CameraAwesomeBuilder.custom(
        previewFit: CameraPreviewFit.cover,
        saveConfig: SaveConfig.photoAndVideo(),
        sensorConfig: SensorConfig.single(
          sensor: Sensor.position(SensorPosition.back),
          flashMode: FlashMode.none,
          aspectRatio: CameraAspectRatios.ratio_16_9,
        ),
        onImageForAnalysis: _tableDetectionService.processImage,
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: const AndroidAnalysisOptions.bgra8888(width: 1080),
          maxFramesPerSecond: 30,
        ),
        builder: (cameraState, preview) {
          return Stack(
            fit: StackFit.expand,
            children: [
              CustomPaint(
                painter: BoxPainter(
                  sensorSize: ui.Size(preview.rect.width, preview.rect.height),
                  detections: _ballDetections,
                ),
              ),
              CustomPaint(
                painter: TablePainter(
                  sensorSize: ui.Size(preview.rect.width, preview.rect.height),
                  quadPoints: _quadPoints,
                ),
              ),
              Align(
                alignment: Alignment.topLeft,
                child: Container(
                  color: Colors.black.withOpacity(0.5),
                  child: Text(
                    'FPS: ${_fps.toStringAsFixed(1)}',
                    style: const TextStyle(color: Colors.white, fontSize: 20),
                  ),
                ),
              ),
              Align(
                alignment: Alignment.bottomLeft,
                child: Container(
                  color: Colors.black.withOpacity(0.5),
                  child: Text(
                    'Points: ${_quadPoints.toString()}',
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ),
              ),
            ],
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => TableScreen(
                tableDetectionService: _tableDetectionService,
              ),
            ),
          );
        },
        tooltip: 'Process Local Image',
        child: const Icon(Icons.image),
      ),
    );
  }
}
