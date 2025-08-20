import 'dart:async';
import 'dart:ui' as ui;

import 'package:app/services/table_detection_result.dart';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart' hide BoxPainter;
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
  StreamSubscription<TableDetectionResult>? _tableDetectionsSubscription;
  List<Detection> _ballDetections = [];
  List<Offset> _quadPoints = [];
  ui.Size? _imageSize;
  double _fps = 0.0;
  int _frameCounter = 0;
  DateTime _lastFrameTime = DateTime.now();

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  @override
  void dispose() {
    _tableDetectionsSubscription?.cancel();
    _tableDetectionService.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
    _lastFrameTime = DateTime.now();

    _tableDetectionsSubscription =
        _tableDetectionService.detections.listen((result) {
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
        _quadPoints = result.points;
        _imageSize = result.imageSize;
        _fps = newFps;
      });
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
              // Use a LayoutBuilder to get the exact size of the preview area.
              if (_imageSize != null)
                LayoutBuilder(builder: (context, constraints) {
                  final screenSize = constraints.biggest;
                  final imageSize = _imageSize!;

                  // Calculate the scale and offset to mimic BoxFit.cover
                  final double imageAspectRatio =
                      imageSize.width / imageSize.height;
                  final double screenAspectRatio =
                      screenSize.width / screenSize.height;

                  double scale;
                  if (screenAspectRatio > imageAspectRatio) {
                    scale = screenSize.width / imageSize.width;
                  } else {
                    scale = screenSize.height / imageSize.height;
                  }

                  final double scaledWidth = imageSize.width * scale;
                  final double scaledHeight = imageSize.height * scale;
                  final double dx = (screenSize.width - scaledWidth) / 2.0;
                  final double dy = (screenSize.height - scaledHeight) / 2.0;

                  // Transform the points from image coordinates to screen coordinates
                  final scaledPoints = _quadPoints.map((p) {
                    return Offset(p.dx * scale + dx, p.dy * scale + dy);
                  }).toList();

                  return CustomPaint(
                    size: screenSize,
                    painter: TablePainter(quadPoints: scaledPoints),
                  );
                }),
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