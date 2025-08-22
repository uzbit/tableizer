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
    //await _ballDetectionService.initialize();
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
        // Use CONTAIN to ensure the preview is not cropped.
        previewFit: CameraPreviewFit.contain,
        saveConfig: SaveConfig.photoAndVideo(),
        sensorConfig: SensorConfig.single(
          sensor: Sensor.position(SensorPosition.back),
          flashMode: FlashMode.none,
          // IMPORTANT: Match the analysis image aspect ratio
          aspectRatio: CameraAspectRatios.ratio_4_3,
        ),
        onImageForAnalysis: _tableDetectionService.processImage,
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: const AndroidAnalysisOptions.bgra8888(width: 1280),
          maxFramesPerSecond: 30,
        ),
        builder: (cameraState, preview) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // The preview is rendered automatically by the builder.
              if (_imageSize != null)
                LayoutBuilder(builder: (context, constraints) {
                  final previewSize = constraints.biggest;
                  final imageSize = _imageSize!;

                  final imageAspectRatio = imageSize.width / imageSize.height;
                  final previewAspectRatio = previewSize.width / previewSize.height;

                  double scale;
                  // This logic mimics BoxFit.contain.
                  if (previewAspectRatio > imageAspectRatio) {
                    // Preview is wider than the image -> letterbox
                    scale = previewSize.height / imageSize.height;
                  } else {
                    // Preview is taller than the image -> pillarbox
                    scale = previewSize.width / imageSize.width;
                  }

                  final scaledWidth = imageSize.width * scale;
                  final scaledHeight = imageSize.height * scale;

                  // Center the scaled image within the preview.
                  final dx = (previewSize.width - scaledWidth) / 2.0;
                  final dy = (previewSize.height - scaledHeight) / 2.0;

                  // Transform points from image-space to screen-space.
                  final scaledPoints = _quadPoints.map((p) {
                    return Offset(p.dx * scale + dx, p.dy * scale + dy);
                  }).toList();

                  return CustomPaint(
                    size: previewSize,
                    painter: TablePainter(quadPoints: scaledPoints),
                  );
                }),

              // --- UI Overlays (FPS, etc.) ---
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
