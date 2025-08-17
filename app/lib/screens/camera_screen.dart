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
  StreamSubscription<TableDetection>? _tableDetectionsSubscription;
  List<Detection> _ballDetections = [];
  TableDetection? _tableDetection;
  ui.Image? _debugUiImage; // Holds the decoded debug image
  double _fps = 0.0;
  int _frameCounter = 0;
  int _lastFrameTime = 0;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  @override
  void dispose() {
    _tableDetectionsSubscription?.cancel();
    _tableDetectionService.dispose();
    _debugUiImage?.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
    _tableDetectionsSubscription =
        _tableDetectionService.detections.listen((detection) {
      if (mounted) {
        setState(() {
          _tableDetection = detection;
        });
        // Asynchronously update the debug image
        _updateDebugImage(detection);
      }
    });
  }

  // New async method to decode raw RGBA bytes
  Future<void> _updateDebugImage(TableDetection detection) async {
    if (detection.debugImage == null) return;

    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      detection.debugImage!,
      detection.imageWidth,
      detection.imageHeight,
      ui.PixelFormat.rgba8888,
      (ui.Image img) {
        completer.complete(img);
      },
    );

    final newImage = await completer.future;
    if (mounted) {
      setState(() {
        _debugUiImage?.dispose(); // Dispose the old image
        _debugUiImage = newImage;
      });
    } else {
      // If the widget is disposed while we were decoding, dispose the new image
      newImage.dispose();
    }
  }

  Future<void> _processCameraImage(AnalysisImage image) async {
    final currentTime = DateTime.now().millisecondsSinceEpoch;
    if (_lastFrameTime != 0) {
      final int aSecondAgo = currentTime - 1000;
      _frameCounter++;
      if (_lastFrameTime < aSecondAgo) {
        if (mounted) {
          setState(() {
            _fps = _frameCounter / ((currentTime - _lastFrameTime) / 1000.0);
          });
        }
        _frameCounter = 0;
        _lastFrameTime = currentTime;
      }
    } else {
      _lastFrameTime = currentTime;
    }

    image.when(
      bgra8888: (frame) {},
      jpeg: (frame) {
        final image = img.decodeJpg(frame.bytes);
        if (image != null) {
          _tableDetectionService.processImage(image);
        }
      },
      nv21: (_) {},
      yuv420: (frame) {},
    );
  }

  @override
  Widget build(BuildContext context) {
    final points = _tableDetection?.quadPoints ?? [];

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
        onImageForAnalysis: _processCameraImage,
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: const AndroidAnalysisOptions.jpeg(width: 1080),
          maxFramesPerSecond: 30,
        ),
        builder: (cameraState, preview) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // Use RawImage to display the decoded ui.Image
              if (_debugUiImage != null)
                FittedBox(
                  fit: BoxFit.cover,
                  child: SizedBox(
                    width: _debugUiImage!.width.toDouble(),
                    height: _debugUiImage!.height.toDouble(),
                    child: RawImage(
                      image: _debugUiImage,
                    ),
                  ),
                ),
              CustomPaint(
                painter: BoxPainter(
                  sensorSize: ui.Size(preview.rect.width, preview.rect.height),
                  detections: _ballDetections,
                ),
              ),
              CustomPaint(
                painter: TablePainter(
                  sensorSize: ui.Size(preview.rect.width, preview.rect.height),
                  quadPoints: points,
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
                    'Points: ${points.toString()}',
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
