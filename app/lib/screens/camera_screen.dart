import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart' hide BoxPainter;
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
  StreamSubscription<Map<String, dynamic>>? _tableDetectionsSubscription;
  List<Detection> _ballDetections = [];
  Map<String, dynamic> _tableDetections = {};
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
    super.dispose();
  }

  Future<void> _initializeServices() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
    _tableDetectionsSubscription =
        _tableDetectionService.detections.listen((detections) {
      print('Quad points received: ${detections['quad_points']}');
      setState(() {
        _tableDetections = detections;
      });
    });
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
      bgra8888: (frame) {
        // For debugging: convert the frame to a displayable image and show it.
        final image = img.Image.fromBytes(
          width: frame.width,
          height: frame.height,
          bytes: frame.planes.first.bytes.buffer,
          order: img.ChannelOrder.rgba, // Based on byte log, the order is RGBA
        );
        showFrameDebug(context, image);

        // Continue with the detection
        _tableDetectionService.detectTableFromByteBuffer(
          frame.planes.first.bytes,
          frame.width,
          frame.height,
        );
      },
      jpeg: (_) {},
      nv21: (_) {},
      yuv420: (_) {},
    );
  }

  void showFrameDebug(BuildContext context, img.Image rgba) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        content: Image.memory(
          Uint8List.fromList(img.encodePng(rgba)),
          gaplessPlayback: true,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final quadPoints = _tableDetections['quad_points'] as List<dynamic>?;
    final points = quadPoints
        ?.map((p) => Offset(p['x'].toDouble(), p['y'].toDouble()))
        .toList() ??
        [];

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
          androidOptions: const AndroidAnalysisOptions.bgra8888(width: 1080),
          maxFramesPerSecond: 30,
        ),
        builder: (cameraState, preview) {
          return Stack(
            fit: StackFit.expand,
            children: [
              if (_tableDetections.containsKey('image'))
                Image.memory(
                  base64Decode(_tableDetections['image']),
                  fit: BoxFit.fill,
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
