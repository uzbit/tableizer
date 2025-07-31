import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart' hide BoxPainter;
import 'package:image/image.dart' as img;
import '../services/detection_service.dart';
import '../utils/image_converter.dart';
import '../widgets/box_painter.dart';
import '../detection_box.dart';
import '../widgets/util_widgets.dart';
import 'display_picture_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.camera});
  final CameraDescription camera;

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  Future<void>? _initializeControllerFuture;
  final DetectionService _detectionService = DetectionService();
  List<Detection> _detections = [];
  bool _isProcessingFrame = false;
  ui.Size _imageSize = ui.Size(0, 0);

  @override
  void initState() {
    super.initState();
    _initializeControllerFuture = _initializeEverything();
  }

  Future<void> _initializeEverything() async {
    _controller = CameraController(widget.camera, ResolutionPreset.high, enableAudio: false);
    await _detectionService.initialize();
    await _controller.initialize();
    if (mounted) {
      await _controller.startImageStream(_processCameraImage);
    }
  }

  void _processCameraImage(CameraImage image) {
    if (_isProcessingFrame) {
      return;
    }
    _isProcessingFrame = true;
    _imageSize = ui.Size(image.width.toDouble(), image.height.toDouble());
    print("Image size: ${_imageSize}");
    _updateDetections(image).then((_) {
      _isProcessingFrame = false;
    });
  }

  Future<void> _updateDetections(CameraImage image) async {
    // Run native inference on the YUV planes
    img.Image rgba = convertCameraImage(image);
    // ── Optional: rotate to match device orientation ────────────────────
    final int deg = _controller.description.sensorOrientation; // 90, 180, 270
    if (deg != 0) {
      rgba = img.copyRotate(rgba, angle: deg);
    }
    //showFrameDebug(context, rgba);

    _detections = await _detectionService.detectFromRGBImage(rgba);

    //_detections = await _detectionService.detectFromYUV(image);

    if (mounted) {
      setState(() {});
    }

    print('Found ${_detections.length} balls!');
  }

  @override
  void dispose() {
    if (_controller.value.isStreamingImages) {
      _controller.stopImageStream();
    }
    _controller.dispose();
    _detectionService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tableizer')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            if (snapshot.hasError) {
              return Center(
                  child: Text('Initialization Error: ${snapshot.error}'));
            }
            return Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_controller),
                CustomPaint(
                  painter: BoxPainter(
                    detections: _detections,
                    imageSize: _imageSize,
                    screenSize: MediaQuery.of(context).size,
                  ),
                ),
              ],
            );
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (context) => DisplayPictureScreen(
                detectionService: _detectionService,
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
