import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart' hide BoxPainter;
import '../services/detection_service.dart';
import '../widgets/box_painter.dart';
import '../detection_box.dart';
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

  @override
  void initState() {
    super.initState();
    _initializeControllerFuture = _initializeEverything();
  }

  Future<void> _initializeEverything() async {
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    await _detectionService.initialize();
    await _controller.initialize();
    if (mounted) {
      await _controller.startImageStream(_processCameraImage);
    }
  }

  void _processCameraImage(CameraImage image) async {
    final detections = await _detectionService.detect(image);
    if (mounted) {
      setState(() {
        _detections = detections;
      });
    }
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
                    cameraPreviewSize:
                        _controller.value.previewSize ?? ui.Size(0, 0),
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
