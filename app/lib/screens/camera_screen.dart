import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart' hide BoxPainter;
import 'package:image/image.dart' as img;
import '../services/detection_service.dart';
import '../utils/image_converter.dart';
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
      // _drawBoxes(screenSize).whenComplete(() => _isProcessingFrame = false);
    });
  }

  Future<void> _updateDetections(CameraImage image) async {
    // Convert the YUV CameraImage ➜ RGBA `img.Image`
    final img.Image rgbaFrame = convertCameraImage(image);

    // Optional debug
    print('Frame: ${rgbaFrame.width}×${rgbaFrame.height} '
        '(${rgbaFrame.lengthInBytes} B)');

    // Run native inference on the decoded frame
    _detections = await _detectionService.detectFromImage(rgbaFrame);

    print('Found ${_detections.length} balls!');
  }

  // Future<void> _drawBoxes(Size screenSize) async {
  //   if (_latestImage == null) return;
  //   final image = await convertCameraImageToUiImage(_latestImage!);
  //   final recorder = ui.PictureRecorder();
  //   final canvas = Canvas(recorder);
  //   final painter = BoxPainter(
  //     detections: _detections,
  //     imageSize: Size(image.width.toDouble(), image.height.toDouble()),
  //     screenSize: screenSize,
  //   );
  //   painter.paint(canvas, Size(image.width.toDouble(), image.height.toDouble()));
  //   final picture = recorder.endRecording();
  //   final newImageWithBoxes = await picture.toImage(image.width, image.height);
  //   if (mounted) {
  //     setState(() {
  //       _imageWithBoxes = newImageWithBoxes;
  //     });
  //   }
  // }

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
