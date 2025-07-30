import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart' hide BoxPainter;
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import '../detection_box.dart';
import '../services/detection_service.dart';
import '../widgets/box_painter.dart';

class DisplayPictureScreen extends StatefulWidget {
  final DetectionService detectionService;

  const DisplayPictureScreen({super.key, required this.detectionService});

  @override
  State<DisplayPictureScreen> createState() => _DisplayPictureScreenState();
}

class _DisplayPictureScreenState extends State<DisplayPictureScreen> {
  List<Detection> _detections = [];
  Uint8List? _imageBytes;
  ui.Image? _image;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final byteData = await rootBundle.load('assets/images/P_20250718_203819.jpg');
    final imageBytes = byteData.buffer.asUint8List();
    final image = await decodeImageFromList(imageBytes);
    setState(() {
      _imageBytes = imageBytes;
      _image = image;
    });
  }

  Future<void> _processImage() async {
    if (_imageBytes == null) return;

    final detections = await widget.detectionService.detectFromBytes(_imageBytes!);
    setState(() {
      _detections = detections;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Confirm Detection')),
      body: Column(
        children: [
          Expanded(
            child: _image != null
                ? Stack(
                    children: [
                      Image.memory(_imageBytes!),
                      CustomPaint(
                        painter: BoxPainter(
                          detections: _detections,
                          cameraPreviewSize: ui.Size(
                            _image!.width.toDouble(),
                            _image!.height.toDouble(),
                          ),
                          screenSize: MediaQuery.of(context).size,
                        ),
                      ),
                    ],
                  )
                : const Center(child: CircularProgressIndicator()),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed: _processImage,
              child: const Text('Detect'),
            ),
          ),
        ],
      ),
    );
  }
}
