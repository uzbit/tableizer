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
  ui.Image? _imageWithBoxes;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  Future<void> _loadImage() async {
    final byteData =
        await rootBundle.load('assets/images/P_20250718_203819.jpg');
    final imageBytes = byteData.buffer.asUint8List();
    final image = await decodeImageFromList(imageBytes);
    setState(() {
      _imageBytes = imageBytes;
      _image = image;
    });
  }

  Future<void> _processImage() async {
    if (_imageBytes == null) return;

    final img.Image? image = img.decodeImage(_imageBytes!);
    if (image == null) return;

    final detections = await widget.detectionService.detectFromRGBImage(image);
    setState(() {
      _detections = detections;
    });
    await _drawBoxes();
  }

  Future<void> _drawBoxes() async {
    if (_image == null) return;
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final painter = BoxPainter(
      detections: _detections,
      imageSize:
          Size(_image!.width.toDouble(), _image!.height.toDouble()),
      screenSize: MediaQuery.of(context).size,
    );
    canvas.drawImage(_image!, Offset.zero, Paint());
    painter.paint(
        canvas, Size(_image!.width.toDouble(), _image!.height.toDouble()));
    final picture = recorder.endRecording();
    final newImageWithBoxes =
        await picture.toImage(_image!.width, _image!.height);
    if (mounted) {
      setState(() {
        _imageWithBoxes = newImageWithBoxes;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Confirm Detection')),
      body: Column(
        children: [
          Expanded(
            child: _image != null
                ? (_imageWithBoxes != null
                    ? FittedBox(
                        fit: BoxFit.contain,
                        child: SizedBox(
                          width: _imageWithBoxes!.width.toDouble(),
                          height: _imageWithBoxes!.height.toDouble(),
                          child: RawImage(image: _imageWithBoxes!),
                        ),
                      )
                    : Image.memory(_imageBytes!))
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
