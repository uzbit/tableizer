import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:convert';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart' hide BoxPainter;
import 'package:image/image.dart' as img;
import '../services/table_detection_service.dart';
import '../services/ball_detection_service.dart';
import '../widgets/box_painter.dart';
import '../detection_box.dart';
import '../widgets/util_widgets.dart';
import 'table_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  final BallDetectionService _ballDetectionService = BallDetectionService();
  final TableDetectionService _tableDetectionService = TableDetectionService();
  List<Detection> _ballDetections = [];
  Map<String, dynamic> _tableDetections = {};
  bool _isProcessingFrame = false;
  double _fps = 0.0;
  int _frameCounter = 0;
  int _lastFrameTime = 0;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
  }

  Future<void> _processCameraImage(AnalysisImage image) async {
    if (_isProcessingFrame) {
      return;
    }
    _isProcessingFrame = true;

    final currentTime = DateTime.now().millisecondsSinceEpoch;
    if (_lastFrameTime != 0) {
      final int aSecondAgo = currentTime - 1000;
      _frameCounter++;
      if (_lastFrameTime < aSecondAgo) {
        _fps = _frameCounter / ((currentTime - _lastFrameTime) / 1000.0);
        _frameCounter = 0;
        _lastFrameTime = currentTime;
      }
    } else {
      _lastFrameTime = currentTime;
    }

    await _updateDetections(image);

    if (mounted) {
      setState(() {});
    }
    _isProcessingFrame = false;
  }

  Future<void> _updateDetections(AnalysisImage image) async {
    await image.when(
      bgra8888: (frame) async {
        final plane = frame.planes.first; // packed BGRA

        // 1) Wrap the incoming BGRA buffer respecting row stride.
        img.Image bgra = img.Image.fromBytes(
          width: frame.width,
          height: frame.height,
          bytes: plane.bytes.buffer,        // ByteBuffer (not Uint8List)
          rowStride: plane.bytesPerRow,     // respect padding
          numChannels: 4,
          order: img.ChannelOrder.bgra,
        );

        // 2) Rotate to portrait based on frame rotation.
        // camerawesome typically provides rotation in degrees (0/90/180/270).
        // final int rot = (frame.rotation ?? 90) % 360;
        // if (rot == 90) {
        //   bgra = img.copyRotate(bgra, angle: -90); // 90° CW
        // } else if (rot == 180) {
        //   bgra = img.copyRotate(bgra, angle: 180);
        // } else if (rot == 270) {
        //   bgra = img.copyRotate(bgra, angle: 90);  // 90° CCW
        // }

        // (Optional) Mirror if front camera delivers mirrored frames.
        // If your API exposes a boolean, plug it here; default false.
        // final bool mirrored = frame.isMirrored ?? false;
        // if (mirrored) {
        //   bgra = img.flipHorizontal(bgra);
        // }

        // 3) Convert BGRA -> RGBA efficiently (packed, no stride).
        final img.Image rgba = img.Image.fromBytes(
          width: bgra.width,
          height: bgra.height,
          bytes: bgra.getBytes(order: img.ChannelOrder.rgba).buffer,  // now tightly packed
          numChannels: 4,
          order: img.ChannelOrder.rgba,
        );

        // Debug preview — now upright and with correct colors.
        // showFrameDebug(context, rgba);

        // 4) (Optional) Send to native detector as tightly packed RGBA.
        _nativeDetect(rgba);
      },

      // Not used here, but keep handlers to satisfy the sealed union:
      jpeg: (_) async {},
      nv21: (_) async {},
      yuv420: (_) async {},
    );
  }

  Future<void> _nativeDetect(img.Image rgba) async {
    final tableDetections = await _tableDetectionService.detectTableFromRGBImage(rgba);

    if (tableDetections.isNotEmpty) {
      final ballDetections = await _ballDetectionService.detectFromRGBImage(rgba);
      if (mounted) {
        setState(() {
          _tableDetections = tableDetections;
          _ballDetections = ballDetections;
        });
      }
    }
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
