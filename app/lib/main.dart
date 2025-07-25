import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Entry-point. Ensure cameras are fetched *before* runApp so that the
/// [CameraController] can be constructed synchronously.
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final cameras = await availableCameras();
  final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
    orElse: () => cameras.first,
  );

  runApp(BilliardVisionDemo(camera: backCamera));
}

/// Root widget.
class BilliardVisionDemo extends StatefulWidget {
  const BilliardVisionDemo({super.key, required this.camera});

  final CameraDescription camera;

  @override
  State<BilliardVisionDemo> createState() => _BilliardVisionDemoState();
}

class _BilliardVisionDemoState extends State<BilliardVisionDemo> {
  late final CameraController _controller;
  late final Future<void> _initialization;

  // Stream of the latest detected balls so the overlay can update in real time.
  final ValueNotifier<List<Ball>> _balls = ValueNotifier<List<Ball>>([]);

  // Throttling to avoid queueing frames faster than we can process them.
  bool _readyForNextFrame = true;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    _initialization = _controller.initialize().then((_) {
      // Start streaming raw YUV420 frames.
      _controller.startImageStream(_handleCameraFrame);
    });
  }

  Future<void> _handleCameraFrame(CameraImage image) async {
    if (!_readyForNextFrame) return;
    _readyForNextFrame = false;
    try {
      final detected = await DetectionService.detectBalls(image);
      _balls.value = detected;
    } finally {
      // Allow the next frame after ~100 ms (≈ 10 fps analysis).
      await Future<void>.delayed(const Duration(milliseconds: 100));
      _readyForNextFrame = true;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    _balls.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: Scaffold(
        body: FutureBuilder(
          future: _initialization,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            return Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_controller),
                ValueListenableBuilder<List<Ball>>(
                  valueListenable: _balls,
                  builder: (context, balls, _) => CustomPaint(
                    painter: BallOverlayPainter(balls),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

/// Simple data-class representing a detected ball.
class Ball {
  const Ball({required this.center, required this.radius, required this.label});

  final Offset center; // Logical pixels relative to the preview widget.
  final double radius; // Logical pixels.
  final String label;  // e.g. "cue", "red", "yellow" …
}

/// Overlays circles on top of the camera preview for each detected [Ball].
class BallOverlayPainter extends CustomPainter {
  BallOverlayPainter(this.balls);

  final List<Ball> balls;

  @override
  void paint(Canvas canvas, Size size) {
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    for (final b in balls) {
      stroke.color = _colorForLabel(b.label);
      canvas.drawCircle(b.center, b.radius, stroke);
    }
  }

  @override
  bool shouldRepaint(covariant BallOverlayPainter oldDelegate) =>
      oldDelegate.balls != balls;

  Color _colorForLabel(String label) {
    switch (label) {
      case 'cue':
        return Colors.white;
      case 'red':
        return Colors.redAccent;
      case 'yellow':
        return Colors.yellowAccent;
      default:
        return Colors.blueAccent;
    }
  }
}

/// Singleton façade hiding the platform-specific detection implementation.
class DetectionService {
  static const _channel = MethodChannel('com.example.tableizer/detect');

  static Future<List<Ball>> detectBalls(CameraImage image) async {
    try {
      final result = await _channel.invokeMethod('detect', {
        'width': image.width,
        'height': image.height,
        'planes': image.planes.map((p) => p.bytes).toList(),
      });

      if (result is List) {
        return result
            .map((r) => Ball(
                  center: Offset(r['x'], r['y']),
                  radius: r['radius'],
                  label: 'ball_${r['class_id']}',
                ))
            .toList();
      }
    } on PlatformException catch (e) {
      print("Failed to detect balls: '${e.message}'.");
    }
    return const [];
  }
}