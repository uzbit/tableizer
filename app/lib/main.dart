import 'dart:async';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'tableizer_bindings.dart';

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
  late final DetectionService _detectionService;

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

    _initialization = _controller.initialize().then((_) async {
      _detectionService = await DetectionService.create();
      // Start streaming raw YUV420 frames.
      _controller.startImageStream(_handleCameraFrame);
    });
  }

  Future<void> _handleCameraFrame(CameraImage image) async {
    if (!_readyForNextFrame) return;
    _readyForNextFrame = false;
    try {
      final detected = await _detectionService.detectBalls(image);
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
    _detectionService.dispose();
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
  final TableizerBindings _bindings;
  final Pointer<Void> _detector;

  DetectionService._(this._bindings, this._detector);

  static Future<DetectionService> create() async {
    final dylib = ffi.DynamicLibrary.open('libtableizer.so');
    final bindings = TableizerBindings(dylib);

    final modelPath = await _getModelPath();
    final modelPathC = modelPath.toNativeUtf8();
    final detector = bindings.create_ball_detector.asFunction<ffi.Pointer<ffi.Void> Function(ffi.Pointer<Utf8>)>()(modelPathC);
    calloc.free(modelPathC);

    return DetectionService._(bindings, detector);
  }

  void dispose() {
    _bindings.destroy_ball_detector.asFunction<void Function(ffi.Pointer<ffi.Void>)>()(_detector);
  }

  Future<List<Ball>> detectBalls(CameraImage image) async {
    final p0 = image.planes[0].bytes.toPtr();
    final p1 = image.planes[1].bytes.toPtr();
    final p2 = image.planes[2].bytes.toPtr();

    final count = calloc<ffi.Int32>();
    final detections = _bindings.detect_balls.asFunction<ffi.Pointer<Detection> Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Pointer<ffi.Uint8>, ffi.Int32, ffi.Int32, ffi.Int32, ffi.Int32, ffi.Int32, ffi.Int32, ffi.Int32, ffi.Float, ffi.Float, ffi.Pointer<ffi.Int32>)>()(
        _detector,
        p0,
        p1,
        p2,
        image.width,
        image.height,
        image.planes[0].bytesPerRow,
        image.planes[1].bytesPerRow,
        image.planes[2].bytesPerRow,
        image.planes[1].bytesPerPixel!,
        image.planes[2].bytesPerPixel!,
        0.25,
        0.45,
        count);

    final result = <Ball>[];
    for (var i = 0; i < count.value; i++) {
      final d = detections.elementAt(i).ref;
      result.add(Ball(
        center: Offset(d.x, d.y),
        radius: d.radius,
        label: 'ball_${d.class_id}',
      ));
    }

    _bindings.free_detections.asFunction<void Function(ffi.Pointer<Detection>)>()(detections);
    calloc.free(count);
    calloc.free(p0);
    calloc.free(p1);
    calloc.free(p2);

    return result;
  }

  static Future<String> _getModelPath() async {
    final docDir = await getApplicationDocumentsDirectory();
    final modelPath = '${docDir.path}/best.onnx';
    final file = File(modelPath);
    if (!await file.exists()) {
      final byteData = await rootBundle.load('assets/best.onnx');
      await file.writeAsBytes(byteData.buffer.asUint8List());
    }
    return modelPath;
  }
}

extension on Uint8List {
  Pointer<Uint8> toPtr() {
    final ptr = calloc<Uint8>(length);
    ptr.asTypedList(length).setAll(0, this);
    return ptr;
  }
}
