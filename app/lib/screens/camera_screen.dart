import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';

import '../controllers/camera_controller.dart';
import '../controllers/table_detection_controller.dart';
import '../controllers/ball_detection_controller.dart';
import '../models/table_detection_result.dart';
import '../widgets/image_capture_overlay.dart';
import '../widgets/camera_preview_widget.dart';
import 'table_results_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  CameraScreenState createState() => CameraScreenState();
}

// Global route observer for tracking route changes
final RouteObserver<PageRoute> routeObserver = RouteObserver<PageRoute>();

class CameraScreenState extends State<CameraScreen> with RouteAware {
  late final CameraController _cameraController;
  late final TableDetectionController _tableDetectionController;
  late final BallDetectionController _ballDetectionController;

  @override
  void initState() {
    super.initState();
    _cameraController = CameraController();
    _tableDetectionController = TableDetectionController();
    _ballDetectionController = BallDetectionController();
    _initializeServices();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Subscribe to route changes
    routeObserver.subscribe(this, ModalRoute.of(context) as PageRoute);
  }

  @override
  void dispose() {
    routeObserver.unsubscribe(this);
    _cameraController.dispose();
    _tableDetectionController.dispose();
    _ballDetectionController.dispose();
    super.dispose();
  }

  // RouteAware callbacks
  @override
  void didPush() {
    // Route was pushed onto navigator and is now topmost route
    _tableDetectionController.resume();
    print('[CAMERA_SCREEN] didPush - resuming table detection');
  }

  @override
  void didPopNext() {
    // Covering route was popped off the navigator, this route is now topmost
    _tableDetectionController.resume();
    print('[CAMERA_SCREEN] didPopNext - resuming table detection');
  }

  @override
  void didPushNext() {
    // New route was pushed on top of this route
    _tableDetectionController.pause();
    print('[CAMERA_SCREEN] didPushNext - pausing table detection');
  }

  @override
  void didPop() {
    // This route was popped off the navigator
    _tableDetectionController.pause();
    print('[CAMERA_SCREEN] didPop - pausing table detection');
  }

  Future<void> _initializeServices() async {
    await _tableDetectionController.initialize();
    await _ballDetectionController.initialize();
  }

  void _onCaptureFrame() {
    // Request capture of next analysis frame
    _cameraController.requestFrameCapture();
  }

  void _onAcceptResults(ui.Size capturedImageSize, InputAnalysisImageRotation rotation) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => TableResultsScreen(
          ballDetections: _ballDetectionController.ballDetections,
          capturedImageSize: capturedImageSize,
          capturedRotation: rotation,
          tableDetectionResult: _tableDetectionController.tableDetectionResult,
          tableDetectionService: _tableDetectionController.tableDetectionService,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tableizer')),
      body: CameraAwesomeBuilder.awesome(
        saveConfig: SaveConfig.photo(),
        previewFit: CameraPreviewFit.contain,
        enablePhysicalButton: false,
        sensorConfig: SensorConfig.single(
          sensor: Sensor.position(SensorPosition.back),
          flashMode: FlashMode.none,
          aspectRatio: CameraAspectRatios.ratio_1_1,
          zoom: 0.0,
        ),
        onImageForAnalysis: (image) async {
          // Process table detection
          await _tableDetectionController.processImage(image);

          // Capture frame if requested
          _cameraController.captureAnalysisFrame(image);

          // Navigate to analysis screen when frame is captured
          if (_cameraController.capturedBgraBytes != null && mounted) {
            final capturedBytes = _cameraController.capturedBgraBytes!;
            final capturedSize = _cameraController.capturedImageSize!;
            final capturedRotation = _cameraController.capturedRotation!;
            final tableResult = _tableDetectionController.tableDetectionResult;

            // Clear the capture flag to prevent re-navigation
            final bgraBytes = capturedBytes;
            final imageSize = capturedSize;
            final rotation = capturedRotation;
            _cameraController.clearCapturedImage();

            Navigator.of(context).push(
              PageRouteBuilder(
                pageBuilder: (context, animation, secondaryAnimation) => _AnalysisScreen(
                  capturedBgraBytes: bgraBytes,
                  capturedImageSize: imageSize,
                  capturedRotation: rotation,
                  tableDetectionResult: tableResult,
                  ballDetectionController: _ballDetectionController,
                  onAcceptResults: _onAcceptResults,
                ),
                transitionsBuilder: (context, animation, secondaryAnimation, child) {
                  const begin = Offset(1.0, 0.0);
                  const end = Offset.zero;
                  const curve = Curves.easeInOut;
                  var tween = Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
                  var offsetAnimation = animation.drive(tween);
                  return SlideTransition(position: offsetAnimation, child: child);
                },
              ),
            );
          }
        },
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: const AndroidAnalysisOptions.bgra8888(width: 1280),
          maxFramesPerSecond: 30,
        ),
        previewDecoratorBuilder: (state, preview) {
          return AnimatedBuilder(
            animation: _tableDetectionController,
            builder: (context, child) {
              return CameraPreviewWidget(
                tableDetectionResult: _tableDetectionController.tableDetectionResult,
                fps: _tableDetectionController.fps,
              );
            },
          );
        },
        topActionsBuilder: (state) => const SizedBox.shrink(),
        bottomActionsBuilder: (state) => const SizedBox.shrink(),
        middleContentBuilder: (state) => const SizedBox.shrink(),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _onCaptureFrame,
        tooltip: 'Capture Frame',
        backgroundColor: const Color.fromRGBO(118, 180, 136, 1.0),
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}

class _AnalysisScreen extends StatefulWidget {
  final Uint8List capturedBgraBytes;
  final ui.Size capturedImageSize;
  final InputAnalysisImageRotation capturedRotation;
  final TableDetectionResult? tableDetectionResult;
  final BallDetectionController ballDetectionController;
  final void Function(ui.Size, InputAnalysisImageRotation) onAcceptResults;

  const _AnalysisScreen({
    required this.capturedBgraBytes,
    required this.capturedImageSize,
    required this.capturedRotation,
    required this.tableDetectionResult,
    required this.ballDetectionController,
    required this.onAcceptResults,
  });

  @override
  State<_AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<_AnalysisScreen> {
  @override
  void initState() {
    super.initState();
    // Start ball detection after the first frame is rendered
    WidgetsBinding.instance.addPostFrameCallback((_) {
      // Convert InputAnalysisImageRotation to degrees
      int rotationDegrees = 0;
      if (widget.capturedRotation == InputAnalysisImageRotation.rotation90deg) {
        rotationDegrees = 90;
      } else if (widget.capturedRotation == InputAnalysisImageRotation.rotation270deg) {
        rotationDegrees = 270;
      } else if (widget.capturedRotation == InputAnalysisImageRotation.rotation180deg) {
        rotationDegrees = 180;
      }

      widget.ballDetectionController.processBallDetection(
        widget.capturedBgraBytes,
        widget.capturedImageSize.width.toInt(),
        widget.capturedImageSize.height.toInt(),
        widget.tableDetectionResult,
        rotationDegrees,
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analyzing Image'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.of(context).pop();
          },
        ),
      ),
      body: AnimatedBuilder(
        animation: widget.ballDetectionController,
        builder: (context, child) {
          return ImageCaptureOverlay(
            capturedBgraBytes: widget.capturedBgraBytes,
            capturedImageSize: widget.capturedImageSize,
            capturedRotation: widget.capturedRotation,
            ballDetections: widget.ballDetectionController.ballDetections,
            tableDetectionResult: widget.tableDetectionResult,
            isProcessingBalls: widget.ballDetectionController.isProcessingBalls,
            statusText: widget.ballDetectionController.buildCaptureStatusText(
              widget.capturedBgraBytes,
            ),
            onRetake: null,
            onAnalyze: null,
            onAccept: widget.ballDetectionController.isProcessingBalls
                ? null
                : () {
                    Navigator.of(context).pop();
                    widget.onAcceptResults(widget.capturedImageSize, widget.capturedRotation);
                  },
          );
        },
      ),
    );
  }
}