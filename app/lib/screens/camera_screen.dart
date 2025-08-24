import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';

import '../controllers/camera_controller.dart';
import '../controllers/table_detection_controller.dart';
import '../controllers/ball_detection_controller.dart';
import '../widgets/image_capture_overlay.dart';
import '../widgets/camera_preview_widget.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
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
  void dispose() {
    _cameraController.dispose();
    _tableDetectionController.dispose();
    _ballDetectionController.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    await _tableDetectionController.initialize();
    await _ballDetectionController.initialize();
  }

  void _onClearImage() {
    _cameraController.clearCapturedImage();
    _ballDetectionController.clearDetections();
  }

  void _onAnalyzeImage() {
    final imageBytes = _cameraController.capturedImageBytes;
    if (imageBytes != null) {
      _ballDetectionController.processBallDetection(imageBytes);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tableizer')),
      body: CameraAwesomeBuilder.custom(
        previewFit: CameraPreviewFit.contain,
        saveConfig: SaveConfig.photoAndVideo(),
        sensorConfig: SensorConfig.single(
          sensor: Sensor.position(SensorPosition.back),
          flashMode: FlashMode.none,
          aspectRatio: CameraAspectRatios.ratio_4_3,
        ),
        onImageForAnalysis: _tableDetectionController.processImage,
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: const AndroidAnalysisOptions.bgra8888(width: 1280),
          maxFramesPerSecond: 30,
        ),
        onMediaCaptureEvent: (event) {
          if (event.status == MediaCaptureStatus.success && event.isPicture) {
            _cameraController.onMediaCaptured(event.captureRequest);
          } else if (event.status == MediaCaptureStatus.failure) {
            print('Photo capture failed: ${event.exception}');
          }
        },
        builder: (cameraState, preview) {
          _cameraController.setCameraState(cameraState);
          
          return AnimatedBuilder(
            animation: Listenable.merge([_cameraController, _tableDetectionController, _ballDetectionController]),
            builder: (context, child) {
              return Stack(
                fit: StackFit.expand,
                children: [
                  // Live camera preview with overlays
                  if (!_cameraController.showCaptureMessage)
                    CameraPreviewWidget(
                      quadPoints: _tableDetectionController.quadPoints,
                      imageSize: _tableDetectionController.imageSize,
                      fps: _tableDetectionController.fps,
                    ),

                  // Capture and analysis overlay
                  if (_cameraController.showCaptureMessage)
                    ImageCaptureOverlay(
                      capturedImageBytes: _cameraController.capturedImageBytes,
                      ballDetections: _ballDetectionController.ballDetections,
                      capturedImageSize: _ballDetectionController.capturedImageSize,
                      isProcessingBalls: _ballDetectionController.isProcessingBalls,
                      statusText: _ballDetectionController.buildCaptureStatusText(
                        _cameraController.capturedImageBytes,
                      ),
                      onRetake: _onClearImage,
                      onAnalyze: _onAnalyzeImage,
                    ),
                ],
              );
            },
          );
        },
      ),
      floatingActionButton: AnimatedBuilder(
        animation: _cameraController,
        builder: (context, child) {
          return FloatingActionButton(
            onPressed: _cameraController.capturePhoto,
            tooltip: 'Capture Image',
            child: const Icon(Icons.camera_alt),
          );
        },
      ),
    );
  }
}