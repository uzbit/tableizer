import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:app/services/table_detection_result.dart';
import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart' hide BoxPainter;
import 'package:image/image.dart' as img;
import '../services/ball_detection_service.dart';
import '../services/table_detection_service.dart';

import '../detection_box.dart';
import '../widgets/table_painter.dart';
import '../widgets/ball_painter.dart';
import '../widgets/bullseye_painter.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  final BallDetectionService _ballDetectionService = BallDetectionService();
  final TableDetectionService _tableDetectionService = TableDetectionService();
  StreamSubscription<TableDetectionResult>? _tableDetectionsSubscription;
  List<Offset> _quadPoints = [];
  ui.Size? _imageSize;
  double _fps = 0.0;
  int _frameCounter = 0;
  DateTime _lastFrameTime = DateTime.now();
  
  // Image capture state
  bool _showCaptureMessage = false;
  Uint8List? _capturedImageBytes;
  CameraState? _cameraState;
  
  // Ball detection state
  List<Detection> _ballDetections = [];
  bool _isProcessingBalls = false;
  ui.Size? _capturedImageSize;
  static const double _confidenceThreshold = 0.6; // Match native CONF_THRESH from tableizer.cpp

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  @override
  void dispose() {
    _tableDetectionsSubscription?.cancel();
    _tableDetectionService.dispose();
    _ballDetectionService.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    await _ballDetectionService.initialize();
    await _tableDetectionService.initialize();
    _lastFrameTime = DateTime.now();

    _tableDetectionsSubscription =
        _tableDetectionService.detections.listen((result) {
      if (!mounted) return;

      _frameCounter++;
      final now = DateTime.now();
      final difference = now.difference(_lastFrameTime);
      double newFps = _fps;

      if (difference.inSeconds >= 1) {
        newFps = _frameCounter / difference.inSeconds;
        _frameCounter = 0;
        _lastFrameTime = now;
      }

      setState(() {
        _quadPoints = result.points;
        _imageSize = result.imageSize;
        _fps = newFps;
      });
    });
  }

  Future<void> _capturePhoto() async {
    if (_cameraState == null) return;
    
    // Use the correct CamerAwesome photo capture API
    _cameraState!.when(
      onPhotoMode: (photoState) async {
        await photoState.takePhoto();
      },
      onVideoMode: (videoState) {
        // Switch to photo mode first, or show error
        print('Switch to photo mode to capture image');
      },
      onVideoRecordingMode: (videoRecordingState) {
        // Switch to photo mode first, or show error  
        print('Stop recording and switch to photo mode to capture image');
      },
    );
  }

  void _onMediaCaptured(CaptureRequest request) async {
    request.when(
      single: (singleRequest) async {
        final file = singleRequest.file;
        if (file != null) {
          final imageBytes = await file.readAsBytes();
          setState(() {
            _capturedImageBytes = imageBytes;
            _showCaptureMessage = true;
          });
          
          // Don't auto-hide - let user control with buttons
        }
      },
      multiple: (multipleRequest) {
        // Handle multiple camera capture if needed
        print('Multiple camera capture not implemented');
      },
    );
  }

  void _clearCapturedImage() {
    setState(() {
      _capturedImageBytes = null;
      _showCaptureMessage = false;
      _ballDetections.clear();
      _capturedImageSize = null;
    });
  }

  Future<void> _processBallDetection() async {
    if (_capturedImageBytes == null || _isProcessingBalls) return;
    
    setState(() {
      _isProcessingBalls = true;
      _ballDetections.clear();
    });

    try {
      // Step 1: Decode JPEG bytes to Image
      final img.Image? decodedImage = img.decodeImage(_capturedImageBytes!);
      if (decodedImage == null) {
        throw Exception('Failed to decode captured image');
      }

      // Step 2: Convert to RGBA format (following test pattern)
      final img.Image rgbaImage = decodedImage.convert(numChannels: 4);
      final Uint8List rgbaBytes = rgbaImage.getBytes(order: img.ChannelOrder.rgba);

      // Step 3: Store image dimensions for coordinate transformation
      setState(() {
        _capturedImageSize = ui.Size(
          rgbaImage.width.toDouble(),
          rgbaImage.height.toDouble(),
        );
      });

      // Step 4: Run ball detection (following test pattern)
      final detections = await _ballDetectionService.detectFromByteBuffer(
        rgbaBytes,
        rgbaImage.width,
        rgbaImage.height,
      );

      // Filter detections by confidence (extra safety, native code should already filter)
      final filteredDetections = detections
          .where((detection) => detection.confidence >= _confidenceThreshold)
          .toList();

      setState(() {
        _ballDetections = filteredDetections;
        _isProcessingBalls = false;
      });

      print('Ball detection completed: ${filteredDetections.length} balls detected (${detections.length} total, filtered by confidence >= $_confidenceThreshold)');
    } catch (e) {
      print('Ball detection failed: $e');
      setState(() {
        _isProcessingBalls = false;
      });
    }
  }

  String _buildCaptureStatusText() {
    if (_capturedImageBytes == null) {
      return 'Image Captured!\n(Ready for ball detection)';
    }
    
    final sizeKB = (_capturedImageBytes!.length / 1024).toStringAsFixed(1);
    
    if (_isProcessingBalls) {
      return 'Processing Image...\n(${sizeKB}KB - Detecting balls)';
    } else if (_ballDetections.isNotEmpty) {
      return 'Detection Complete!\n(${_ballDetections.length} balls found - ${sizeKB}KB)';
    } else {
      return 'Image Captured!\n(${sizeKB}KB - Ready for analysis)';
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tableizer')),
      body: CameraAwesomeBuilder.custom(
        // Use CONTAIN to ensure the preview is not cropped.
        previewFit: CameraPreviewFit.contain,
        saveConfig: SaveConfig.photoAndVideo(),
        sensorConfig: SensorConfig.single(
          sensor: Sensor.position(SensorPosition.back),
          flashMode: FlashMode.none,
          // IMPORTANT: Match the analysis image aspect ratio
          aspectRatio: CameraAspectRatios.ratio_4_3,
        ),
        onImageForAnalysis: _tableDetectionService.processImage,
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: const AndroidAnalysisOptions.bgra8888(width: 1280),
          maxFramesPerSecond: 30,
        ),
        onMediaCaptureEvent: (event) {
          if (event.status == MediaCaptureStatus.success && event.isPicture) {
            _onMediaCaptured(event.captureRequest);
          } else if (event.status == MediaCaptureStatus.failure) {
            print('Photo capture failed: ${event.exception}');
          }
        },
        builder: (cameraState, preview) {
          _cameraState = cameraState;
          return Stack(
            fit: StackFit.expand,
            children: [
              // The preview is rendered automatically by the builder.
              if (_imageSize != null)
                LayoutBuilder(builder: (context, constraints) {
                  final previewSize = constraints.biggest;
                  final imageSize = _imageSize!;

                  final imageAspectRatio = imageSize.width / imageSize.height;
                  final previewAspectRatio = previewSize.width / previewSize.height;

                  double scale;
                  // This logic mimics BoxFit.contain.
                  if (previewAspectRatio > imageAspectRatio) {
                    // Preview is wider than the image -> letterbox
                    scale = previewSize.height / imageSize.height;
                  } else {
                    // Preview is taller than the image -> pillarbox
                    scale = previewSize.width / imageSize.width;
                  }

                  final scaledWidth = imageSize.width * scale;
                  final scaledHeight = imageSize.height * scale;

                  // Center the scaled image within the preview.
                  final dx = (previewSize.width - scaledWidth) / 2.0;
                  final dy = (previewSize.height - scaledHeight) / 2.0;

                  // Transform points from image-space to screen-space.
                  final scaledPoints = _quadPoints.map((p) {
                    return Offset(p.dx * scale + dx, p.dy * scale + dy);
                  }).toList();

                  return CustomPaint(
                    size: previewSize,
                    painter: TablePainter(quadPoints: scaledPoints),
                  );
                }),

              // --- Capture Message Overlay ---
              if (_showCaptureMessage)
                Container(
                  color: Colors.black87,
                  child: Column(
                    children: [
                      // Captured image display area
                      Expanded(
                        child: _capturedImageBytes != null
                            ? Stack(
                                fit: StackFit.expand,
                                children: [
                                  // Display captured image
                                  Center(
                                    child: Image.memory(
                                      _capturedImageBytes!,
                                      fit: BoxFit.contain,
                                    ),
                                  ),
                                  // Ball detection overlay
                                  if (_ballDetections.isNotEmpty && _capturedImageSize != null)
                                    LayoutBuilder(
                                      builder: (context, constraints) {
                                        return CustomPaint(
                                          size: constraints.biggest,
                                          painter: BallPainter(
                                            detections: _ballDetections,
                                            imageSize: _capturedImageSize!,
                                            displaySize: constraints.biggest,
                                          ),
                                        );
                                      },
                                    ),
                                ],
                              )
                            : Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const Icon(
                                      Icons.photo_camera,
                                      size: 80,
                                      color: Colors.white,
                                    ),
                                    const SizedBox(height: 16),
                                    Text(
                                      _buildCaptureStatusText(),
                                      textAlign: TextAlign.center,
                                      style: TextStyle(
                                        color: Colors.white,
                                        fontSize: 18,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                      ),
                      // Status and controls area
                      Container(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          children: [
                            if (_capturedImageBytes != null) ...[
                              Text(
                                _buildCaptureStatusText(),
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 16),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                                children: [
                                  ElevatedButton(
                                    onPressed: _clearCapturedImage,
                                    child: const Text('Retake'),
                                  ),
                                  ElevatedButton(
                                    onPressed: _isProcessingBalls ? null : _processBallDetection,
                                    child: _isProcessingBalls 
                                        ? const SizedBox(
                                            width: 16,
                                            height: 16,
                                            child: CircularProgressIndicator(
                                              strokeWidth: 2,
                                              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                            ),
                                          )
                                        : const Text('Analyze'),
                                  ),
                                ],
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

              // --- Bullseye Reticule (Center) ---
              if (!_showCaptureMessage)
                Center(
                  child: CustomPaint(
                    size: const Size(40, 40),
                    painter: BullseyePainter(),
                  ),
                ),

              // --- UI Overlays (FPS, etc.) ---
              if (!_showCaptureMessage)
                Align(
                  alignment: Alignment.topLeft,
                  child: Container(
                    color: Colors.black.withValues(alpha: 0.5),
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
        onPressed: _capturePhoto,
        tooltip: 'Capture Image',
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}