import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../controllers/ball_detection_controller.dart';
import '../detection_box.dart';
import 'ball_painter.dart';

class ImageCaptureOverlay extends StatelessWidget {
  final Uint8List? capturedImageBytes;
  final List<Detection> ballDetections;
  final ui.Size? capturedImageSize;
  final bool isProcessingBalls;
  final String statusText;
  final VoidCallback onRetake;
  final VoidCallback onAnalyze;

  const ImageCaptureOverlay({
    super.key,
    required this.capturedImageBytes,
    required this.ballDetections,
    required this.capturedImageSize,
    required this.isProcessingBalls,
    required this.statusText,
    required this.onRetake,
    required this.onAnalyze,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black87,
      child: Column(
        children: [
          // Captured image display area
          Expanded(
            child: capturedImageBytes != null
                ? Stack(
                    fit: StackFit.expand,
                    children: [
                      // Display captured image
                      Center(
                        child: Image.memory(
                          capturedImageBytes!,
                          fit: BoxFit.contain,
                        ),
                      ),
                      // Ball detection overlay
                      if (ballDetections.isNotEmpty && capturedImageSize != null)
                        LayoutBuilder(
                          builder: (context, constraints) {
                            return CustomPaint(
                              size: constraints.biggest,
                              painter: BallPainter(
                                detections: ballDetections,
                                imageSize: capturedImageSize!,
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
                          statusText,
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
                if (capturedImageBytes != null) ...[
                  Text(
                    statusText,
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
                        onPressed: onRetake,
                        child: const Text('Retake'),
                      ),
                      ElevatedButton(
                        onPressed: isProcessingBalls ? null : onAnalyze,
                        child: isProcessingBalls 
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
    );
  }
}