import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import '../models/ball_detection_result.dart';
import '../models/table_detection_result.dart';
import 'ball_painter.dart';
import 'table_painter.dart';

class ImageCaptureOverlay extends StatelessWidget {
  final Uint8List? capturedImageBytes;
  final List<BallDetectionResult> ballDetections;
  final ui.Size? capturedImageSize;
  final TableDetectionResult? tableDetectionResult;
  final bool isProcessingBalls;
  final String statusText;
  final VoidCallback onRetake;
  final VoidCallback onAnalyze;
  final VoidCallback? onAccept;
  final VoidCallback? onClose;

  const ImageCaptureOverlay({
    super.key,
    required this.capturedImageBytes,
    required this.ballDetections,
    required this.capturedImageSize,
    this.tableDetectionResult,
    required this.isProcessingBalls,
    required this.statusText,
    required this.onRetake,
    required this.onAnalyze,
    this.onAccept,
    this.onClose,
  });

  List<Offset> _transformQuadPoints(List<Offset> points, ui.Size imageSize, Size displaySize) {
    // Transform coordinates from image space to display space
    final scaleX = displaySize.width / imageSize.width;
    final scaleY = displaySize.height / imageSize.height;
    
    // Use the same scaling approach as BallPainter
    final scale = scaleX < scaleY ? scaleX : scaleY;
    final scaledWidth = imageSize.width * scale;
    final scaledHeight = imageSize.height * scale;
    
    final offsetX = (displaySize.width - scaledWidth) / 2;
    final offsetY = (displaySize.height - scaledHeight) / 2;
    
    return points.map((point) {
      return Offset(
        point.dx * scale + offsetX,
        point.dy * scale + offsetY,
      );
    }).toList();
  }

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
                      // Detection overlays (both balls and table)
                      if (capturedImageSize != null && (ballDetections.isNotEmpty || tableDetectionResult != null))
                        LayoutBuilder(
                          builder: (context, constraints) {
                            return Stack(
                              children: [
                                // Table quad overlay (drawn first, underneath balls)
                                if (tableDetectionResult != null)
                                  CustomPaint(
                                    size: constraints.biggest,
                                    painter: TablePainter(
                                      quadPoints: _transformQuadPoints(
                                        tableDetectionResult!.points,
                                        capturedImageSize!,
                                        constraints.biggest,
                                      ),
                                    ),
                                  ),
                                // Ball detection overlay (drawn on top)
                                if (ballDetections.isNotEmpty)
                                  CustomPaint(
                                    size: constraints.biggest,
                                    painter: BallPainter(
                                      detections: ballDetections,
                                      imageSize: capturedImageSize!,
                                      displaySize: constraints.biggest,
                                    ),
                                  ),
                              ],
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
                        onPressed: isProcessingBalls 
                            ? null 
                            : ballDetections.isNotEmpty && onAccept != null
                                ? onAccept
                                : onAnalyze,
                        child: isProcessingBalls 
                            ? const SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                ),
                              )
                            : Text(ballDetections.isNotEmpty ? 'Accept' : 'Analyze'),
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