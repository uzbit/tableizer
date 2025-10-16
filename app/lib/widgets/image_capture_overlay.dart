import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';
import '../models/ball_detection_result.dart';
import '../models/table_detection_result.dart';
import 'ball_painter.dart';
import 'table_painter.dart';

class ImageCaptureOverlay extends StatelessWidget {
  final Uint8List? capturedBgraBytes;
  final ui.Size? capturedImageSize;
  final InputAnalysisImageRotation? capturedRotation;
  final List<BallDetectionResult> ballDetections;
  final TableDetectionResult? tableDetectionResult;
  final bool isProcessingBalls;
  final String statusText;
  final VoidCallback? onRetake;
  final VoidCallback? onAnalyze;
  final VoidCallback? onAccept;
  final VoidCallback? onClose;

  const ImageCaptureOverlay({
    super.key,
    required this.capturedBgraBytes,
    required this.capturedImageSize,
    required this.capturedRotation,
    required this.ballDetections,
    this.tableDetectionResult,
    required this.isProcessingBalls,
    required this.statusText,
    this.onRetake,
    this.onAnalyze,
    this.onAccept,
    this.onClose,
  });

  /// Compute canvas size based on ImageAdapter's 16:9 normalization logic
  Size _computeCanvasSize(ui.Size originalImageSize, InputAnalysisImageRotation rotation) {
    // Apply rotation to get post-rotation dimensions
    double width = originalImageSize.width;
    double height = originalImageSize.height;

    if (rotation == InputAnalysisImageRotation.rotation90deg ||
        rotation == InputAnalysisImageRotation.rotation270deg) {
      // Swap dimensions for 90/270 degree rotations
      final temp = width;
      width = height;
      height = temp;
    }

    // ImageAdapter uses max dimension as width, then computes 16:9 height
    const targetAspectRatio = 16.0 / 9.0;
    final canvasWidth = width > height ? width : height;
    final canvasHeight = canvasWidth / targetAspectRatio;

    return Size(canvasWidth, canvasHeight);
  }

  List<Offset> _scalePoints({
    required List<Offset> points,
    required Size sourceSize,
    required Size targetSize,
    BoxFit fit = BoxFit.contain,
  }) {
    final double scaleX = targetSize.width / sourceSize.width;
    final double scaleY = targetSize.height / sourceSize.height;

    double scale;
    double offsetX = 0;
    double offsetY = 0;

    if (fit == BoxFit.contain) {
      scale = scaleX < scaleY ? scaleX : scaleY;
      final scaledWidth = sourceSize.width * scale;
      final scaledHeight = sourceSize.height * scale;
      offsetX = (targetSize.width - scaledWidth) / 2;
      offsetY = (targetSize.height - scaledHeight) / 2;
    } else {
      // Other BoxFit modes can be implemented here if needed
      scale = 1.0;
    }

    return points.map((point) {
      return Offset(
        point.dx * scale + offsetX,
        point.dy * scale + offsetY,
      );
    }).toList();
  }

  List<Offset> _transformQuadPoints(
    TableDetectionResult tableResult,
    ui.Size capturedImageSize,
    InputAnalysisImageRotation rotation,
    Size displaySize,
  ) {
    // The source for scaling is the size of the image *after* rotation,
    // which is what the quad points are relative to.
    final Size sourceSize =
        tableResult.originalImageSize ?? _computeRotatedImageSize(capturedImageSize, rotation);

    // Scale points from the rotated image space to the display space
    return _scalePoints(
      points: tableResult.points,
      sourceSize: sourceSize,
      targetSize: displaySize,
      fit: BoxFit.contain,
    );
  }

  Size _computeRotatedImageSize(ui.Size originalSize, InputAnalysisImageRotation rotation) {
    if (rotation == InputAnalysisImageRotation.rotation90deg ||
        rotation == InputAnalysisImageRotation.rotation270deg) {
      return Size(originalSize.height, originalSize.width);
    }
    return Size(originalSize.width, originalSize.height);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black87,
      child: Column(
        children: [
          // Captured image display area
          Expanded(
            child: capturedBgraBytes != null && capturedImageSize != null && capturedRotation != null
                ? Stack(
                    fit: StackFit.expand,
                    children: [
                      // Display captured BGRA image directly with rotation handling
                      Center(
                        child: _BgraImageWidget(
                          bgraBytes: capturedBgraBytes!,
                          width: capturedImageSize!.width.toInt(),
                          height: capturedImageSize!.height.toInt(),
                          rotation: capturedRotation!,
                          maskBytes: tableDetectionResult?.maskBytes,
                        ),
                      ),
                      // Detection overlays (both balls and table)
                      if (ballDetections.isNotEmpty || tableDetectionResult != null)
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
                                        tableDetectionResult!,
                                        capturedImageSize!,
                                        capturedRotation!,
                                        constraints.biggest,
                                      ),
                                      orientation: tableDetectionResult!.orientation,
                                    ),
                                  ),
                                // Ball detection overlay (drawn on top)
                                if (ballDetections.isNotEmpty)
                                  CustomPaint(
                                    size: constraints.biggest,
                                    painter: CanvasSpaceBallPainter(
                                      detections: ballDetections,
                                      canvasSize: tableDetectionResult?.imageSize ??
                                          _computeCanvasSize(capturedImageSize!, capturedRotation!),
                                      rotatedImageSize:
                                          _computeRotatedImageSize(capturedImageSize!, capturedRotation!),
                                      displaySize: constraints.biggest,
                                    ),
                                  ),
                              ],
                            );
                          },
                        ),
                      // Loading spinner overlay
                      if (isProcessingBalls)
                        Container(
                          color: Colors.black54,
                          child: const Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                CircularProgressIndicator(
                                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                ),
                                SizedBox(height: 16),
                                Text(
                                  'Analyzing...',
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
                if (capturedBgraBytes != null) ...[
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
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      if (onRetake != null) ...[
                        ElevatedButton(
                          onPressed: onRetake,
                          child: const Text('Retake'),
                        ),
                        const SizedBox(width: 16),
                      ],
                      if (onAnalyze != null)
                        ElevatedButton(
                          onPressed: isProcessingBalls ? null : onAnalyze,
                          child: const Text('Analyze'),
                        ),
                      if (onAccept != null && onAnalyze == null)
                        ElevatedButton(
                          onPressed: onAccept,
                          child: const Text('Accept'),
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

/// Widget to display raw BGRA bytes as an image with rotation handling
class _BgraImageWidget extends StatelessWidget {
  final Uint8List bgraBytes;
  final int width;
  final int height;
  final InputAnalysisImageRotation rotation;
  final Uint8List? maskBytes;

  const _BgraImageWidget({
    required this.bgraBytes,
    required this.width,
    required this.height,
    required this.rotation,
    this.maskBytes,
  });

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<ui.Image>(
      future: _createAndMaskImage(),
      builder: (context, snapshot) {
        if (snapshot.hasData) {
          // Apply rotation transform to match camera orientation
          Widget imageWidget = RawImage(
            image: snapshot.data,
            fit: BoxFit.contain,
          );

          // Rotate the image to match the camera orientation
          // The BGRA bytes are in sensor orientation, need to rotate for display
          if (rotation == InputAnalysisImageRotation.rotation90deg) {
            // Portrait mode on phones: sensor is landscape, rotate 90° clockwise
            imageWidget = RotatedBox(
              quarterTurns: 1, // 90° clockwise
              child: imageWidget,
            );
          } else if (rotation == InputAnalysisImageRotation.rotation180deg) {
            imageWidget = RotatedBox(
              quarterTurns: 2, // 180°
              child: imageWidget,
            );
          } else if (rotation == InputAnalysisImageRotation.rotation270deg) {
            imageWidget = RotatedBox(
              quarterTurns: 3, // 270° clockwise (90° counter-clockwise)
              child: imageWidget,
            );
          }
          // rotation0deg: no rotation needed

          return imageWidget;
        } else {
          return const Center(
            child: CircularProgressIndicator(),
          );
        }
      },
    );
  }

  Future<ui.Image> _createAndMaskImage() async {
    final image = await _createImageFromBgra(bgraBytes, width, height);
    if (maskBytes != null) {
      return await _applyMaskToImage(image, maskBytes!);
    }
    return image;
  }

  Future<ui.Image> _createImageFromBgra(Uint8List bgra, int width, int height) async {
    print('[_BgraImageWidget] Creating image from ${bgra.length} bytes, size: ${width}x$height');
    print('[_BgraImageWidget] First 16 bytes: ${bgra.sublist(0, 16)}');

    // The native pipeline is now standardized to always output RGBA.
    const pixelFormat = ui.PixelFormat.rgba8888;
    print('[_BgraImageWidget] Using pixel format: RGBA (all platforms)');

    final completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      bgra, // variable name is now a misnomer, but we'll leave it
      width,
      height,
      pixelFormat,
      (ui.Image image) {
        completer.complete(image);
      },
    );
    return completer.future;
  }

  Future<ui.Image> _applyMaskToImage(ui.Image image, Uint8List maskPngBytes) async {
    // Decode mask PNG to image
    final maskCompleter = Completer<ui.Image>();
    ui.decodeImageFromList(maskPngBytes, (ui.Image maskImage) {
      maskCompleter.complete(maskImage);
    });
    final maskImage = await maskCompleter.future;

    // Create a canvas to composite the image with the mask
    final recorder = ui.PictureRecorder();
    final canvas = Canvas(recorder);
    final paint = Paint();

    // Draw the original image
    canvas.drawImage(image, Offset.zero, paint);

    // Apply mask using destination-in blend mode (keeps only pixels where mask is opaque)
    paint.blendMode = BlendMode.dstIn;
    canvas.drawImage(maskImage, Offset.zero, paint);

    final picture = recorder.endRecording();
    final composited = await picture.toImage(image.width, image.height);

    return composited;
  }
}

/// Ball painter for detections in canvas space (already rotated by ImageAdapter)
class CanvasSpaceBallPainter extends CustomPainter {
  final List<BallDetectionResult> detections;
  final Size canvasSize; // The 16:9 canvas size from ImageAdapter
  final Size rotatedImageSize; // The size of the image inside the canvas
  final Size displaySize;

  CanvasSpaceBallPainter({
    required this.detections,
    required this.canvasSize,
    required this.rotatedImageSize,
    required this.displaySize,
  });

  String _getClassLabel(int classId) {
    switch (classId) {
      case 0:
        return 'Black';
      case 1:
        return 'Cue';
      case 2:
        return 'Solid';
      case 3:
        return 'Stripe';
      default:
        return 'Ball';
    }
  }

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    print('[BALL_OVERLAY] ═══════════════════════════════════════');
    print('[BALL_OVERLAY] Canvas size: ${canvasSize.width}x${canvasSize.height}');
    print('[BALL_OVERLAY] Rotated Image size: ${rotatedImageSize.width}x${rotatedImageSize.height}');
    print('[BALL_OVERLAY] Display size: ${displaySize.width}x${displaySize.height}');
    print('[BALL_OVERLAY] Ball count: ${detections.length}');

    // Calculate how the `rotatedImageSize` (the content) fits into `displaySize`
    final double displayScaleX = displaySize.width / rotatedImageSize.width;
    final double displayScaleY = displaySize.height / rotatedImageSize.height;
    final double displayScale = displayScaleX < displayScaleY ? displayScaleX : displayScaleY;

    final double scaledContentWidth = rotatedImageSize.width * displayScale;
    final double scaledContentHeight = rotatedImageSize.height * displayScale;
    final double displayOffsetX = (displaySize.width - scaledContentWidth) / 2;
    final double displayOffsetY = (displaySize.height - scaledContentHeight) / 2;

    // Calculate the padding the C++ side added to fit `rotatedImageSize` into `canvasSize`
    final double contentPadX = (canvasSize.width - rotatedImageSize.width) / 2;
    final double contentPadY = (canvasSize.height - rotatedImageSize.height) / 2;

    print('[BALL_OVERLAY] Display Scale: $displayScale, Display Offset: ($displayOffsetX, $displayOffsetY)');
    print('[BALL_OVERLAY] C++ Padding: ($contentPadX, $contentPadY)');

    // Paint for bounding boxes
    final Paint boxPaint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final Paint textBgPaint = Paint()..color = Colors.black54;

    for (final detection in detections) {
      // The detection box is relative to `canvasSize`.
      // 1. Un-pad the coordinates to make them relative to `rotatedImageSize`.
      final Rect unpaddedBox = Rect.fromLTWH(
        detection.box.x.toDouble() - contentPadX,
        detection.box.y.toDouble() - contentPadY,
        detection.box.width.toDouble(),
        detection.box.height.toDouble(),
      );

      // 2. Scale the un-padded box to the final display coordinates.
      final transformedBox = Rect.fromLTWH(
        (unpaddedBox.left * displayScale) + displayOffsetX,
        (unpaddedBox.top * displayScale) + displayOffsetY,
        unpaddedBox.width * displayScale,
        unpaddedBox.height * displayScale,
      );

      // Draw bounding box
      canvas.drawRect(transformedBox, boxPaint);

      // Draw label
      final classLabel = _getClassLabel(detection.classId);
      final confidencePercent = (detection.confidence * 100).toStringAsFixed(1);
      final labelText = '$classLabel ${confidencePercent}%';

      final textSpan = TextSpan(
        text: labelText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();

      final textBgRect = Rect.fromLTWH(
        transformedBox.left,
        transformedBox.top - 20,
        textPainter.width + 6,
        18,
      );
      canvas.drawRect(textBgRect, textBgPaint);
      textPainter.paint(
        canvas,
        Offset(transformedBox.left + 3, transformedBox.top - 17),
      );
    }

    print('[BALL_OVERLAY] ═══════════════════════════════════════');
  }

  @override
  bool shouldRepaint(CanvasSpaceBallPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.canvasSize != canvasSize ||
        oldDelegate.rotatedImageSize != rotatedImageSize ||
        oldDelegate.displaySize != displaySize;
  }
}