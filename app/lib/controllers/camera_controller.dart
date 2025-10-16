import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';

class CameraController extends ChangeNotifier {
  // Analysis frame capture state
  Uint8List? _capturedBgraBytes;
  ui.Size? _capturedImageSize;
  InputAnalysisImageRotation? _capturedRotation;
  bool _waitingForFrame = false;

  // Getters
  Uint8List? get capturedBgraBytes => _capturedBgraBytes;
  ui.Size? get capturedImageSize => _capturedImageSize;
  InputAnalysisImageRotation? get capturedRotation => _capturedRotation;

  /// Request to capture the next analysis frame
  void requestFrameCapture() {
    _waitingForFrame = true;
  }

  /// Called from the analysis stream to capture a frame
  void captureAnalysisFrame(AnalysisImage analysisImage) {
    if (!_waitingForFrame) return;

    analysisImage.when(
      bgra8888: (image) {
        final plane = image.planes[0];
        final bytes = plane.bytes;
        final bytesPerRow = plane.bytesPerRow;
        final expectedRowBytes = image.width * 4; // BGRA = 4 bytes per pixel

        print('[CAMERA_CONTROLLER] ═══════════════════════════════');
        print('[CAMERA_CONTROLLER] Image size: ${image.width}x${image.height}');
        print('[CAMERA_CONTROLLER] Bytes per row (stride): $bytesPerRow');
        print('[CAMERA_CONTROLLER] Expected row bytes: $expectedRowBytes');
        print('[CAMERA_CONTROLLER] Total bytes: ${bytes.length}');
        print('[CAMERA_CONTROLLER] Has padding: ${bytesPerRow > expectedRowBytes}');

        // Check if image has row padding
        if (bytesPerRow > expectedRowBytes) {
          // Image has padding - need to remove it for correct decoding
          print('[CAMERA_CONTROLLER] Removing row padding...');
          final tightlyPacked = Uint8List(image.width * image.height * 4);

          for (int y = 0; y < image.height; y++) {
            final srcOffset = y * bytesPerRow;
            final dstOffset = y * expectedRowBytes;
            tightlyPacked.setRange(
              dstOffset,
              dstOffset + expectedRowBytes,
              bytes,
              srcOffset,
            );
          }

          _capturedBgraBytes = tightlyPacked;
          print('[CAMERA_CONTROLLER] Created tightly packed buffer: ${tightlyPacked.length} bytes');
        } else {
          // No padding - use as-is
          _capturedBgraBytes = Uint8List.fromList(bytes);
          print('[CAMERA_CONTROLLER] No padding detected, using original bytes');
        }

        _capturedImageSize = ui.Size(image.width.toDouble(), image.height.toDouble());
        _capturedRotation = image.rotation;

        print('[CAMERA_CONTROLLER] ═══════════════════════════════');

        _waitingForFrame = false;
        notifyListeners();
      },
    );
  }

  void clearCapturedImage() {
    _capturedBgraBytes = null;
    _capturedImageSize = null;
    _capturedRotation = null;
    notifyListeners();
  }
}