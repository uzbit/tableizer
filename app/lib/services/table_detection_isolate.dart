import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:ui';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart' hide Size;

import '../models/table_detection_result.dart';
import '../native/library_loader.dart';

// --- FFI Function Signatures ---
typedef DetectTableBgraC = Pointer<Utf8> Function(
    Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 stride, Int32 channelFormat);
typedef DetectTableBgraDart = Pointer<Utf8> Function(
    Pointer<Uint8> imageBytes, int width, int height, int stride, int channelFormat);

typedef NormalizeImageBgraC = Pointer<Utf8> Function(
    Pointer<Uint8> inputBytes,
    Int32 inputWidth,
    Int32 inputHeight,
    Int32 inputStride,
    Int32 rotationDegrees,
    Pointer<Uint8> outputBytes,
    Int32 outputBufferSize,
    Int32 channelFormat);
typedef NormalizeImageBgraDart = Pointer<Utf8> Function(
    Pointer<Uint8> inputBytes,
    int inputWidth,
    int inputHeight,
    int inputStride,
    int rotationDegrees,
    Pointer<Uint8> outputBytes,
    int outputBufferSize,
    int channelFormat);

/// The entry point for the table detection isolate.
void tableDetectionIsolateEntry(List<dynamic> args) async {
  final sendPort = args[0] as SendPort;
  final rootToken = args[1] as RootIsolateToken;

  // --- Required for path_provider and other plugins in isolates ---
  BackgroundIsolateBinaryMessenger.ensureInitialized(rootToken);
  // ----------------------------------------------------------------

  final receivePort = ReceivePort();
  sendPort.send(receivePort.sendPort);

  // --- Isolate-local FFI setup ---
  final dylib = LibraryLoader.library;
  final detectTableBgra = dylib
      .lookup<NativeFunction<DetectTableBgraC>>('detect_table_bgra')
      .asFunction<DetectTableBgraDart>();
  final normalizeImageBgra = dylib
      .lookup<NativeFunction<NormalizeImageBgraC>>('normalize_image_bgra')
      .asFunction<NormalizeImageBgraDart>();
  // --- End FFI setup ---

  // --- Pre-allocate native buffers to avoid allocation on every frame ---
  // Input buffer for raw camera image
  final Pointer<Uint8> inputPtr = calloc<Uint8>(3840 * 2160 * 4);
  // Output buffer for normalized image (16:9 canvas)
  final Pointer<Uint8> outputPtr = calloc<Uint8>(3840 * 2160 * 4);

  // Channel format: 0=BGRA (Android), 1=RGBA (iOS)
  final int channelFormat = Platform.isIOS ? 1 : 0;

  receivePort.listen((dynamic message) {
    // Message is now a Map with 'image' and 'capture' keys
    if (message is Map) {
      final AnalysisImage? analysisImage = message['image'] as AnalysisImage?;
      final bool captureRequested = message['capture'] as bool? ?? false;

      if (analysisImage == null) {
        sendPort.send(true);
        return;
      }

      analysisImage.when(bgra8888: (image) {
        Pointer<Utf8> resultPtr = nullptr;
        final totalStopwatch = Stopwatch()..start();

        try {
          final plane = image.planes[0];
          final originalBytes = plane.bytes;
          final originalWidth = image.width;
          final originalHeight = image.height;
          final originalStride = plane.bytesPerRow;

          // Store original dimensions before any transformation
          final capturedWidth = originalWidth;
          final capturedHeight = originalHeight;

          // Determine rotation degrees from camera orientation
          int rotationDegrees = 0;
          if (image.rotation == InputAnalysisImageRotation.rotation90deg) {
            rotationDegrees = 90;
          } else if (image.rotation == InputAnalysisImageRotation.rotation270deg) {
            rotationDegrees = 270;
          } else if (image.rotation == InputAnalysisImageRotation.rotation180deg) {
            rotationDegrees = 180;
          }

          // Copy original image bytes to input buffer
          final inputCopyStopwatch = Stopwatch()..start();
          inputPtr.asTypedList(originalBytes.length).setAll(0, originalBytes);
          inputCopyStopwatch.stop();

          // Call C++ normalize function
          final normalizeStopwatch = Stopwatch()..start();
          final normalizeResult = normalizeImageBgra(
            inputPtr,
            originalWidth,
            originalHeight,
            originalStride,
            rotationDegrees,
            outputPtr,
            3840 * 2160 * 4, // output buffer size
            channelFormat,
          );
          normalizeStopwatch.stop();

          if (normalizeResult == nullptr) {
            sendPort.send(true);
            return;
          }

          // Parse normalization result
          final normalizeJson = normalizeResult.toDartString();
          final Map<String, dynamic> normalizeData = json.decode(normalizeJson);

          if (normalizeData.containsKey('error')) {
            print('[TABLE_ISOLATE] Normalization error: ${normalizeData['error']}');
            sendPort.send(true);
            return;
          }

          final int normalizedWidth = normalizeData['width'] ?? 0;
          final int normalizedHeight = normalizeData['height'] ?? 0;
          final int normalizedStride = normalizeData['stride'] ?? 0;
          final int offsetX = normalizeData['offset_x'] ?? 0;
          final int offsetY = normalizeData['offset_y'] ?? 0;

          // Calculate original image size after rotation but before padding
          // If rotation is 90 or 270, dimensions are swapped
          int unpaddedWidth;
          int unpaddedHeight;
          if (rotationDegrees == 90 || rotationDegrees == 270) {
            unpaddedWidth = capturedHeight;
            unpaddedHeight = capturedWidth;
          } else {
            unpaddedWidth = capturedWidth;
            unpaddedHeight = capturedHeight;
          }

          // Call table detection with normalized image (output is always BGRA after normalization)
          final detectStopwatch = Stopwatch()..start();
          resultPtr = detectTableBgra(
            outputPtr,
            normalizedWidth,
            normalizedHeight,
            normalizedStride,
            1, // After normalization, output is always RGBA
          );
          detectStopwatch.stop();

          if (resultPtr != nullptr) {
            final jsonResult = resultPtr.toDartString();
            final Map<String, dynamic> parsed = json.decode(jsonResult);

            if (parsed.containsKey('error')) {
              // Handle error
            } else {
              // Parse quad points from C++ (in canvas coordinates)
              final List<dynamic> quadPointsJson = parsed['quad_points'] ?? [];
              var canvasQuadPoints = quadPointsJson.map<Offset>((point) {
                return Offset(point[0].toDouble(), point[1].toDouble());
              }).toList();

              final canvasWidth = (parsed['image_width'] ?? normalizedWidth).toDouble();
              final canvasHeight = (parsed['image_height'] ?? normalizedHeight).toDouble();

              // Parse orientation from C++ quad analysis
              final String? orientation = parsed['orientation'];

              // Transform quad points from canvas space to original image space
              final originalQuadPoints = canvasQuadPoints.map((point) {
                return Offset(point.dx - offsetX, point.dy - offsetY);
              }).toList();

              // Parse mask if available
              Uint8List? maskBytes;
              if (parsed.containsKey('mask') && parsed['mask'] is Map) {
                final maskData = parsed['mask']['data'];
                if (maskData is String) {
                  try {
                    maskBytes = base64.decode(maskData);
                  } catch (e) {
                    // Failed to decode mask
                  }
                }
              }

              // Only copy normalized buffer if capture was requested (saves ~11.7MB copy per frame)
              Uint8List? normalizedBytesCopy;
              final int normalizedBufferSize = normalizedWidth * normalizedHeight * 4;
              if (captureRequested) {
                final outputCopyStopwatch = Stopwatch()..start();
                normalizedBytesCopy = Uint8List.fromList(
                  outputPtr.asTypedList(normalizedBufferSize)
                );
                outputCopyStopwatch.stop();
                print('[TABLE_ISOLATE] Captured normalized buffer: ${(normalizedBufferSize / 1024 / 1024).toStringAsFixed(1)}MB in ${outputCopyStopwatch.elapsedMilliseconds}ms');
              }

              totalStopwatch.stop();

              // Send result with points in original image coordinates
              sendPort.send(TableDetectionResult(
                originalQuadPoints,
                Size(canvasWidth, canvasHeight),
                maskBytes: maskBytes,
                canvasOffsetX: offsetX,
                canvasOffsetY: offsetY,
                originalImageSize: Size(unpaddedWidth.toDouble(), unpaddedHeight.toDouble()),
                normalizedBytes: normalizedBytesCopy,
                normalizedWidth: normalizedWidth,
                normalizedHeight: normalizedHeight,
                normalizedStride: normalizedStride,
                orientation: orientation,
              ));
            }
          }
        } catch (e) {
          // Error processing image: $e
        } finally {
          // Signal that we are ready for the next frame
          sendPort.send(true);
        }
      });
    }
  });
}
