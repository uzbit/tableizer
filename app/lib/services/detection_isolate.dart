import 'dart:ffi' hide Size;
import 'dart:io';
import 'dart:isolate';
import 'dart:ui';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart' hide Size;
import 'package:path_provider/path_provider.dart';

import 'table_detection_result.dart';
import 'table_detection_service.dart';

/// The entry point for the detection isolate.
void detectionIsolateEntry(List<dynamic> args) async {
  final sendPort = args[0] as SendPort;
  final rootToken = args[1] as RootIsolateToken;

  // --- Required for path_provider and other plugins in isolates ---
  BackgroundIsolateBinaryMessenger.ensureInitialized(rootToken);
  // ----------------------------------------------------------------

  print('[ISOLATE] Isolate started.');
  final receivePort = ReceivePort();
  sendPort.send(receivePort.sendPort);

  // --- Isolate-local FFI setup ---
  final dylib = Platform.isAndroid
      ? DynamicLibrary.open('libtableizer_lib.so')
      : DynamicLibrary.process();

  final detectTableBgra = dylib
      .lookup<NativeFunction<DetectTableBgraC>>('detect_table_bgra')
      .asFunction<DetectTableBgraDart>();
  final freeBgraDetectionResult = dylib
      .lookup<NativeFunction<FreeBgraDetectionResultC>>('free_bgra_detection_result')
      .asFunction<FreeBgraDetectionResultDart>();
  // --- End FFI setup ---

  int frameCount = 0;
  final tempDir = await getTemporaryDirectory();

  // --- Pre-allocate a native buffer to avoid allocation on every frame ---
  final Pointer<Uint8> imagePtr = calloc<Uint8>(3840 * 2160 * 4);

  receivePort.listen((dynamic message) {
    if (message is AnalysisImage) {
      message.when(bgra8888: (image) {
        Pointer<DetectionResult> resultPtr = nullptr;
        Pointer<Utf8> pathPtr = nullptr;

        try {
          final plane = image.planes[0];
          final bytes = plane.bytes;

          imagePtr.asTypedList(bytes.length).setAll(0, bytes);

          // String debugImagePath = '';
          // if (frameCount % 150 == 0) {
          //   debugImagePath = '${tempDir.path}/native_debug_$frameCount.png';
          // }
          // pathPtr = debugImagePath.toNativeUtf8();
          pathPtr = nullptr;

          int rotationDegrees = 0;
          switch (image.rotation) {
            case InputAnalysisImageRotation.rotation90deg:
              rotationDegrees = 90;
              break;
            case InputAnalysisImageRotation.rotation180deg:
              rotationDegrees = 180;
              break;
            case InputAnalysisImageRotation.rotation270deg:
              rotationDegrees = 270;
              break;
            default:
              rotationDegrees = 0;
          }

          resultPtr = detectTableBgra(
            imagePtr,
            image.width,
            image.height,
            plane.bytesPerRow,
            rotationDegrees,
            pathPtr,
          );

          frameCount++;

          if (resultPtr != nullptr) {
            final result = resultPtr.ref;
            final quadPoints = <Offset>[];
            for (var i = 0; i < result.quad_points_count; i++) {
              quadPoints.add(Offset(
                result.quad_points[i].x,
                result.quad_points[i].y,
              ));
            }
            // Send the complete result object back
            sendPort.send(TableDetectionResult(
              quadPoints,
              Size(
                result.image_width.toDouble(),
                result.image_height.toDouble(),
              ),
            ));
          }
        } catch (e) {
          print('[ISOLATE] Error processing image: $e');
        } finally {
          if (resultPtr != nullptr) {
            freeBgraDetectionResult(resultPtr);
          }
          if (pathPtr != nullptr) {
            calloc.free(pathPtr);
          }
          // Signal that we are ready for the next frame
          sendPort.send(true);
        }
      });
    }
  });
}
