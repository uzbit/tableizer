import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

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

  receivePort.listen((dynamic message) {
    if (message is AnalysisImage) {
      message.when(bgra8888: (image) {
        Pointer<DetectionResult> resultPtr = nullptr;
        Pointer<Uint8> imagePtr = nullptr;
        Pointer<Utf8> pathPtr = nullptr;

        try {
          final plane = image.planes[0];
          final bytes = plane.bytes;

          imagePtr = calloc<Uint8>(bytes.length);
          imagePtr.asTypedList(bytes.length).setAll(0, bytes);

          // --- Native Debug Image Saving ---
          // We only save the image periodically to avoid spamming the disk.
          String debugImagePath = '';
          if (frameCount % 150 == 0) {
            debugImagePath = '${tempDir.path}/native_debug_$frameCount.png';
            print('[ISOLATE] Requesting native debug image at: $debugImagePath');
          }
          pathPtr = debugImagePath.toNativeUtf8();
          // --- End Native Debug Image Saving ---

          resultPtr = detectTableBgra(
            imagePtr,
            image.width,
            image.height,
            plane.bytesPerRow,
            pathPtr,
          );

          if (resultPtr != nullptr) {
            final result = resultPtr.ref;
            final quadPoints = <Offset>[];
            for (var i = 0; i < result.quad_points_count; i++) {
              quadPoints.add(Offset(
                result.quad_points[i].x,
                result.quad_points[i].y,
              ));
            }
            sendPort.send(quadPoints);
          }
        } catch (e) {
          print('[ISOLATE] Error processing image: $e');
        } finally {
          if (resultPtr != nullptr) {
            freeBgraDetectionResult(resultPtr);
          }
          if (imagePtr != nullptr) {
            calloc.free(imagePtr);
          }
          if (pathPtr != nullptr) {
            calloc.free(pathPtr);
          }
        }
      });
      frameCount++;
    }
  });
}
