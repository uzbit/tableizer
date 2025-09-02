import 'dart:convert';
import 'dart:ffi' hide Size;
import 'dart:isolate';
import 'dart:ui';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/services.dart' hide Size;

import '../models/table_detection_result.dart';
import '../native/library_loader.dart';

// --- FFI Function Signatures ---
typedef DetectTableBgraC = Pointer<Utf8> Function(Pointer<Uint8> imageBytes, Int32 width,
    Int32 height, Int32 stride, Int32 rotationDegrees, Pointer<Utf8> debugImagePath);
typedef DetectTableBgraDart = Pointer<Utf8> Function(Pointer<Uint8> imageBytes,
    int width, int height, int stride, int rotationDegrees, Pointer<Utf8> debugImagePath);

/// The entry point for the table detection isolate.
void tableDetectionIsolateEntry(List<dynamic> args) async {
  final sendPort = args[0] as SendPort;
  final rootToken = args[1] as RootIsolateToken;

  // --- Required for path_provider and other plugins in isolates ---
  BackgroundIsolateBinaryMessenger.ensureInitialized(rootToken);
  // ----------------------------------------------------------------

  print('[TABLE_ISOLATE] Table detection isolate started.');
  final receivePort = ReceivePort();
  sendPort.send(receivePort.sendPort);

  // --- Isolate-local FFI setup ---
  // This needs to be loaded in the isolate separately because
  // it has it's own memory space.
  final dylib = LibraryLoader.library;

  final detectTableBgra = dylib
      .lookup<NativeFunction<DetectTableBgraC>>('detect_table_bgra')
      .asFunction<DetectTableBgraDart>();
  // --- End FFI setup ---

  int frameCount = 0;

  // --- Pre-allocate a native buffer to avoid allocation on every frame ---
  final Pointer<Uint8> imagePtr = calloc<Uint8>(3840 * 2160 * 4);

  receivePort.listen((dynamic message) {
    if (message is AnalysisImage) {
      message.when(bgra8888: (image) {
        Pointer<Utf8> resultPtr = nullptr;
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
            final jsonResult = resultPtr.toDartString();
            final Map<String, dynamic> parsed = json.decode(jsonResult);
            
            if (parsed.containsKey('error')) {
              print('[TABLE_ISOLATE] Error from native: ${parsed['error']}');
            } else {
              final List<dynamic> quadPointsJson = parsed['quad_points'] ?? [];
              final quadPoints = quadPointsJson.map<Offset>((point) {
                return Offset(point['x'].toDouble(), point['y'].toDouble());
              }).toList();
              
              // Send the complete result object back
              sendPort.send(TableDetectionResult(
                quadPoints,
                Size(
                  (parsed['image_width'] ?? image.width).toDouble(),
                  (parsed['image_height'] ?? image.height).toDouble(),
                ),
              ));
            }
          }
        } catch (e) {
          print('[TABLE_ISOLATE] Error processing image: $e');
        } finally {
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
