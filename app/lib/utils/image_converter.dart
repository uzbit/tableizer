import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

img.Image convertCameraImage(CameraImage frame, {bool assumeNv21 = true}) {
  if (frame.format.group != ImageFormatGroup.yuv420) {
    return img.Image.fromBytes(
      width: frame.width,
      height: frame.height,
      bytes: frame.planes[0].bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
  }

  final int w = frame.width, h = frame.height;
  final yBytes  = frame.planes[0].bytes;
  final uvBytes = frame.planes[1].bytes;
  final vBytes  = frame.planes[2].bytes;

  final yStride     = frame.planes[0].bytesPerRow;
  final uvStride    = frame.planes[1].bytesPerRow;
  final uvPixStride = frame.planes[1].bytesPerPixel ?? 1;

  final img.Image out = img.Image(width: w, height: h, numChannels: 4);

  for (int row = 0; row < h; row++) {
    final int yBase  = row * yStride;
    final int uvBase = (row >> 1) * uvStride;

    final int yRowCap  = yBytes.length  - yBase;
    final int uvRowCap = uvBytes.length - uvBase;   // for NV21/12

    for (int col = 0; col < w; col++) {
      if (col >= yRowCap) break;                    // guard Y

      final int yIdx = yBase + col;

      int u, v;
      if (uvPixStride == 2) {                       // interleaved
        final int uvIdx = uvBase + (col >> 1) * uvPixStride;
        if (uvIdx + 1 >= uvRowCap) break;           // guard UV

        v = assumeNv21 ? uvBytes[uvIdx]     : uvBytes[uvIdx + 1];
        u = assumeNv21 ? uvBytes[uvIdx + 1] : uvBytes[uvIdx];
      } else {                                     // planar (I420)
        final int uvIdx = (row >> 1) * (w >> 1) + (col >> 1);
        if (uvIdx >= uvBytes.length || uvIdx >= vBytes.length) break;
        u = uvBytes[uvIdx];
        v = vBytes[uvIdx];
      }

      int yy = yBytes[yIdx];
      u -= 128; v -= 128;

      int r = (yy + 1.402   * v).round().clamp(0, 255);
      int g = (yy - 0.34414 * u - 0.71414 * v).round().clamp(0, 255);
      int b = (yy + 1.772   * u).round().clamp(0, 255);

      out.setPixelRgba(col, row, r, g, b, 255);
    }
  }
  return out;
}


Future<ui.Image> convertCameraImageToUiImage(CameraImage cameraImage) async {
  final img.Image image = convertCameraImage(cameraImage);
  final Uint8List list = Uint8List.fromList(img.encodePng(image));
  final ui.Codec codec = await ui.instantiateImageCodec(list);
  final ui.FrameInfo frameInfo = await codec.getNextFrame();
  return frameInfo.image;
}