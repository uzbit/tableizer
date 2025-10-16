import 'dart:typed_data';
import 'dart:math' as math;
import 'package:image/image.dart' as img;

/// Normalized image data ready for C++ FFI processing
class NormalizedImageData {
  final Uint8List bytes;
  final int width;
  final int height;
  final int stride;
  final bool wasRotated;
  final int offsetX; // Offset where original image starts in canvas
  final int offsetY; // Offset where original image starts in canvas

  NormalizedImageData({
    required this.bytes,
    required this.width,
    required this.height,
    required this.stride,
    required this.wasRotated,
    this.offsetX = 0,
    this.offsetY = 0,
  });
}

/// Adapter to normalize images to landscape BGRA format for C++ FFI
class ImageAdapter {
  static const double TARGET_ASPECT_RATIO = 16.0 / 9.0;

  /// Converts any image to landscape BGRA format on a 16:9 black canvas
  ///
  /// This ensures C++ code always receives:
  /// - BGRA color format (4 channels)
  /// - Landscape 16:9 aspect ratio
  /// - Image centered on black canvas
  /// - Dense stride (width * 4)
  static NormalizedImageData normalize({
    required Uint8List bytes,
    required int width,
    required int height,
    int? stride,
    int channels = 4,
  }) {
    // Create image from bytes
    img.Image image = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: bytes.buffer,
      format: img.Format.uint8,
      numChannels: channels,
    );

    // Determine canvas dimensions (16:9 landscape)
    // Use the larger dimension as the height reference
    final int canvasHeight = math.max(width, height);
    final int canvasWidth = (canvasHeight * TARGET_ASPECT_RATIO).round();

    // Create black canvas
    final img.Image canvas = img.Image(
      width: canvasWidth,
      height: canvasHeight,
      numChannels: 4,
    );

    // Fill with black (RGBA: 0, 0, 0, 255)
    img.fill(canvas, color: img.ColorRgba8(0, 0, 0, 255));

    // Calculate centered position
    final int offsetX = ((canvasWidth - width) / 2).round();
    final int offsetY = ((canvasHeight - height) / 2).round();

    // Composite original image onto canvas
    img.compositeImage(canvas, image, dstX: offsetX, dstY: offsetY);

    // Convert RGBA to BGRA (swap R and B channels)
    final Uint8List bgraBytes = _convertRGBAtoBGRA(canvas);

    return NormalizedImageData(
      bytes: bgraBytes,
      width: canvasWidth,
      height: canvasHeight,
      stride: canvasWidth * 4,
      wasRotated: false,
      offsetX: offsetX,
      offsetY: offsetY,
    );
  }

  /// Converts RGBA bytes to BGRA bytes (swap R and B channels)
  static Uint8List _convertRGBAtoBGRA(img.Image image) {
    final int width = image.width;
    final int height = image.height;
    final Uint8List rgba = image.toUint8List();
    final Uint8List bgra = Uint8List(rgba.length);

    for (int i = 0; i < rgba.length; i += 4) {
      bgra[i] = rgba[i + 2];     // B = R
      bgra[i + 1] = rgba[i + 1]; // G = G
      bgra[i + 2] = rgba[i];     // R = B
      bgra[i + 3] = rgba[i + 3]; // A = A
    }

    return bgra;
  }

  /// Normalizes image from camera input
  ///
  /// Handles rotation based on camera orientation, then places on 16:9 canvas
  static NormalizedImageData normalizeFromCamera({
    required Uint8List bytes,
    required int width,
    required int height,
    required int stride,
    required int rotationDegrees, // 0, 90, 180, 270
  }) {
    // First convert to standard format
    img.Image image = img.Image.fromBytes(
      width: width,
      height: height,
      bytes: bytes.buffer,
      format: img.Format.uint8,
      numChannels: 4,
    );

    // Apply camera rotation to correct orientation
    bool wasRotated = false;
    if (rotationDegrees == 90) {
      image = img.copyRotate(image, angle: 90);
      wasRotated = true;
    } else if (rotationDegrees == 270 || rotationDegrees == -90) {
      image = img.copyRotate(image, angle: -90);
      wasRotated = true;
    } else if (rotationDegrees == 180) {
      image = img.copyRotate(image, angle: 180);
      wasRotated = true;
    }

    // Determine canvas dimensions (16:9 landscape)
    // Use the larger dimension as the height reference
    final int canvasHeight = math.max(image.width, image.height);
    final int canvasWidth = (canvasHeight * TARGET_ASPECT_RATIO).round();

    // Create black canvas
    final img.Image canvas = img.Image(
      width: canvasWidth,
      height: canvasHeight,
      numChannels: 4,
    );

    // Fill with black (RGBA: 0, 0, 0, 255)
    img.fill(canvas, color: img.ColorRgba8(0, 0, 0, 255));

    // Calculate centered position
    final int offsetX = ((canvasWidth - image.width) / 2).round();
    final int offsetY = ((canvasHeight - image.height) / 2).round();

    // Composite rotated image onto canvas
    img.compositeImage(canvas, image, dstX: offsetX, dstY: offsetY);

    // Convert RGBA to BGRA
    final Uint8List bgraBytes = _convertRGBAtoBGRA(canvas);

    return NormalizedImageData(
      bytes: bgraBytes,
      width: canvasWidth,
      height: canvasHeight,
      stride: canvasWidth * 4,
      wasRotated: wasRotated,
      offsetX: offsetX,
      offsetY: offsetY,
    );
  }
}
