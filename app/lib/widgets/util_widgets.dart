import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;

/// Pops up the converted frame; returns when user taps "Continue".
Future<void> showFrameDebug(BuildContext context, img.Image bgra) async {
  final Uint8List pngBytes = Uint8List.fromList(img.encodePng(bgra));

  await showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (_) => AlertDialog(
      content: Image.memory(pngBytes),          // just the image
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Continue'),
        ),
      ],
    ),
  );
}