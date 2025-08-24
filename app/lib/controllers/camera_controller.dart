import 'dart:typed_data';

import 'package:camerawesome/camerawesome_plugin.dart';
import 'package:flutter/material.dart';

class CameraController extends ChangeNotifier {
  // Image capture state
  bool _showCaptureMessage = false;
  Uint8List? _capturedImageBytes;
  CameraState? _cameraState;

  // Getters
  bool get showCaptureMessage => _showCaptureMessage;
  Uint8List? get capturedImageBytes => _capturedImageBytes;
  CameraState? get cameraState => _cameraState;

  void setCameraState(CameraState state) {
    _cameraState = state;
  }

  Future<void> capturePhoto() async {
    if (_cameraState == null) return;
    
    _cameraState!.when(
      onPhotoMode: (photoState) async {
        await photoState.takePhoto();
      },
      onVideoMode: (videoState) {
        print('Switch to photo mode to capture image');
      },
      onVideoRecordingMode: (videoRecordingState) {
        print('Stop recording and switch to photo mode to capture image');
      },
    );
  }

  void onMediaCaptured(CaptureRequest request) async {
    request.when(
      single: (singleRequest) async {
        final file = singleRequest.file;
        if (file != null) {
          final imageBytes = await file.readAsBytes();
          _capturedImageBytes = imageBytes;
          _showCaptureMessage = true;
          notifyListeners();
        }
      },
      multiple: (multipleRequest) {
        print('Multiple camera capture not implemented');
      },
    );
  }

  void clearCapturedImage() {
    _capturedImageBytes = null;
    _showCaptureMessage = false;
    notifyListeners();
  }

  @override
  void dispose() {
    super.dispose();
  }
}