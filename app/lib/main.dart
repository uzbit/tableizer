import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

// --- FFI Signatures ---
typedef InitializeDetectorC = Pointer<Void> Function(Pointer<Utf8> modelPath);
typedef InitializeDetectorDart = Pointer<Void> Function(Pointer<Utf8> modelPath);

typedef DetectObjectsC = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, Int32 width, Int32 height, Int32 channels);
typedef DetectObjectsDart = Pointer<Utf8> Function(Pointer<Void> detector, Pointer<Uint8> imageBytes, int width, int height, int channels);

typedef ReleaseDetectorC = Void Function(Pointer<Void> detector);
typedef ReleaseDetectorDart = void Function(Pointer<Void> detector);


void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(
    MaterialApp(
      theme: ThemeData.dark(),
      home: CameraScreen(camera: firstCamera),
    ),
  );
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.camera});
  final CameraDescription camera;

  @override
  CameraScreenState createState() => CameraScreenState();
}

class CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  Future<void>? _initializeControllerFuture;

  Pointer<Void> _detector = nullptr;
  late InitializeDetectorDart _initializeDetector;
  late DetectObjectsDart _detectObjects;
  late ReleaseDetectorDart _releaseDetector;

  bool _isDetecting = false;

  @override
  void initState() {
    super.initState();
    _initializeControllerFuture = _initializeEverything();
  }

  Future<void> _initializeEverything() async {
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    await _loadLibrary();
    await _controller.initialize();
    if (mounted) {
      await _controller.startImageStream(_processCameraImage);
    }
  }

  Future<void> _loadLibrary() async {
    final dylib = Platform.isAndroid
        ? DynamicLibrary.open("libtableizer_lib.so")
        : DynamicLibrary.process();

    _initializeDetector = dylib
        .lookup<NativeFunction<InitializeDetectorC>>('initialize_detector')
        .asFunction();
    _detectObjects = dylib
        .lookup<NativeFunction<DetectObjectsC>>('detect_objects')
        .asFunction();
    _releaseDetector = dylib
        .lookup<NativeFunction<ReleaseDetectorC>>('release_detector')
        .asFunction();

    final modelPath = await _getAssetPath('assets/detection_model.onnx');
    _detector = _initializeDetector(modelPath.toNativeUtf8());
  }

  Future<String> _getAssetPath(String asset) async {
    final path = p.join((await getApplicationSupportDirectory()).path, asset);
    await Directory(p.dirname(path)).create(recursive: true);
    final file = File(path);
    if (!await file.exists()) {
      final byteData = await rootBundle.load(asset);
      await file.writeAsBytes(byteData.buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }
    return path;
  }

  @override
  void dispose() {
    if (_controller.value.isStreamingImages) {
      _controller.stopImageStream();
    }
    _controller.dispose();
    if (_detector != nullptr) {
      _releaseDetector(_detector);
    }
    super.dispose();
  }

  void _processCameraImage(CameraImage image) {
    if (_isDetecting || _detector == nullptr) return;

    setState(() {
      _isDetecting = true;
    });

    Isolate.spawn(_detectionIsolate, {
      'detector_ptr': _detector.address,
      'camera_image': image,
    }).then((results) {
       if (mounted) {
        setState(() {
          print("Camera detection results: $results"); 
          _isDetecting = false;
        });
      }
    });
  }

  Future<void> _showConfirmationScreen() async {
    // Load image from assets and navigate to the confirmation screen
    final byteData = await rootBundle.load('assets/images/P_20250718_203819.jpg');
    final imageBytes = byteData.buffer.asUint8List();
    
    if (!mounted) return;

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => DisplayPictureScreen(
          imageBytes: imageBytes,
          detector: _detector,
          detectObjects: _detectObjects,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Tableizer')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            if (snapshot.hasError) {
              return Center(child: Text('Initialization Error: ${snapshot.error}'));
            }
            return CameraPreview(_controller);
          } else {
            return const Center(child: CircularProgressIndicator());
          }
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          if (_detector != nullptr) {
            _showConfirmationScreen();
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Detector not initialized yet.')),
            );
          }
        },
        tooltip: 'Process Local Image',
        child: const Icon(Icons.image),
      ),
    );
  }
}

// --- Confirmation Screen ---
class DisplayPictureScreen extends StatelessWidget {
  final Uint8List imageBytes;
  final Pointer<Void> detector;
  final DetectObjectsDart detectObjects;

  const DisplayPictureScreen({
    super.key,
    required this.imageBytes,
    required this.detector,
    required this.detectObjects,
  });

  Future<void> _processImage(BuildContext context) async {
    final img.Image? decodedImage = img.decodeImage(imageBytes);

    if (decodedImage == null) {
      print("Failed to decode image from assets");
      return;
    }
    
    final img.Image rgbaImage = img.Image(width: decodedImage.width, height: decodedImage.height);
    for (int y = 0; y < decodedImage.height; ++y) {
        for (int x = 0; x < decodedImage.width; ++x) {
            final pixel = decodedImage.getPixel(x, y);
            rgbaImage.setPixelRgba(x, y, pixel.r.toInt(), pixel.g.toInt(), pixel.b.toInt(), pixel.a.toInt());
        }
    }

    final imageBytesPtr = calloc<Uint8>(rgbaImage.lengthInBytes);
    imageBytesPtr.asTypedList(rgbaImage.lengthInBytes).setAll(0, rgbaImage.getBytes(order: img.ChannelOrder.rgba));

    final resultPtr = detectObjects(detector, imageBytesPtr, rgbaImage.width, rgbaImage.height, 4);
    final resultJson = resultPtr.toDartString();

    calloc.free(imageBytesPtr);

    print("Local image detection results: $resultJson");

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Detection complete! Check console for results.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Confirm Detection')),
      body: Column(
        children: [
          Expanded(child: Image.memory(imageBytes)),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed: () => _processImage(context),
              child: const Text('Detect'),
            ),
          ),
        ],
      ),
    );
  }
}


void _detectionIsolate(Map<String, dynamic> context) {
  final detectorPtr = Pointer<Void>.fromAddress(context['detector_ptr']);
  final CameraImage image = context['camera_image'];

  final img.Image convertedImage = _convertCameraImage(image);
  
  final imageBytes = convertedImage.getBytes(order: img.ChannelOrder.rgba);
  if (convertedImage.lengthInBytes != imageBytes.lengthInBytes) {
    Isolate.exit(Isolate.current.controlPort, '{"error": "Image buffer size mismatch"}');
    return;
  }

  final imageBytesPtr = calloc<Uint8>(convertedImage.lengthInBytes);
  imageBytesPtr.asTypedList(convertedImage.lengthInBytes).setAll(0, imageBytes);

  final dylib = Platform.isAndroid
      ? DynamicLibrary.open("libtableizer_lib.so")
      : DynamicLibrary.process();
  final detectObjects = dylib
      .lookup<NativeFunction<DetectObjectsC>>('detect_objects')
      .asFunction<DetectObjectsDart>();

  final resultPtr = detectObjects(detectorPtr, imageBytesPtr, convertedImage.width, convertedImage.height, 4);
  final resultJson = resultPtr.toDartString();
  
  calloc.free(imageBytesPtr);

  Isolate.exit(Isolate.current.controlPort, resultJson);
}

img.Image _convertCameraImage(CameraImage cameraImage) {
  if (cameraImage.format.group == ImageFormatGroup.yuv420) {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

    final image = img.Image(width: width, height: height, numChannels: 4);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
        final int index = y * width + x;

        final yp = cameraImage.planes[0].bytes[index];
        final up = cameraImage.planes[1].bytes[uvIndex];
        final vp = cameraImage.planes[2].bytes[uvIndex];
        
        int r = (yp + vp * 1.402).round().clamp(0, 255);
        int g = (yp - up * 0.344 - vp * 0.714).round().clamp(0, 255);
        int b = (yp + up * 1.772).round().clamp(0, 255);

        image.setPixelRgba(x, y, r, g, b, 255);
      }
    }
    return image;
  } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
     return img.Image.fromBytes(
        width: cameraImage.width,
        height: cameraImage.height,
        bytes: cameraImage.planes[0].bytes.buffer,
        order: img.ChannelOrder.bgra,
     );
  } else {
    throw Exception('Unsupported image format: ${cameraImage.format.group}');
  }
}
