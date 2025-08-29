import 'package:flutter/material.dart';
import 'package:tableizer_detection/tableizer_detection.dart';
import 'package:camerawesome/camerawesome_plugin.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tableizer Detection Example',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: TableDetectionScreen(),
    );
  }
}

class TableDetectionScreen extends StatefulWidget {
  @override
  _TableDetectionScreenState createState() => _TableDetectionScreenState();
}

class _TableDetectionScreenState extends State<TableDetectionScreen> {
  late TableDetectionController _tableController;
  late BallDetectionController _ballController;
  
  // Custom configuration
  final DetectionConfig config = DetectionConfig(
    quadAlpha: 0.2,  // Smoother quad filtering
    confidenceThreshold: 0.7,  // Higher confidence for balls
    deltaEThreshold: 20.0,  // More sensitive table detection
    maxFramesPerSecond: 25,  // Slightly lower FPS for performance
  );
  
  @override
  void initState() {
    super.initState();
    _tableController = TableDetectionController();
    _ballController = BallDetectionController();
    _initializeControllers();
  }
  
  Future<void> _initializeControllers() async {
    await _tableController.initialize();
    await _ballController.initialize();
  }
  
  @override
  void dispose() {
    _tableController.dispose();
    _ballController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Pool Table Detection'),
      ),
      body: CameraAwesomeBuilder.custom(
        saveConfig: SaveConfig.photoAndVideo(),
        sensorConfig: SensorConfig.single(
          sensor: Sensor.position(SensorPosition.back),
          flashMode: FlashMode.none,
          aspectRatio: CameraAspectRatios.ratio_4_3,
        ),
        onImageForAnalysis: _tableController.processImage,
        imageAnalysisConfig: AnalysisConfig(
          androidOptions: AndroidAnalysisOptions.bgra8888(width: 1280),
          maxFramesPerSecond: config.maxFramesPerSecond,
        ),
        builder: (cameraState, preview) {
          return AnimatedBuilder(
            animation: _tableController,
            builder: (context, child) {
              return Stack(
                fit: StackFit.expand,
                children: [
                  preview,
                  
                  // Table detection overlay
                  if (_tableController.quadPoints.isNotEmpty)
                    CustomPaint(
                      painter: TableOverlayPainter(
                        quadPoints: _tableController.quadPoints,
                        imageSize: _tableController.imageSize,
                      ),
                    ),
                  
                  // Detection info overlay
                  Positioned(
                    top: 50,
                    left: 20,
                    child: Container(
                      padding: EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        'FPS: ${_tableController.fps.toStringAsFixed(1)}\\n'
                        'Detection: ${_tableController.isEnabled ? "ON" : "OFF"}',
                        style: TextStyle(color: Colors.white, fontSize: 12),
                      ),
                    ),
                  ),
                ],
              );
            },
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          // Toggle table detection
          _tableController.setEnabled(!_tableController.isEnabled);
        },
        child: Icon(
          _tableController.isEnabled ? Icons.pause : Icons.play_arrow,
        ),
      ),
    );
  }
}

// Simple overlay painter for demonstration
class TableOverlayPainter extends CustomPainter {
  final List<Offset> quadPoints;
  final ui.Size? imageSize;
  
  TableOverlayPainter({
    required this.quadPoints,
    this.imageSize,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    if (quadPoints.length != 4) return;
    
    final paint = Paint()
      ..color = Colors.green
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    
    final path = Path();
    path.moveTo(quadPoints[0].dx, quadPoints[0].dy);
    for (int i = 1; i < quadPoints.length; i++) {
      path.lineTo(quadPoints[i].dx, quadPoints[i].dy);
    }
    path.close();
    
    canvas.drawPath(path, paint);
  }
  
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}