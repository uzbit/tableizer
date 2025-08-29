/// Configuration class for table and ball detection parameters
class DetectionConfig {
  /// Table detection settings
  final int resizeHeight;
  final int cellSize;
  final double deltaEThreshold;
  final double quadAlpha; // Alpha filter smoothing factor
  
  /// Ball detection settings  
  final double confidenceThreshold;
  final double iouThreshold;
  final String modelPath;
  
  /// Performance settings
  final int maxFramesPerSecond;
  final bool enableTableDetection;
  final bool enableBallDetection;
  
  const DetectionConfig({
    // Table detection defaults
    this.resizeHeight = 800,
    this.cellSize = 10,
    this.deltaEThreshold = 15.0,
    this.quadAlpha = 0.3,
    
    // Ball detection defaults
    this.confidenceThreshold = 0.6,
    this.iouThreshold = 0.5,
    this.modelPath = 'assets/detection_model.onnx',
    
    // Performance defaults
    this.maxFramesPerSecond = 30,
    this.enableTableDetection = true,
    this.enableBallDetection = true,
  });
  
  DetectionConfig copyWith({
    int? resizeHeight,
    int? cellSize,
    double? deltaEThreshold,
    double? quadAlpha,
    double? confidenceThreshold,
    double? iouThreshold,
    String? modelPath,
    int? maxFramesPerSecond,
    bool? enableTableDetection,
    bool? enableBallDetection,
  }) {
    return DetectionConfig(
      resizeHeight: resizeHeight ?? this.resizeHeight,
      cellSize: cellSize ?? this.cellSize,
      deltaEThreshold: deltaEThreshold ?? this.deltaEThreshold,
      quadAlpha: quadAlpha ?? this.quadAlpha,
      confidenceThreshold: confidenceThreshold ?? this.confidenceThreshold,
      iouThreshold: iouThreshold ?? this.iouThreshold,
      modelPath: modelPath ?? this.modelPath,
      maxFramesPerSecond: maxFramesPerSecond ?? this.maxFramesPerSecond,
      enableTableDetection: enableTableDetection ?? this.enableTableDetection,
      enableBallDetection: enableBallDetection ?? this.enableBallDetection,
    );
  }
}