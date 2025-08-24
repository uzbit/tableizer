import 'dart:math' as math;

class SettingsService {
  static final SettingsService _instance = SettingsService._internal();
  factory SettingsService() => _instance;
  SettingsService._internal();

  // Table size setting
  String _tableSize = '9-foot';

  String get tableSize => _tableSize;
  
  void setTableSize(String size) {
    _tableSize = size;
  }

  // Convert table size to inches (playing surface length)
  double get tableSizeInches {
    switch (_tableSize) {
      case '9-foot':
        return 100.0; // Standard 9-foot table playing surface
      case '8-foot':
        return 88.0;  // 8-foot table playing surface
      case '7-foot':
        return 76.0;  // 7-foot table playing surface
      default:
        return 100.0; // Default to 9-foot
    }
  }
}

class BallScaling {
  static const double BALL_DIAMETER_INCHES = 2.25;
  static const double MIN_RADIUS_PX = 4.0;
  
  static double calculateBallRadius(double displayWidth, double displayHeight, double tableSizeInches) {
    // Find the longest edge of the canonical table display
    double longEdgePx = math.max(displayWidth, displayHeight);
    
    // Calculate radius using the same formula as C++
    double radius = longEdgePx * (BALL_DIAMETER_INCHES / tableSizeInches) / 2.0;
    
    // Apply minimum radius constraint
    return math.max(radius, MIN_RADIUS_PX);
  }
  
  static double calculateTextSize(double radius) {
    return 0.7 * (radius / 8.0);
  }
}