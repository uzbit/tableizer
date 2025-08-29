library tableizer_detection;

// Export main public API
export 'src/controllers/table_detection_controller.dart';
export 'src/controllers/ball_detection_controller.dart';

export 'src/services/table_detection_service.dart';
export 'src/services/ball_detection_service.dart';

export 'src/models/table_detection_result.dart';
export 'src/models/ball_detection.dart';
export 'src/models/detection_config.dart';

// Export native library utilities
export 'src/native/library_loader.dart';

// Export widgets if we create any
// export 'src/widgets/table_overlay_widget.dart';
// export 'src/widgets/ball_overlay_widget.dart';