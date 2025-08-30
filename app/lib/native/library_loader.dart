import 'dart:ffi';
import 'dart:io';

/// Loads the native tableizer library for the current platform
class LibraryLoader {
  static DynamicLibrary? _library;
  
  /// Get the loaded native library, loading it if necessary
  static DynamicLibrary get library {
    _library ??= _loadLibrary();
    return _library!;
  }
  
  static DynamicLibrary _loadLibrary() {
    if (Platform.isAndroid) {
      return DynamicLibrary.open('libtableizer_lib.so');
    } else if (Platform.isIOS) {
      return DynamicLibrary.open('libtableizer_lib.dylib');
    } else {
      // Desktop platforms - use process lookup
      return DynamicLibrary.process();
    }
  }
  
  /// Release the library (for cleanup)
  static void release() {
    _library = null;
  }
}