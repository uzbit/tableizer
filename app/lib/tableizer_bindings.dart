import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

// ignore_for_file: unused_import

final class Detection extends ffi.Struct {
  @ffi.Float()
  external double x;

  @ffi.Float()
  external double y;

  @ffi.Float()
  external double radius;

  @ffi.Int32()
  external int class_id;
}

class TableizerBindings {
  /// The dynamic library in which the symbols for [TableizerBindings] can be found.
  final ffi.DynamicLibrary _dylib;

  /// The symbols are looked up in [dynamicLibrary].
  TableizerBindings(ffi.DynamicLibrary dynamicLibrary) : _dylib = dynamicLibrary;

  late final ffi.Pointer<
      ffi.NativeFunction<
          ffi.Pointer<ffi.Void> Function(
              ffi.Pointer<Utf8>)>> create_ball_detector = _dylib.lookup<
      ffi.NativeFunction<
          ffi.Pointer<ffi.Void> Function(ffi.Pointer<Utf8>)>>('create_ball_detector');

  late final ffi.Pointer<
          ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
      destroy_ball_detector = _dylib.lookup<
          ffi.NativeFunction<
              ffi.Void Function(ffi.Pointer<ffi.Void>)>>('destroy_ball_detector');

  late final ffi.Pointer<
      ffi.NativeFunction<
          ffi.Pointer<Detection> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Uint8>,
              ffi.Pointer<ffi.Uint8>,
              ffi.Pointer<ffi.Uint8>,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Float,
              ffi.Float,
              ffi.Pointer<ffi.Int32>)>> detect_balls = _dylib.lookup<
      ffi.NativeFunction<
          ffi.Pointer<Detection> Function(
              ffi.Pointer<ffi.Void>,
              ffi.Pointer<ffi.Uint8>,
              ffi.Pointer<ffi.Uint8>,
              ffi.Pointer<ffi.Uint8>,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Int32,
              ffi.Float,
              ffi.Float,
              ffi.Pointer<ffi.Int32>)>>('detect_balls');

  late final ffi.Pointer<
          ffi.NativeFunction<ffi.Void Function(ffi.Pointer<Detection>)>>
      free_detections = _dylib.lookup<
          ffi.NativeFunction<
              ffi.Void Function(ffi.Pointer<Detection>)>>('free_detections');
}
