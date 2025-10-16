# Tableizer

## Overview
Tableizer combines a native C++ vision engine, a Flutter capture app, and Python tooling to detect pool tables, locate balls, and map their positions onto ShotStudio-style overlays.

## Native Core
The C++ library in `lib/src` wraps ONNX Runtime YOLO ball detection and quad analysis, exposing FFI entry points such as `detect_table_bgra` and `transform_points_using_quad`. It drives table warping, ball inference, and compositing, with builds orchestrated through the CMake setup and platform scripts under `lib/`.

## Mobile App
The Flutter client in `app/lib` relies on CamerAwesome for live preview, captures BGRA frames, and routes them through FFI isolates for real-time quad tracking before presenting analysis results. Controllers manage detection streams, smooth quad jitter, and communicate with the native library.

## Python Tooling
Python utilities deliver FFI bindings, dataset transforms, training helpers, and debugging pipelines that call the same native functions. The environment is defined in `requirements.txt`, covering Ultralytics, Torch, OpenCV, and supporting packages for experimentation.

## Models and Data
Trained YOLO weights and experiment artifacts reside under `tableizer/exp*`, while `data/` stores raw imagery, transformed sets, and ShotStudio background assets consumed across the stack.

## Notable Issues
Multiple modules reference hard-coded absolute paths, which limits portability. Centralized configuration or environment-driven paths would improve maintainability.
