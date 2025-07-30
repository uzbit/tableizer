#ifndef TABLEIZER_FFI_H
#define TABLEIZER_FFI_H

#include <cstdint>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default"))) __attribute__((used))
#endif

extern "C" {

// Represents a single detection.
// The definition is in ball_detector.hpp

// Creates a BallDetector instance and returns a pointer to it.
// The model_path is the absolute path to the ONNX model file.
EXPORT void* create_ball_detector(const char* model_path);

// Destroys a BallDetector instance.
EXPORT void destroy_ball_detector(void* detector);

// Performs ball detection on an image.
// The image is provided as a byte array in YUV420 format.
// Returns a pointer to an array of Detection structs.
// The number of detections is returned in the detection_count output parameter.
EXPORT Detection* detect_balls(void* detector, uint8_t* plane0, uint8_t* plane1, uint8_t* plane2,
                               int width, int height, int p0_stride, int p1_stride, int p2_stride,
                               int p1_pix_stride, int p2_pix_stride, float conf_threshold,
                               float iou_threshold, int* detection_count);

// Frees the memory allocated for the array of detections.
EXPORT void free_detections(Detection* detections);
}

#endif  // TABLEIZER_FFI_H
