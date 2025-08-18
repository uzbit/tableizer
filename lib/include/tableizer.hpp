#ifndef TABLEIZER_HPP
#define TABLEIZER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "ball_detector.hpp"

// A placeholder for the final library API.
using namespace cv;

struct Ball {
    Point2f position;
    int classId;
};

class TableState {
   public:
    Mat getWarpedTable() const;
    std::vector<Ball> getBalls() const;

   private:
    Mat warpedTable;
    std::vector<Ball> balls;
};

int runTableizerForImage(Mat image, BallDetector& ballDetector);

// FFI entry points
#ifdef __cplusplus
extern "C" {
#endif

// Represents a single point.
struct FFI_Point {
    float x;
    float y;
};

// The struct that will be returned from the native code.
// The caller is responsible for freeing this struct using `free_detection_result`.
struct DetectionResult {
    FFI_Point quad_points[4];
    int quad_points_count;
};

__attribute__((visibility("default"))) __attribute__((used)) void* initialize_detector(
    const char* model_path);

__attribute__((visibility("default"))) __attribute__((used)) const char* detect_objects_rgba(
    void* detector_ptr, const unsigned char* image_bytes, int width, int height, int channels);

__attribute__((visibility("default"))) __attribute__((used)) const char* detect_table_rgba(
    const unsigned char* image_bytes, int width, int height, int channels, int stride);

// High-performance, zero-copy alternative to detect_table_rgba.
// Allocates a DetectionResult struct that must be freed by the caller.
__attribute__((visibility("default"))) __attribute__((used)) DetectionResult* detect_table_bgra(
    const unsigned char* image_bytes, int width, int height, int stride,
    const char* debug_image_path);

// Frees the memory allocated by detect_table_bgra.
__attribute__((visibility("default"))) __attribute__((used)) void free_bgra_detection_result(
    DetectionResult* result);

__attribute__((visibility("default"))) __attribute__((used)) void release_detector(
    void* detector_ptr);

#ifdef __cplusplus
}
#endif

#endif  // TABLEIZER_HPP
