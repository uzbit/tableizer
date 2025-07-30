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

int runTableizerForImage(Mat image, BallDetector &ballDetector);

// FFI entry points
#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) __attribute__((used))
void* initialize_detector(const char* model_path);

__attribute__((visibility("default"))) __attribute__((used))
const char* detect_objects(void* detector_ptr, const unsigned char* image_bytes, int width, int height, int channels);

__attribute__((visibility("default"))) __attribute__((used))
void release_detector(void* detector_ptr);

#ifdef __cplusplus
}
#endif

#endif  // TABLEIZER_HPP
