#ifndef BALL_DETECTOR_HPP
#define BALL_DETECTOR_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include "tableizer_ffi.h"

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class BallDetector {
   public:
    BallDetector(const std::string& modelPath);
    ~BallDetector(); // Required for std::unique_ptr with forward-declared type
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold = 0.25,
                                  float iouThreshold = 0.45);
    ~BallDetector();

   private:
    struct Impl; // Forward declaration
    std::unique_ptr<Impl> pimpl;
};

#endif  // BALL_DETECTOR_HPP