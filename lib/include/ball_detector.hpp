#ifndef BALL_DETECTOR_HPP
#define BALL_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "tableizer_ffi.h"

namespace Ort {
class Env;
class Session;
class AllocatorWithDefaultOptions;
}

class BallDetector {
   public:
    BallDetector(const std::string& modelPath);
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold = 0.25,
                                  float iouThreshold = 0.45);
    ~BallDetector();

   private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

#endif  // BALL_DETECTOR_HPP