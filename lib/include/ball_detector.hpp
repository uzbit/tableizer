#ifndef BALL_DETECTOR_HPP
#define BALL_DETECTOR_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    cv::Rect box;
    cv::Point2f center;
    int classId;
    float confidence;
};

class BallDetector {
   public:
    BallDetector(const std::string& modelPath);
    ~BallDetector();  // Required for std::unique_ptr with forward-declared type
    std::vector<Detection> detect(const cv::Mat& image, float confThreshold, float iouThreshold);

   private:
    struct Impl;  // Forward declaration
    std::unique_ptr<Impl> pimpl;
};

#endif  // BALL_DETECTOR_HPP