#ifndef BALL_DETECTOR_HPP
#define BALL_DETECTOR_HPP

#include <torch/script.h>

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
};

class BallDetector {
   public:
    BallDetector(const std::string &modelPath);
    std::vector<Detection> detect(const cv::Mat &image, float confThreshold = 0.25,
                                  float iouThreshold = 0.45);

   private:
    torch::jit::script::Module module;
    torch::Device device;
};

#endif  // BALL_DETECTOR_HPP
