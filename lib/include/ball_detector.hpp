#ifndef BALL_DETECTOR_HPP
#define BALL_DETECTOR_HPP

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <vector>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class BallDetector {
public:
    BallDetector(const std::string& model_path);
    std::vector<Detection> detect(const cv::Mat& image, float conf_threshold = 0.25, float iou_threshold = 0.45);

private:
    torch::jit::script::Module module_;
    torch::Device device_;
};

#endif // BALL_DETECTOR_HPP
