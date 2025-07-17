#include "ball_detector.hpp"
#include <torch/torch.h>
#include <opencv2/imgproc/imgproc.hpp>

BallDetector::BallDetector(const std::string& model_path) : device_(torch::kCPU) {
    try {
        module_ = torch::jit::load(model_path);
        module_.to(device_);
        module_.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        exit(-1);
    }
}

std::vector<Detection> BallDetector::detect(const cv::Mat& image, float conf_threshold, float iou_threshold) {
    // Pre-process the image
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);

    auto tensor_image = torch::from_blob(resized_image.data, {1, 640, 640, 3});
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = tensor_image.to(device_);

    // Inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_image);
    auto output = module_.forward(inputs).toTuple()->elements()[0].toTensor();

    // Post-process
    std::vector<Detection> detections;
    auto* data = output.data_ptr<float>();

    for (int i = 0; i < output.size(1); ++i) {
        float confidence = data[i * output.size(2) + 4];
        if (confidence > conf_threshold) {
            float x = data[i * output.size(2) + 0];
            float y = data[i * output.size(2) + 1];
            float w = data[i * output.size(2) + 2];
            float h = data[i * output.size(2) + 3];

            int class_id = 0;
            float max_class_score = 0;
            for (int j = 5; j < output.size(2); ++j) {
                if (data[i * output.size(2) + j] > max_class_score) {
                    max_class_score = data[i * output.size(2) + j];
                    class_id = j - 5;
                }
            }

            Detection det;
            det.box = cv::Rect(x - w / 2, y - h / 2, w, h);
            det.confidence = confidence;
            det.class_id = class_id;
            detections.push_back(det);
        }
    }

    // Non-Maximum Suppression
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for(const auto& det : detections) {
        boxes.push_back(det.box);
        confidences.push_back(det.confidence);
    }
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    std::vector<Detection> final_detections;
    for (int idx : indices) {
        final_detections.push_back(detections[idx]);
    }

    return final_detections;
}