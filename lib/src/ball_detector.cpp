#include "ball_detector.hpp"
#include "../libs/onnxruntime/onnxruntime/include/core/session/onnxruntime_cxx_api.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;

struct BallDetector::Impl {
    Ort::Env env;
    Ort::Session session;

    std::vector<std::string> input_node_names_str;
    std::vector<std::string> output_node_names_str;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

    Impl(const std::string& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "test"),
          session(env, modelPath.c_str(), Ort::SessionOptions{nullptr}) {
        
        Ort::AllocatorWithDefaultOptions allocator;

        // Get input names
        size_t num_input_nodes = session.GetInputCount();
        input_node_names_str.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name_ptr = session.GetInputNameAllocated(i, allocator);
            input_node_names_str.push_back(input_name_ptr.get());
        }
        input_node_names.reserve(input_node_names_str.size());
        for(const auto& s : input_node_names_str) {
            input_node_names.push_back(s.c_str());
        }

        // Get output names
        size_t num_output_nodes = session.GetOutputCount();
        output_node_names_str.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_ptr = session.GetOutputNameAllocated(i, allocator);
            output_node_names_str.push_back(output_name_ptr.get());
        }
        output_node_names.reserve(output_node_names_str.size());
        for(const auto& s : output_node_names_str) {
            output_node_names.push_back(s.c_str());
        }
    }
};

BallDetector::BallDetector(const std::string& modelPath)
    : pimpl(std::make_unique<Impl>(modelPath)) {}

BallDetector::~BallDetector() = default;

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

std::vector<Detection> BallDetector::detect(const cv::Mat& image, float confThreshold,
                                            float iouThreshold) {
    /* 1. letter-box to N × N ------------------------------------------------ */
    constexpr int kTarget = 800;
    int ow = image.cols, oh = image.rows;
    float r = std::min(float(kTarget) / ow, float(kTarget) / oh);
    int nw = std::round(ow * r), nh = std::round(oh * r);
    int pw = (kTarget - nw) / 2, ph = (kTarget - nh) / 2;

    cv::Mat lb(kTarget, kTarget, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(image, lb(cv::Rect(pw, ph, nw, nh)), {nw, nh});
    cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
    lb.convertTo(lb, CV_32F, 1.f / 255);

    /* 2. Create input tensor ------------------------------------------------- */
    std::vector<int64_t> input_shape = {1, 3, kTarget, kTarget};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, (float*)lb.data, lb.total() * lb.channels(), input_shape.data(),
        input_shape.size());

    /* 3. Run inference ------------------------------------------------------- */
    auto output_tensors = pimpl->session.Run(Ort::RunOptions{nullptr}, pimpl->input_node_names.data(),
                                      &input_tensor, 1, pimpl->output_node_names.data(), 1);

    /* 4. Parse output -------------------------------------------------------- */
    auto* raw_output = output_tensors[0].GetTensorData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_detections = output_shape[1];
    const int num_classes = output_shape[2] - 5;

    std::vector<Detection> pre;
    for (int i = 0; i < num_detections; ++i) {
        const float* detection = raw_output + i * (5 + num_classes);
        float conf = detection[4];
        if (conf < confThreshold) continue;

        float best_class_score = 0;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            if (detection[5 + j] > best_class_score) {
                best_class_score = detection[5 + j];
                class_id = j;
            }
        }

        float cx = detection[0], cy = detection[1], w = detection[2], h = detection[3];
        float x1 = (cx - w / 2 - pw) / r;
        float y1 = (cy - h / 2 - ph) / r;
        float x2 = (cx + w / 2 - pw) / r;
        float y2 = (cy + h / 2 - ph) / r;

        Detection d;
        d.x = x1;
        d.y = y1;
        d.radius = (x2 - x1 + y2 - y1) / 4;
        d.class_id = class_id;
        pre.push_back(d);
    }

    /* 5. OpenCV NMS ------------------------------------------------------------- */
    std::vector<cv::Rect> boxes;
    boxes.reserve(pre.size());
    std::vector<float> scores;
    scores.reserve(pre.size());
    for (auto& det : pre) {
        boxes.emplace_back(det.x, det.y, det.radius, det.radius);
        scores.push_back(det.x); // a dummy value, since confidence is not in the struct
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);

    std::vector<Detection> finalDet;
    finalDet.reserve(keep.size());
    for (int k : keep) finalDet.push_back(pre[k]);
    return finalDet;
}
