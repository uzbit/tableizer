#include "ball_detector.hpp"

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

// --- PIMPL Implementation ---
struct BallDetector::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_node_names_str;
    std::vector<const char*> input_node_names;
    std::vector<std::string> output_node_names_str;
    std::vector<const char*> output_node_names;

    Impl(const std::string& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "ball_detector"),
          session(env, modelPath.c_str(), Ort::SessionOptions{nullptr}) {
        size_t num_input_nodes = session.GetInputCount();
        input_node_names_str.reserve(num_input_nodes);
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            input_node_names_str.push_back(name.get());
        }
        for (const auto& s : input_node_names_str) input_node_names.push_back(s.c_str());

        size_t num_output_nodes = session.GetOutputCount();
        output_node_names_str.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            output_node_names_str.push_back(name.get());
        }
        for (const auto& s : output_node_names_str) output_node_names.push_back(s.c_str());
    }
};

// --- Class Implementation ---
BallDetector::BallDetector(const std::string& modelPath)
    : pimpl(std::make_unique<Impl>(modelPath)) {}
BallDetector::~BallDetector() = default;

std::vector<Detection> BallDetector::detect(const cv::Mat& image, float confThreshold,
                                            float iouThreshold) {
    constexpr int kTarget = 800;
    constexpr int num_classes = 4;

    int img_w = image.cols, img_h = image.rows;
    float r = std::min(float(kTarget) / img_w, float(kTarget) / img_h);
    int new_w = int(round(img_w * r));
    int new_h = int(round(img_h * r));
    int pad_w = kTarget - new_w, pad_h = kTarget - new_h;
    int pad_left = pad_w / 2, pad_top = pad_h / 2;

    // Letterbox manually
    cv::Mat resized;
    cv::resize(image, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);
    cv::Mat input(kTarget, kTarget, CV_8UC3, Scalar(114, 114, 114));
    resized.copyTo(input(Rect(pad_left, pad_top, new_w, new_h)));

    // Convert to model input format
    cv::cvtColor(input, input, COLOR_BGR2RGB);
    input.convertTo(input, CV_32F, 1.f / 255.f);

    cv::Mat blob;
    dnn::blobFromImage(input, blob);  // NHWC → NCHW, float32

    std::vector<int64_t> input_shape = {1, 3, kTarget, kTarget};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, blob.ptr<float>(), blob.total(), input_shape.data(), input_shape.size());

    auto output_tensors =
        pimpl->session.Run(Ort::RunOptions{nullptr}, pimpl->input_node_names.data(), &input_tensor,
                           1, pimpl->output_node_names.data(), 1);

    auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* raw = output_tensors[0].GetTensorData<float>();

    int num_attrs = shape[1];  // 9 (cx,cy,w,h,obj,cls0,cls1,cls2,cls3)
    int num_preds = shape[2];  // 13125

    auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };

    std::vector<Detection> pre_nms;
    for (int i = 0; i < num_preds; ++i) {
        float cx = raw[0 * num_preds + i];
        float cy = raw[1 * num_preds + i];
        float w = raw[2 * num_preds + i];
        float h = raw[3 * num_preds + i];

        float best_cls = 0.f;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            float cls_score = sigmoid(raw[(4 + j) * num_preds + i]);
            if (cls_score > best_cls) {
                best_cls = cls_score;
                class_id = j;
            }
        }

        float conf = best_cls;
        if (conf < confThreshold) continue;

        float x1 = (cx - w / 2 - pad_left) / r;
        float y1 = (cy - h / 2 - pad_top) / r;
        float x2 = (cx + w / 2 - pad_left) / r;
        float y2 = (cy + h / 2 - pad_top) / r;

        x1 = std::clamp(x1, 0.f, float(img_w - 1));
        y1 = std::clamp(y1, 0.f, float(img_h - 1));
        x2 = std::clamp(x2, 0.f, float(img_w - 1));
        y2 = std::clamp(y2, 0.f, float(img_h - 1));

        pre_nms.push_back({cv::Rect(Point(x1, y1), Point(x2, y2)), conf, class_id});
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& d : pre_nms) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);

    std::vector<Detection> final;
    for (int idx : keep) final.push_back(pre_nms[idx]);
    return final;
}
