#include "ball_detector.hpp"

#include <torch/torch.h>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

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
// Assumes Detection { cv::Rect2f box; float confidence; int class_id; }

std::vector<Detection> BallDetector::detect(const cv::Mat& image, float confThreshold,
                                            float iouThreshold) {
    // ───────────────────────────────────────────────────────── 1. Letter-box to 640×640
    constexpr int kTarget = 640;
    const int origW = image.cols;
    const int origH = image.rows;

    const float r = std::min(static_cast<float>(kTarget) / origW,
                             static_cast<float>(kTarget) / origH);  // uniform resize
    const int newW = static_cast<int>(std::round(origW * r));
    const int newH = static_cast<int>(std::round(origH * r));

    const int padW = (kTarget - newW) / 2;  // left/right padding
    const int padH = (kTarget - newH) / 2;  // top/bottom padding

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));

    cv::Mat letterbox(kTarget, kTarget, CV_8UC3, cv::Scalar(114, 114, 114));  // YOLOv5 fill
    resized.copyTo(letterbox(cv::Rect(padW, padH, newW, newH)));

    // ───────────────────────────────────────────────────────── 2. To tensor
    cv::cvtColor(letterbox, letterbox, cv::COLOR_BGR2RGB);
    letterbox.convertTo(letterbox, CV_32F, 1.0 / 255.0);

    auto tensorImg = torch::from_blob(letterbox.data, {1, kTarget, kTarget, 3})
                         .to(device_)            // CPU / CUDA
                         .permute({0, 3, 1, 2})  // NHWC → NCHW
                         .contiguous();

    // ───────────────────────────────────────────────────────── 3. Forward
    std::vector<torch::jit::IValue> inputs{tensorImg};
    at::Tensor out =
        module_.forward(inputs).toTuple()->elements()[0].toTensor().squeeze(0);  // shape: (N, D)

    const int nDet = out.size(0);
    const int nDim = out.size(1);  // 6 or 85 depending on training
    auto a = out.accessor<float, 2>();

    // ───────────────────────────────────────────────────────── 4. Parse + de-letterbox
    std::vector<Detection> preNMS;
    preNMS.reserve(nDet);

    for (int i = 0; i < nDet; ++i) {
        float objConf = a[i][4];
        if (objConf < confThreshold) continue;

        // best class
        float bestCls = 0.f;
        int clsId = -1;
        for (int j = 5; j < nDim; ++j) {
            if (a[i][j] > bestCls) {
                bestCls = a[i][j];
                clsId = j - 5;
            }
        }
        float conf = objConf * bestCls;
        if (conf < confThreshold || clsId == 4) continue;  // skip diamonds (class 4)

        // xywh in 640-space
        float cx = a[i][0], cy = a[i][1], w = a[i][2], h = a[i][3];
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // undo padding → scaling
        x1 = (x1 - padW) / r;
        y1 = (y1 - padH) / r;
        x2 = (x2 - padW) / r;
        y2 = (y2 - padH) / r;

        // clip
        x1 = std::clamp(x1, 0.f, static_cast<float>(origW - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(origH - 1));
        x2 = std::clamp(x2, 0.f, static_cast<float>(origW - 1));
        y2 = std::clamp(y2, 0.f, static_cast<float>(origH - 1));

        Detection d;
        d.box = cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        d.confidence = conf;
        d.class_id = clsId;
        preNMS.push_back(std::move(d));
    }

    // ───────────────────────────────────────────────────────── 5. NMS (OpenCV)
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (const auto& d : preNMS) {
        boxes.emplace_back(
            cv::Rect(cv::Point(static_cast<int>(d.box.x), static_cast<int>(d.box.y)),
                     cv::Point(static_cast<int>(d.box.br().x), static_cast<int>(d.box.br().y))));
        scores.push_back(d.confidence);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);

    std::vector<Detection> finalDet;
    finalDet.reserve(keep.size());
    for (int idx : keep) finalDet.push_back(preNMS[idx]);

    return finalDet;  // all boxes in ORIGINAL image coordinates
}