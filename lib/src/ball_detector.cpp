#include "ball_detector.hpp"

#include <torch/torch.h>

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

BallDetector::BallDetector(const string& modelPath) : device(torch::kCPU) {
    try {
        module = torch::jit::load(modelPath, device);
        module.to(device);
        module.eval();
    } catch (const c10::Error& e) {
        cerr << "Error loading the model: " << e.what() << endl;
        exit(-1);
    }
}

inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

// Assumes Detection { Rect2f box; float confidence; int classId; }
std::vector<Detection> BallDetector::detect(const cv::Mat& image, float confThreshold,
                                            float iouThreshold) {
    /* 1. letter-box to 640 × 640 ------------------------------------------------ */
    constexpr int kTarget = 640;
    int ow = image.cols, oh = image.rows;
    float r = std::min(float(kTarget) / ow, float(kTarget) / oh);
    int nw = std::round(ow * r), nh = std::round(oh * r);
    int pw = (kTarget - nw) / 2, ph = (kTarget - nh) / 2;

    cv::Mat lb(kTarget, kTarget, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(image, lb(cv::Rect(pw, ph, nw, nh)), {nw, nh});
    cv::cvtColor(lb, lb, cv::COLOR_BGR2RGB);
    lb.convertTo(lb, CV_32F, 1.f / 255);

    auto tensor = torch::from_blob(lb.data, {1, kTarget, kTarget, 3}, torch::kFloat32)
                      .to(device)
                      .permute({0, 3, 1, 2})
                      .contiguous();

    /* 2. forward – raw head is (1, 8, N) --------------------------------------- */
    auto out = module.forward({tensor}).toTensor();     // [1,8,N]
    out = out.squeeze(0).transpose(0, 1).contiguous();  // → [N,8]

    /* 3. parse + de-letter-box -------------------------------------------------- */
    std::vector<Detection> pre;
    auto a = out.accessor<float, 2>();
    const int N = out.size(0);

    for (int i = 0; i < N; ++i) {
        float best = 0.f;
        int cls = -1;
        for (int c = 0; c < 4; ++c) {
            float p = sigmoid(a[i][4 + c]);  // ← add sigmoid
            if (p > best) {
                best = p;
                cls = c;
            }
        }
        float conf = best;
        if (conf < confThreshold) continue;

        /*float obj = a[i][4];
        float best = 0.f;
        int cls = -1;
        for (int c = 0; c < 4; ++c)
            if (a[i][5 + c] > best) {
                best = a[i][5 + c];
                cls = c;
            }

        float conf = obj * best;
        if (conf < confThreshold) continue;  // skip low-conf
        */

        /* cx cy w h  →  x1 y1 x2 y2 in letter-box space */
        float cx = a[i][0], cy = a[i][1], w = a[i][2], h = a[i][3];
        float x1 = cx - w * 0.5f, y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f, y2 = cy + h * 0.5f;

        /* undo padding + scale back */
        x1 = (x1 - pw) / r;
        y1 = (y1 - ph) / r;
        x2 = (x2 - pw) / r;
        y2 = (y2 - ph) / r;
        x1 = std::clamp(x1, 0.f, float(ow - 1));
        y1 = std::clamp(y1, 0.f, float(oh - 1));
        x2 = std::clamp(x2, 0.f, float(ow - 1));
        y2 = std::clamp(y2, 0.f, float(oh - 1));

        pre.push_back({cv::Rect2f(x1, y1, x2 - x1, y2 - y1), conf, cls});
    }

    /* 4. OpenCV NMS ------------------------------------------------------------- */
    std::vector<cv::Rect> boxes;
    boxes.reserve(pre.size());
    std::vector<float> scores;
    scores.reserve(pre.size());
    for (auto& d : pre) {
        boxes.emplace_back(d.box);
        scores.push_back(d.confidence);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);

    std::vector<Detection> finalDet;
    finalDet.reserve(keep.size());
    for (int k : keep) finalDet.push_back(pre[k]);
    return finalDet;  // pixel coords, cls 0-3
}