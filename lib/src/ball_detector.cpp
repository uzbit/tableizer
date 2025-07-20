#include "ball_detector.hpp"

#include <torch/torch.h>

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

BallDetector::BallDetector(const string& modelPath) : device(torch::kCPU) {
    try {
        module = torch::jit::load(modelPath);
        module.to(device);
        module.eval();
    } catch (const c10::Error& e) {
        cerr << "Error loading the model: " << e.what() << endl;
        exit(-1);
    }
}
// Assumes Detection { Rect2f box; float confidence; int classId; }

vector<Detection> BallDetector::detect(const Mat& image, float confThreshold, float iouThreshold) {
    // ───────────────────────────────────────────────────────── 1. Letter-box to kTarget×kTarget
    constexpr int kTarget = 640;
    const int origW = image.cols;
    const int origH = image.rows;

    const float r = min(static_cast<float>(kTarget) / origW,
                        static_cast<float>(kTarget) / origH);  // uniform resize
    const int newW = static_cast<int>(round(origW * r));
    const int newH = static_cast<int>(round(origH * r));

    const int padW = (kTarget - newW) / 2;  // left/right padding
    const int padH = (kTarget - newH) / 2;  // top/bottom padding

    Mat resized;
    resize(image, resized, Size(newW, newH));

    Mat letterbox(kTarget, kTarget, CV_8UC3, Scalar(114, 114, 114));  // YOLOv5 fill
    resized.copyTo(letterbox(Rect(padW, padH, newW, newH)));

    // ───────────────────────────────────────────────────────── 2. To tensor
    cvtColor(letterbox, letterbox, COLOR_BGR2RGB);
    letterbox.convertTo(letterbox, CV_32F, 1.0 / 255.0);

    auto tensorImg = torch::from_blob(letterbox.data, {1, kTarget, kTarget, 3},
                                      torch::kFloat32)  // ⟵ dtype
                         .to(device)
                         .permute({0, 3, 1, 2})  // NHWC → NCHW
                         .contiguous();

    // 3. forward  ---------------------------------------------------------
    auto preds = module.forward({tensorImg}).toTensor();                   // [B,N,6]
    if (preds.dim() == 3 && preds.size(0) == 1) preds = preds.squeeze(0);  // [N,6]

    const int nDet = preds.size(0);
    auto a = preds.accessor<float, 2>();  // (N,6)

    // 4. parse & de-letterbox  -------------------------------------------
    vector<Detection> preNMS;
    preNMS.reserve(nDet);

    for (int i = 0; i < nDet; ++i) {
        float conf = a[i][4];
        if (conf < confThreshold) continue;

        int clsId = static_cast<int>(a[i][5]);
        // if you had a diamond class (id 4) and want to ignore it:
        if (clsId == 4) continue;

        // xyxy are already in model space
        float x1 = a[i][0], y1 = a[i][1];
        float x2 = a[i][2], y2 = a[i][3];

        // de-letterbox
        x1 = (x1 - padW) / r;
        y1 = (y1 - padH) / r;
        x2 = (x2 - padW) / r;
        y2 = (y2 - padH) / r;

        // clip to image
        x1 = clamp(x1, 0.f, origW - 1.f);
        y1 = clamp(y1, 0.f, origH - 1.f);
        x2 = clamp(x2, 0.f, origW - 1.f);
        y2 = clamp(y2, 0.f, origH - 1.f);

        Detection d;
        d.box = Rect2f(Point2f(x1, y1), Point2f(x2, y2));
        d.confidence = conf;
        d.classId = clsId;
        preNMS.push_back(std::move(d));
    }

    // 5. (optional) extra NMS  -------------------------------------------
    // ───────────────────────────────────────── 5. (optional) extra NMS
    std::vector<Rect> boxes;
    std::vector<float> scores;
    for (const auto& d : preNMS) {
        boxes.emplace_back(
            Rect(Point(static_cast<int>(d.box.x),  // x1,y1,x2,y2
                       static_cast<int>(d.box.y)),
                 Point(static_cast<int>(d.box.br().x), static_cast<int>(d.box.br().y))));
        scores.emplace_back(d.confidence);
    }

    std::vector<int> keep;
    if (!boxes.empty())
        dnn::NMSBoxes(boxes, scores,
                      confThreshold,  // score_threshold
                      iouThreshold,   // nms_threshold
                      keep,           // output indices
                      1.f,            // eta  (default)
                      0);             // top_k (default)

    std::vector<Detection> finalDet;
    finalDet.reserve(keep.empty() ? preNMS.size() : keep.size());

    if (keep.empty())  // NMS skipped or returned everything
        finalDet = std::move(preNMS);
    else
        for (int idx : keep) finalDet.push_back(preNMS[idx]);

    return finalDet;
}