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
    // ───────────────────────────────────────────────────────── 1. Letter-box to 640×640
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

    auto tensorImg = torch::from_blob(letterbox.data, {1, kTarget, kTarget, 3})
                         .to(device)             // CPU / CUDA
                         .permute({0, 3, 1, 2})  // NHWC → NCHW
                         .contiguous();

    // ───────────────────────────────────────────────────────── 3. Forward
    vector<torch::jit::IValue> inputs{tensorImg};
    at::Tensor out =
        module.forward(inputs).toTuple()->elements()[0].toTensor().squeeze(0);  // shape: (N, D)

    const int nDet = out.size(0);
    const int nDim = out.size(1);  // 6 or 85 depending on training
    auto a = out.accessor<float, 2>();

    // ───────────────────────────────────────────────────────── 4. Parse + de-letterbox
    vector<Detection> preNMS;
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
        x1 = clamp(x1, 0.f, static_cast<float>(origW - 1));
        y1 = clamp(y1, 0.f, static_cast<float>(origH - 1));
        x2 = clamp(x2, 0.f, static_cast<float>(origW - 1));
        y2 = clamp(y2, 0.f, static_cast<float>(origH - 1));

        Detection d;
        d.box = Rect2f(Point2f(x1, y1), Point2f(x2, y2));
        d.confidence = conf;
        d.classId = clsId;
        preNMS.push_back(std::move(d));
    }

    // ───────────────────────────────────────────────────────── 5. NMS (OpenCV)
    vector<Rect> boxes;
    vector<float> scores;
    for (const auto& d : preNMS) {
        boxes.emplace_back(
            Rect(Point(static_cast<int>(d.box.x), static_cast<int>(d.box.y)),
                 Point(static_cast<int>(d.box.br().x), static_cast<int>(d.box.br().y))));
        scores.push_back(d.confidence);
    }

    vector<int> keep;
    dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);

    vector<Detection> finalDet;
    finalDet.reserve(keep.size());
    for (int idx : keep) finalDet.push_back(preNMS[idx]);

    return finalDet;  // all boxes in ORIGINAL image coordinates
}