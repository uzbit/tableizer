#include "ball_detector.hpp"

#include "utilities.hpp"

#if defined(PLATFORM_ANDROID)
#include <core/session/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct BallDetector::Impl {
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    vector<string> inputNodeNamesStr;
    vector<const char*> inputNodeNames;
    vector<string> outputNodeNamesStr;
    vector<const char*> outputNodeNames;

    Impl(const string& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "ball_detector"),
          session(env, modelPath.c_str(), Ort::SessionOptions{nullptr}) {
        LOGI("ONNX session created successfully for model: %s", modelPath.c_str());

        size_t numInputNodes = session.GetInputCount();
        inputNodeNamesStr.reserve(numInputNodes);
        for (size_t i = 0; i < numInputNodes; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            inputNodeNamesStr.push_back(name.get());
        }
        for (const auto& s : inputNodeNamesStr) inputNodeNames.push_back(s.c_str());

        size_t numOutputNodes = session.GetOutputCount();
        outputNodeNamesStr.reserve(numOutputNodes);
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            outputNodeNamesStr.push_back(name.get());
        }
        for (const auto& s : outputNodeNamesStr) outputNodeNames.push_back(s.c_str());
    }
};

BallDetector::BallDetector(const string& modelPath) : pimpl(make_unique<Impl>(modelPath)) {}
BallDetector::~BallDetector() = default;

vector<Detection> BallDetector::detect(const Mat& image, float confThreshold, float iouThreshold) {
    constexpr int kTarget = 1280;  // MUST MATCH BALL DETECTION "imgsz" in model_table.py
    constexpr int numClasses = 4;

    int imgW = image.cols, imgH = image.rows;
    float scale = min(float(kTarget) / imgW, float(kTarget) / imgH);
    int newW = int(round(imgW * scale));
    int newH = int(round(imgH * scale));
    int padW = kTarget - newW, padH = kTarget - newH;
    int padLeft = padW / 2, padTop = padH / 2;

    Mat resized;
    resize(image, resized, {newW, newH}, 0, 0, INTER_LINEAR);
    Mat input(kTarget, kTarget, CV_8UC3, Scalar(114, 114, 114));
    resized.copyTo(input(Rect(padLeft, padTop, newW, newH)));

    cvtColor(input, input, COLOR_BGR2RGB);
    input.convertTo(input, CV_32F, 1.f / 255.f);

    Mat blob;
    dnn::blobFromImage(input, blob);  // NHWC → NCHW, float32

    vector<int64_t> inputShape = {1, 3, kTarget, kTarget};
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, blob.ptr<float>(), blob.total(), inputShape.data(), inputShape.size());

    auto outputTensors =
        pimpl->session.Run(Ort::RunOptions{nullptr}, pimpl->inputNodeNames.data(), &inputTensor,
                           1, pimpl->outputNodeNames.data(), 1);

    auto shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* raw = outputTensors[0].GetTensorData<float>();

    int numAttrs = shape[1];
    int numPreds = shape[2];

    auto sigmoid = [](float x) { return 1.f / (1.f + exp(-x)); };

    vector<Detection> preNms;
    for (int i = 0; i < numPreds; ++i) {
        float cx = raw[0 * numPreds + i];
        float cy = raw[1 * numPreds + i];
        float width = raw[2 * numPreds + i];
        float height = raw[3 * numPreds + i];

        float bestCls = 0.f;
        int classId = -1;
        for (int j = 0; j < numClasses; ++j) {
            float clsScore = sigmoid(raw[(4 + j) * numPreds + i]);
            if (clsScore > bestCls) {
                bestCls = clsScore;
                classId = j;
            }
        }

        float conf = bestCls;
        if (conf < confThreshold) continue;

        float x1 = (cx - width / 2 - padLeft) / scale;
        float y1 = (cy - height / 2 - padTop) / scale;
        float x2 = (cx + width / 2 - padLeft) / scale;
        float y2 = (cy + height / 2 - padTop) / scale;

        x1 = clamp(x1, 0.f, float(imgW - 1));
        y1 = clamp(y1, 0.f, float(imgH - 1));
        x2 = clamp(x2, 0.f, float(imgW - 1));
        y2 = clamp(y2, 0.f, float(imgH - 1));

        Rect box = Rect(Point(x1, y1), Point(x2, y2));
        Point2f center = Point2f(box.x + box.width / 2, box.y + box.height / 2);
        preNms.push_back({box, center, classId, conf});
    }

    vector<Rect> boxes;
    vector<float> scores;
    for (const auto& d : preNms) {
        boxes.push_back(d.box);
        scores.push_back(d.confidence);
    }

    vector<int> keep;
    dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);

    vector<Detection> final;
    for (int idx : keep) final.push_back(preNms[idx]);
    return final;
}
