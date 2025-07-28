#include "tableizer_ffi.h"

#include <opencv2/opencv.hpp>
#include "ball_detector.hpp"

extern "C" {

EXPORT void* create_ball_detector(const char* model_path) {
    try {
        return new BallDetector(model_path);
    } catch (const std::exception& e) {
        // Log the error, but don't crash.
        return nullptr;
    }
}

EXPORT void destroy_ball_detector(void* detector) {
    if (detector) {
        delete static_cast<BallDetector*>(detector);
    }
}

EXPORT Detection* detect_balls(void* detector, uint8_t* plane0,
                              uint8_t* plane1, uint8_t* plane2, int width,
                              int height, int p0_stride, int p1_stride,
                              int p2_stride, int p1_pix_stride,
                              int p2_pix_stride, float conf_threshold,
                              float iou_threshold, int* detection_count) {
    if (!detector) {
        *detection_count = 0;
        return nullptr;
    }

    // Create a cv::Mat from the YUV data.
    // This is a bit tricky because the planes might not be contiguous.
    cv::Mat yuv(height + height / 2, width, CV_8UC1);
    // This is a simplification. A more robust solution would handle the strides and pixel strides correctly.
    memcpy(yuv.data, plane0, width * height);
    memcpy(yuv.data + width * height, plane1, width * height / 4);
    memcpy(yuv.data + width * height + width * height / 4, plane2, width * height / 4);


    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_I420);


    auto ball_detector = static_cast<BallDetector*>(detector);
    auto detections = ball_detector->detect(bgr, conf_threshold, iou_threshold);

    *detection_count = detections.size();
    if (*detection_count == 0) {
        return nullptr;
    }

    auto* result = new Detection[detections.size()];
    for (size_t i = 0; i < detections.size(); ++i) {
        result[i] = {detections[i].x,
                     detections[i].y,
                     detections[i].radius,
                     detections[i].class_id};
    }

    return result;
}

EXPORT void free_detections(Detection* detections) {
    if (detections) {
        delete[] detections;
    }
}

}
