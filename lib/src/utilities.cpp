
#include "utilities.hpp"

#include <opencv2/opencv.hpp>

WarpResult warpTable(const cv::Mat& bgrImg, const std::vector<cv::Point2f>& quad,
                     const std::string& imagePath, int outW, bool rotate, double scaleF) {
    int outH;
    int canvasW = outW;

    if (rotate) {
        outH = static_cast<int>(scaleF * 2.0 * canvasW);
    } else {
        outH = static_cast<int>(scaleF * canvasW);
        canvasW *= 2;  // 2:1 landscape
    }

    std::vector<cv::Point2f> dst = {
        cv::Point2f{0.f, 0.f}, cv::Point2f{static_cast<float>(canvasW - 1), 0.f},
        cv::Point2f{static_cast<float>(canvasW - 1), static_cast<float>(outH - 1)},
        cv::Point2f{0.f, static_cast<float>(outH - 1)}};

    cv::Mat Hpersp = cv::getPerspectiveTransform(quad.data(), dst.data());
    Hpersp.convertTo(Hpersp, CV_32F);
    cv::Mat warped, finalH;

    if (!rotate) {
        cv::warpPerspective(bgrImg, warped, Hpersp, {canvasW, outH});
        finalH = Hpersp.clone();
        cv::imwrite(imagePath, warped);
        return {warped, finalH};
    }

    // ---- embed 90° CCW rotation ------------------------------------------
    cv::Mat rot = (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, canvasW - 1, 0, 0, 1);
    finalH = rot * Hpersp;

    cv::warpPerspective(bgrImg, warped, finalH, {outH, canvasW});

    cv::imwrite(imagePath, warped);
    return {warped, finalH};
}