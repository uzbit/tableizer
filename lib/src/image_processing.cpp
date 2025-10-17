#include "image_processing.hpp"
#include "utilities.hpp"
#include <algorithm>

void drawBallsOnImages(const std::vector<Detection>& detections, cv::Mat& warpedOut,
                       cv::Mat& shotStudio, const cv::Mat& transform) {
    if (detections.empty()) {
        LOGI("No balls detected.");
        return;
    }

    std::vector<cv::Point2f> ballCentersOrig;
    for (const auto& d : detections) {
        ballCentersOrig.emplace_back(d.center);
    }

    std::vector<cv::Point2f> ballCentersCanonical;
    cv::perspectiveTransform(ballCentersOrig, ballCentersCanonical, transform);

    int tableSizeInches = 100;
    int longEdgePx = std::max(warpedOut.cols, warpedOut.rows);
    int radius = std::max((int)round(longEdgePx * (2.25 / tableSizeInches) / 2.0), 4);
    float textSize = 0.7 * (radius / 8.0);
    const cv::Scalar textColor(255, 255, 255);

    for (size_t i = 0; i < ballCentersCanonical.size(); ++i) {
        const auto& ballPosition = ballCentersCanonical[i];
        LOGI("  • class %d conf %.3f @ (%.1f, %.1f)", detections[i].classId,
             detections[i].confidence, ballPosition.x, ballPosition.y);

        cv::Scalar ballColor;
        switch (detections[i].classId) {
            case 3:
                ballColor = cv::Scalar(0, 0, 255);
                break;
            case 2:
                ballColor = cv::Scalar(255, 222, 33);
                break;
            case 1:
                ballColor = cv::Scalar(255, 255, 255);
                break;
            case 0:
                ballColor = cv::Scalar(0, 0, 0);
                break;
            default:
                ballColor = cv::Scalar(128, 128, 128);
                break;
        }

        cv::circle(warpedOut, ballPosition, radius, ballColor, cv::FILLED, cv::LINE_AA);
        cv::putText(warpedOut, std::to_string(detections[i].classId),
                    ballPosition + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX, textSize,
                    textColor, 2);

        if (!shotStudio.empty()) {
            cv::circle(shotStudio, ballPosition, radius, ballColor, cv::FILLED);
            cv::putText(shotStudio, std::to_string(detections[i].classId),
                        ballPosition + cv::Point2f(radius + 2, 0), cv::FONT_HERSHEY_SIMPLEX,
                        textSize, textColor, 2);
        }
    }
}

cv::Mat createMaskedImage(const cv::Mat& image, const std::vector<cv::Point2f>& quadPoints) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::Mat maskedImage;

    if (quadPoints.size() == 4) {
        std::vector<cv::Point> quadPointsInt;
        for (const auto& pt : quadPoints) {
            quadPointsInt.emplace_back(cv::Point(cvRound(pt.x), cvRound(pt.y)));
        }
        cv::fillConvexPoly(mask, quadPointsInt, cv::Scalar(255));
        image.copyTo(maskedImage, mask);

        // Draw the quad edges with different colors
        // Edge definitions: 0=TL, 1=TR, 2=BR, 3=BL
        cv::line(maskedImage, quadPoints[0], quadPoints[1], cv::Scalar(0, 0, 255), 2);  // Top: Red
        cv::line(maskedImage, quadPoints[1], quadPoints[2], cv::Scalar(0, 255, 0),
                 2);  // Right: Green
        cv::line(maskedImage, quadPoints[2], quadPoints[3], cv::Scalar(255, 0, 0),
                 2);  // Bottom: Blue
        cv::line(maskedImage, quadPoints[3], quadPoints[0], cv::Scalar(0, 255, 255),
                 2);  // Left: Yellow
    } else {
        // If no quad, return a copy of the original image
        image.copyTo(maskedImage);
    }

    return maskedImage;
}
