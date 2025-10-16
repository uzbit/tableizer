
#include "utilities.hpp"
#include "quad_analysis.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

std::string base64Encode(unsigned char const* bytesToEncode, unsigned int inLen) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char charArray3[3];
    unsigned char charArray4[4];

    while (inLen--) {
        charArray3[i++] = *(bytesToEncode++);
        if (i == 3) {
            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[3] = charArray3[2] & 0x3f;

            for (i = 0; (i < 4); i++) ret += base64_chars[charArray4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) charArray3[j] = '\0';

        charArray4[0] = (charArray3[0] & 0xfc) >> 2;
        charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
        charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
        charArray4[3] = charArray3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++) ret += base64_chars[charArray4[j]];

        while ((i++ < 3)) ret += '=';
    }

    return ret;
}

std::vector<unsigned char> base64Decode(std::string const& encodedString) {
    int inLen = encodedString.size();
    int i = 0;
    int j = 0;
    int inIdx = 0;
    unsigned char charArray4[4], charArray3[3];
    std::vector<unsigned char> ret;

    while (inLen-- && (encodedString[inIdx] != '=') && isBase64(encodedString[inIdx])) {
        charArray4[i++] = encodedString[inIdx];
        inIdx++;
        if (i == 4) {
            for (i = 0; i < 4; i++) charArray4[i] = base64_chars.find(charArray4[i]);

            charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
            charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
            charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

            for (i = 0; (i < 3); i++) ret.push_back(charArray3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) charArray4[j] = 0;

        for (j = 0; j < 4; j++) charArray4[j] = base64_chars.find(charArray4[j]);

        charArray3[0] = (charArray4[0] << 2) + ((charArray4[1] & 0x30) >> 4);
        charArray3[1] = ((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2);
        charArray3[2] = ((charArray4[2] & 0x3) << 6) + charArray4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(charArray3[j]);
    }

    return ret;
}

WarpResult warpTable(const cv::Mat& bgrImg, const std::vector<cv::Point2f>& quad,
                     const std::string& imagePath, int outW, bool rotate, double scaleF) {
    int outH;
    int canvasW = outW;

    if (rotate) {
        outH = static_cast<int>(scaleF * 2.0 * canvasW);
    } else {
        outH = static_cast<int>(scaleF * canvasW);
        canvasW *= 2;  // LANDSCAPE_ASPECT_RATIO:1 landscape
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
        return {warped, finalH};
    }

    cv::Mat rot = (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, canvasW - 1, 0, 0, 1);
    finalH = rot * Hpersp;

    cv::warpPerspective(bgrImg, warped, finalH, {outH, canvasW});

    return {warped, finalH};
}

vector<Point2f> orderQuad(const vector<Point2f>& pts) {
    vector<Point2f> sortedPts = pts;
    Scalar centroid = mean(pts);
    sort(sortedPts.begin(), sortedPts.end(), [centroid](const Point2f& a, const Point2f& b) {
        return atan2(a.y - centroid[1], a.x - centroid[0]) <
               atan2(b.y - centroid[1], b.x - centroid[0]);
    });
    return sortedPts;
}

TransformationResult transformPointsUsingQuad(
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& quadPoints,
    cv::Size imageSize,
    cv::Size displaySize,
    int inputRotationDegrees
) {
    TransformationResult result;

    if (quadPoints.size() != 4 || points.empty()) {
        result.transformedPoints = points;
        result.needsRotation = false;
        result.orientation = OTHER;
        return result;
    }

    // Step 1: Rotate input points to match quad coordinate space
    std::vector<cv::Point2f> rotatedPoints = points;
    cv::Size adjustedImageSize = imageSize;

    if (inputRotationDegrees == 90) {
        // 90° CW: (x, y) → (height - y, x)
        for (auto& point : rotatedPoints) {
            float newX = imageSize.height - point.y;
            float newY = point.x;
            point.x = newX;
            point.y = newY;
        }
        adjustedImageSize = cv::Size(imageSize.height, imageSize.width);
        LOGI("[transformPointsUsingQuad] Applied 90° input rotation, imageSize: %dx%d → %dx%d",
             imageSize.width, imageSize.height, adjustedImageSize.width, adjustedImageSize.height);
    } else if (inputRotationDegrees == 180) {
        // 180°: (x, y) → (width - x, height - y)
        for (auto& point : rotatedPoints) {
            point.x = imageSize.width - point.x;
            point.y = imageSize.height - point.y;
        }
        LOGI("[transformPointsUsingQuad] Applied 180° input rotation");
    } else if (inputRotationDegrees == 270) {
        // 270° CW: (x, y) → (y, width - x)
        for (auto& point : rotatedPoints) {
            float newX = point.y;
            float newY = imageSize.width - point.x;
            point.x = newX;
            point.y = newY;
        }
        adjustedImageSize = cv::Size(imageSize.height, imageSize.width);
        LOGI("[transformPointsUsingQuad] Applied 270° input rotation, imageSize: %dx%d → %dx%d",
             imageSize.width, imageSize.height, adjustedImageSize.width, adjustedImageSize.height);
    }

    std::vector<cv::Point2f> orderedQuad = orderQuad(quadPoints);

    // Use QuadAnalysis to determine orientation
    QuadOrientation orientation = QuadAnalysis::orientation(orderedQuad);

    // Calculate aspect ratio for TOP_DOWN cases
    float topLength = cv::norm(orderedQuad[1] - orderedQuad[0]);
    float rightLength = cv::norm(orderedQuad[2] - orderedQuad[1]);
    float aspectRatio = (rightLength > 1e-6f) ? (topLength / rightLength) : 0.0f;

    // Rotation needed for LONG_SIDE or TOP_DOWN with landscape aspect ratio
    bool needsRotation = (orientation == LONG_SIDE) ||
                        (orientation == TOP_DOWN && aspectRatio >= 1.75f);

    LOGI("[transformPointsUsingQuad] Quad orientation: %s, aspectRatio: %.3f, needsRotation: %s",
         QuadAnalysis::orientationToString(orientation).c_str(),
         aspectRatio,
         needsRotation ? "TRUE" : "FALSE");

    result.needsRotation = needsRotation;
    result.orientation = orientation;

    // Set up destination corners and compute homography
    cv::Mat H_final;
    if (needsRotation) {
        // Destination for landscape warping (to be rotated to portrait)
        float out_w = displaySize.width;
        float out_h = displaySize.height;
        std::vector<cv::Point2f> landscape_dst = {
            {0, 0}, {out_h - 1, 0}, {out_h - 1, out_w - 1}, {0, out_w - 1}
        };
        cv::Mat H_landscape = cv::getPerspectiveTransform(orderedQuad, landscape_dst);

        // 90° CCW rotation matrix: landscape -> portrait
        cv::Mat rot = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, out_h - 1, 0, 0, 1);
        H_final = rot * H_landscape;
    } else {
        // Direct warp to portrait destination
        std::vector<cv::Point2f> portrait_dst = {
            {0, 0},
            {(float)displaySize.width - 1, 0},
            {(float)displaySize.width - 1, (float)displaySize.height - 1},
            {0, (float)displaySize.height - 1}
        };
        H_final = cv::getPerspectiveTransform(orderedQuad, portrait_dst);
    }

    // Apply the final transformation to all points
    if (!rotatedPoints.empty()) {
        cv::perspectiveTransform(rotatedPoints, result.transformedPoints, H_final);
    }

    return result;
}

