
#include "utilities.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++) ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++) ret += base64_chars[char_array_4[j]];

        while ((i++ < 3)) ret += '=';
    }

    return ret;
}

std::vector<unsigned char> base64_decode(std::string const& encoded_string) {
    int in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<unsigned char> ret;

    while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i == 4) {
            for (i = 0; i < 4; i++) char_array_4[i] = base64_chars.find(char_array_4[i]);

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++) ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 4; j++) char_array_4[j] = 0;

        for (j = 0; j < 4; j++) char_array_4[j] = base64_chars.find(char_array_4[j]);

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
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

    // ---- embed ROTATION_DEGREES_CCW rotation ------------------------------------------
    cv::Mat rot = (cv::Mat_<float>(3, 3) << 0, 1, 0, -1, 0, canvasW - 1, 0, 0, 1);
    finalH = rot * Hpersp;

    cv::warpPerspective(bgrImg, warped, finalH, {outH, canvasW});

    return {warped, finalH};
}

// Function to order the quad points counter-clockwise
vector<Point2f> orderQuad(const vector<Point2f>& pts) {
    vector<Point2f> sorted_pts = pts;
    Scalar centroid = mean(pts);
    sort(sorted_pts.begin(), sorted_pts.end(), [centroid](const Point2f& a, const Point2f& b) {
        return atan2(a.y - centroid[1], a.x - centroid[0]) <
               atan2(b.y - centroid[1], b.x - centroid[0]);
    });
    return sorted_pts;
}

// Coordinate transformation functions implementation
TransformationResult transformPointsUsingQuad(
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& quadPoints,
    cv::Size imageSize,
    cv::Size displaySize
) {
    TransformationResult result;
    
    if (quadPoints.size() != 4 || points.empty()) {
        result.transformedPoints = points;
        result.needsRotation = false;
        return result;
    }
    
    // Order quad points counter-clockwise
    std::vector<cv::Point2f> orderedQuad = orderQuad(quadPoints);
    
    // Check if we need rotation based on quad orientation
    float topLength = cv::norm(orderedQuad[1] - orderedQuad[0]);
    float rightLength = cv::norm(orderedQuad[2] - orderedQuad[1]);
    bool needsRotation = topLength > rightLength * 1.75f;
    
    result.needsRotation = needsRotation;
    
    // Define destination corners based on rotation requirement
    std::vector<cv::Point2f> dstCorners;
    
    if (needsRotation) {
        // Map landscape quad to portrait display by rotating ROTATION_DEGREES_CCW CCW
        dstCorners = {
            cv::Point2f(0, displaySize.height),                    // top-left of quad → bottom-left of display
            cv::Point2f(0, 0),                                     // top-right of quad → top-left of display
            cv::Point2f(displaySize.width, 0),                     // bottom-right of quad → top-right of display
            cv::Point2f(displaySize.width, displaySize.height)     // bottom-left of quad → bottom-right of display
        };
    } else {
        // Direct mapping for portrait quad
        dstCorners = {
            cv::Point2f(0, 0),
            cv::Point2f(displaySize.width, 0),
            cv::Point2f(displaySize.width, displaySize.height),
            cv::Point2f(0, displaySize.height)
        };
    }
    
    // Transform all points
    result.transformedPoints.reserve(points.size());
    for (const auto& point : points) {
        cv::Point2f transformed = perspectiveTransform(point, orderedQuad, dstCorners);
        result.transformedPoints.push_back(transformed);
    }
    
    return result;
}

cv::Point2f perspectiveTransform(
    const cv::Point2f& point,
    const std::vector<cv::Point2f>& srcQuad,
    const std::vector<cv::Point2f>& dstRect
) {
    if (srcQuad.size() != 4 || dstRect.size() != 4) {
        return point;
    }
    
    // Source quad points (ordered counter-clockwise)
    const cv::Point2f& p0 = srcQuad[0]; // top-left
    const cv::Point2f& p1 = srcQuad[1]; // top-right
    const cv::Point2f& p2 = srcQuad[2]; // bottom-right
    const cv::Point2f& p3 = srcQuad[3]; // bottom-left
    
    // Destination rectangle corners
    const cv::Point2f& q0 = dstRect[0]; // top-left
    const cv::Point2f& q1 = dstRect[1]; // top-right
    const cv::Point2f& q2 = dstRect[2]; // bottom-right
    const cv::Point2f& q3 = dstRect[3]; // bottom-left
    
    // Find normalized coordinates (u,v) within the source quadrilateral
    cv::Point2f uv = findUVInQuad(point, p0, p1, p2, p3);
    float u = uv.x;
    float v = uv.y;
    
    // Apply bilinear interpolation to destination rectangle
    cv::Point2f top = q0 + u * (q1 - q0);    // lerp(q0, q1, u)
    cv::Point2f bottom = q3 + u * (q2 - q3); // lerp(q3, q2, u)
    return top + v * (bottom - top);          // lerp(top, bottom, v)
}

cv::Point2f findUVInQuad(
    const cv::Point2f& P,
    const cv::Point2f& p0, const cv::Point2f& p1,
    const cv::Point2f& p2, const cv::Point2f& p3
) {
    // Use Newton's method to solve the bilinear equation:
    // P = (1-u)(1-v)*p0 + u*(1-v)*p1 + u*v*p2 + (1-u)*v*p3
    
    float u = 0.5f, v = 0.5f; // Initial guess
    
    for (int i = 0; i < 10; ++i) { // Max MAX_NEWTON_ITERATIONS iterations
        // Current point using bilinear interpolation
        cv::Point2f currentP(
            (1-u)*(1-v)*p0.x + u*(1-v)*p1.x + u*v*p2.x + (1-u)*v*p3.x,
            (1-u)*(1-v)*p0.y + u*(1-v)*p1.y + u*v*p2.y + (1-u)*v*p3.y
        );
        
        // Error vector
        float dx = P.x - currentP.x;
        float dy = P.y - currentP.y;
        
        // If close enough, break
        if (std::abs(dx) < 0.1f && std::abs(dy) < 0.1f) break;
        
        // Jacobian matrix partial derivatives
        float dPdu_x = -(1-v)*p0.x + (1-v)*p1.x + v*p2.x - v*p3.x;
        float dPdu_y = -(1-v)*p0.y + (1-v)*p1.y + v*p2.y - v*p3.y;
        float dPdv_x = -(1-u)*p0.x - u*p1.x + u*p2.x + (1-u)*p3.x;
        float dPdv_y = -(1-u)*p0.y - u*p1.y + u*p2.y + (1-u)*p3.y;
        
        // Inverse Jacobian determinant
        float det = dPdu_x * dPdv_y - dPdu_y * dPdv_x;
        if (std::abs(det) < 1e-10f) break; // Avoid division by zero
        
        // Newton step
        float du = (dPdv_y * dx - dPdv_x * dy) / det;
        float dv = (-dPdu_y * dx + dPdu_x * dy) / det;
        
        u += du;
        v += dv;
        
        // Clamp to [NORMALIZED_MIN, NORMALIZED_MAX]
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);
    }
    
    return cv::Point2f(u, v);
}
