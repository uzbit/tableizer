
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
