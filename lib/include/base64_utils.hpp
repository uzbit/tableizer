#ifndef BASE64_UTILS_HPP
#define BASE64_UTILS_HPP

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace Base64Utils {
    /**
     * Encode binary data to base64 string
     */
    std::string encode(const unsigned char* data, size_t len);
    
    /**
     * Encode OpenCV Mat to base64 string (PNG compressed)
     */
    std::string encodeMat(const cv::Mat& mat);
    
    /**
     * Decode base64 string to binary data
     */
    std::vector<unsigned char> decode(const std::string& encoded);
}

#endif // BASE64_UTILS_HPP