#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tableizer.hpp"
#include "utilities.hpp"

int main() {
#if LOCAL_BUILD
    std::string image_path =
        "/Users/uzbit/Documents/projects/tableizer/app/assets/images/P_20250718_203819.jpg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return -1;
    }

    // Convert to BGRA as expected by the function
    cv::Mat bgra_image;
    cv::cvtColor(image, bgra_image, cv::COLOR_BGR2BGRA);

    // Call the detection function
    DetectionResult* result = detect_table_bgra(
        bgra_image.data, bgra_image.cols, bgra_image.rows, bgra_image.step, nullptr);

    if (result == nullptr) {
        std::cerr << "Error: detect_table_bgra returned null." << std::endl;
        return -1;
    }

    std::cout << "Successfully received detection result." << std::endl;
    std::cout << "Detected " << result->quad_points_count << " quad points." << std::endl;

    if (result->quad_points_count != 4) {
        std::cerr << "Error: Expected 4 quad points, but got " << result->quad_points_count
                  << std::endl;
        free_bgra_detection_result(result);
        return -1;
    }

    for (int i = 0; i < result->quad_points_count; ++i) {
        std::cout << "  - Point " << i << ": (" << result->quad_points[i].x << ", "
                  << result->quad_points[i].y << ")" << std::endl;
    }

    // Clean up
    free_bgra_detection_result(result);

    std::cout << "Test completed successfully." << std::endl;
    return 0;
#else
    std::cout << "Test skipped: Not a LOCAL_BUILD." << std::endl;
    return 0;
#endif
}
