#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tableizer.hpp"
#include "utilities.hpp"

int main() {
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
    const char* jsonResult = detect_table_bgra(bgra_image.data, bgra_image.cols, bgra_image.rows,
                                               bgra_image.step, 0, nullptr);

    if (jsonResult == nullptr) {
        std::cerr << "Error: detect_table_bgra returned null." << std::endl;
        return -1;
    }

    std::cout << "Successfully received detection result." << std::endl;
    std::cout << "JSON result: " << jsonResult << std::endl;

    // Simple test that we got some JSON response
    std::string jsonStr(jsonResult);
    if (jsonStr.find("quad_points") == std::string::npos) {
        std::cerr << "Error: No quad_points found in JSON response." << std::endl;
        return -1;
    }

    std::cout << "Test passed: JSON response contains quad_points." << std::endl;

    std::cout << "Test completed successfully." << std::endl;
    return 0;
}
