#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "tableizer.hpp"
#include "utilities.hpp"

// Basic JSON parser to find a string value
std::string parseJson(const std::string& json, const std::string& key) {
    std::string search_key = std::string("\"") + key + std::string("\": \"");
    size_t start_pos = json.find(search_key);
    if (start_pos == std::string::npos) {
        return "";
    }
    start_pos += search_key.length();
    size_t end_pos = json.find("\"", start_pos);
    if (end_pos == std::string::npos) {
        return "";
    }
    return json.substr(start_pos, end_pos - start_pos);
}

int main() {
    std::string image_path = "/Users/uzbit/Documents/projects/tableizer/app/assets/images/P_20250718_203819.jpg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return -1;
    }

    // Convert to RGBA
    cv::Mat rgba_image;
    cv::cvtColor(image, rgba_image, cv::COLOR_BGR2RGBA);

    // Call the detection function
    const char* result_json = detect_table_rgba(
        rgba_image.data, rgba_image.cols, rgba_image.rows, rgba_image.channels());

    if (result_json == nullptr) {
        std::cerr << "Error: detect_table_rgba returned null." << std::endl;
        return -1;
    }

    std::string result_str(result_json);
    if (result_str.find("error") != std::string::npos) {
        std::cerr << "Error from detection: " << result_str << std::endl;
        return -1;
    }

    // Parse the JSON and decode the image
    std::string base64_image_str = parseJson(result_str, "image");
    if (base64_image_str.empty()) {
        std::cerr << "Error: Could not find 'image' in JSON response." << std::endl;
        return -1;
    }

    try {
        std::vector<unsigned char> decoded_bytes = base64_decode(base64_image_str);
        cv::Mat decoded_image = cv::imdecode(decoded_bytes, cv::IMREAD_COLOR);

        if (decoded_image.empty()) {
            std::cerr << "Error: Decoded image is empty." << std::endl;
            return -1;
        }

        // Display the image
        cv::imshow("Detected Table", decoded_image);
        std::cout << "Displaying detected table. Press any key to exit." << std::endl;
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during image decoding or display: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Test completed successfully." << std::endl;
    return 0;
}
