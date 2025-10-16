#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "ball_detector.hpp"
#include "tableizer.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory>" << std::endl;
        return 1;
    }

    std::string directory = argv[1];

    const string modelPath =
        "/Users/uzbit/Documents/projects/tableizer/app/assets/detection_model.onnx";
    BallDetector ballDetector(modelPath);

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            std::cout << "Found: " << entry.path() << std::endl;
            Mat image = imread(entry.path());
            if (image.empty()) {
                cerr << "Error: Could not open image at " << entry.path() << endl;
                return -1;
            }
            cout << "--- Step 1: " << entry.path() << " Loading ---" << endl;
            cout << "Original image dimensions: " << image.cols << "x" << image.rows << endl;

            // Resize to match mobile app resolution (max width 1280)
            constexpr int kMaxWidth = 1280;
            Mat resizedImage;
            if (image.cols > kMaxWidth) {
                double scale = static_cast<double>(kMaxWidth) / image.cols;
                int newHeight = static_cast<int>(image.rows * scale);
                resize(image, resizedImage, Size(kMaxWidth, newHeight), 0, 0, INTER_LINEAR);
                cout << "Resized to match mobile app: " << resizedImage.cols << "x" << resizedImage.rows << endl;
            } else {
                resizedImage = image;
            }
            cout << endl;

            runTableizerForImage(resizedImage, ballDetector);
        }
    }

    return 0;
}
