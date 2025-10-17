#include "ffi_api.hpp"

#include <opencv2/opencv.hpp>

#include "ball_detection.hpp"
#include "ball_detector.hpp"
#include "base64_utils.hpp"
#include "image_processing.hpp"
#include "json_parser.hpp"
#include "quad_analysis.hpp"
#include "table_detector.hpp"
#include "utilities.hpp"

// Table detection constants
#define CELL_SIZE 10
#define DELTAE_THRESH 40.0
#define RESIZE_MAX_SIZE 800

// Detection thresholds
#define CONF_THRESH 0.6
#define IOU_THRESH 0.5

extern "C" {

void* initialize_detector(const char* modelPath) {
    try {
        LOGI("initialize_detector called with path: %s", modelPath ? modelPath : "NULL");
        BallDetector* detector = new BallDetector(modelPath);
        LOGI("BallDetector created successfully");
        return static_cast<void*>(detector);
    } catch (const std::exception& e) {
        LOGE("Error initializing detector: %s", e.what());
        return nullptr;
    } catch (...) {
        LOGE("Unknown error initializing detector");
        return nullptr;
    }
}

const char* detect_balls_bgra(void* detectorPtr, const unsigned char* imageBytes, int width,
                              int height, int stride, const float* quadPoints, int quadPointsLength,
                              int channelFormat) {
    static std::string resultStr;
    if (!detectorPtr) {
        resultStr = "{\"error\": \"Invalid detector instance\"}";
        return resultStr.c_str();
    }
    try {
        LOGI("[detect_balls_bgra] Input: %dx%d, stride: %d, channelFormat: %d", width, height,
             stride, channelFormat);
        LOGI(
            "[detect_balls_bgra] First 16 bytes: %d,%d,%d,%d | %d,%d,%d,%d | %d,%d,%d,%d | "
            "%d,%d,%d,%d",
            imageBytes[0], imageBytes[1], imageBytes[2], imageBytes[3], imageBytes[4],
            imageBytes[5], imageBytes[6], imageBytes[7], imageBytes[8], imageBytes[9],
            imageBytes[10], imageBytes[11], imageBytes[12], imageBytes[13], imageBytes[14],
            imageBytes[15]);

        cv::Mat inputImage(height, width, CV_8UC4, (void*)imageBytes, stride);
        if (inputImage.empty()) {
            LOGE("[detect_balls_bgra] Failed to create Mat from bytes");
            resultStr = "{\"error\": \"Failed to create image from bytes\"}";
            return resultStr.c_str();
        }

        LOGI("[detect_balls_bgra] Created Mat: %dx%d, channels: %d", inputImage.cols,
             inputImage.rows, inputImage.channels());

        // Convert to BGRA if needed (channelFormat: 0=BGRA, 1=RGBA)
        cv::Mat bgraImage;
        if (channelFormat == 1) {
            // Input is RGBA, convert to BGRA
            LOGI("[detect_balls_bgra] Converting RGBA to BGRA");
            cv::cvtColor(inputImage, bgraImage, cv::COLOR_RGBA2BGRA);
        } else {
            // Input is already BGRA
            bgraImage = inputImage;
        }

        // Sample a pixel from the center to verify color values
        cv::Vec4b centerPixel = bgraImage.at<cv::Vec4b>(height / 2, width / 2);
        LOGI("[detect_balls_bgra] Center pixel (BGRA): B=%d G=%d R=%d A=%d", centerPixel[0],
             centerPixel[1], centerPixel[2], centerPixel[3]);

        cv::Mat imageRgb;
        // Convert from BGRA to RGB (YOLO expects RGB format)
        cv::cvtColor(bgraImage, imageRgb, cv::COLOR_BGRA2RGB);

        LOGI("[detect_balls_bgra] Converted to RGB: %dx%d, channels: %d", imageRgb.cols,
             imageRgb.rows, imageRgb.channels());

        // Sample the same pixel after conversion
        cv::Vec3b centerPixelRgb = imageRgb.at<cv::Vec3b>(height / 2, width / 2);
        LOGI("[detect_balls_bgra] Center pixel (RGB): R=%d G=%d B=%d", centerPixelRgb[0],
             centerPixelRgb[1], centerPixelRgb[2]);

        // Apply mask if provided
        cv::Mat imageForDetection;
        if (quadPoints != nullptr && quadPointsLength == 4) {
            std::vector<cv::Point2f> quadPointsVec;
            for (int i = 0; i < quadPointsLength; ++i) {
                quadPointsVec.emplace_back(quadPoints[i * 2], quadPoints[i * 2 + 1]);
            }
            LOGI("[detect_balls_bgra] Applying mask from %d quad points", quadPointsLength);
            imageForDetection = createMaskedImage(imageRgb, quadPointsVec);
            LOGI("[detect_balls_bgra] Mask applied successfully");

#if defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_WINDOWS)
            LOGI("Image size before imshow (Masked Image for Ball Detection): %dx%d",
                 imageForDetection.cols, imageForDetection.rows);
            cv::imshow("Masked Image for Ball Detection", imageForDetection);
            cv::waitKey(0);  // Wait for user to inspect both windows
#endif
        } else {
            LOGI("[detect_balls_bgra] No mask provided, using full image");
            imageForDetection = imageRgb;
        }

#if DEBUG_OUTPUT
        // Write debug image showing what's being sent to ball detection
#if defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_WINDOWS)
        cv::imwrite("/tmp/ball_debug.jpg", imageForDetection);
        LOGI("[detect_balls_bgra] Wrote debug image to: /tmp/ball_debug.jpg");
#else
        cv::imwrite("/sdcard/Download/ball_debug.jpg", imageForDetection);
        LOGI("[detect_balls_bgra] Wrote debug image to: /sdcard/Download/ball_debug.jpg");
#endif
#endif
        const auto detections = static_cast<BallDetector*>(detectorPtr)
                                    ->detect(imageForDetection, CONF_THRESH, IOU_THRESH);

        LOGI("[detect_balls_bgra] Detection complete: %zu objects found", detections.size());

        resultStr = formatDetectionsJson(detections);
        return resultStr.c_str();
    } catch (const std::exception& e) {
        LOGE("[detect_balls_bgra] Exception: %s", e.what());
        resultStr = std::string("{\"error\": \"") + e.what() + "\"}";
        return resultStr.c_str();
    }
}

void release_detector(void* detectorPtr) {
    if (detectorPtr) {
        delete static_cast<BallDetector*>(detectorPtr);
    }
}

const char* detect_table_bgra(const unsigned char* imageBytes, int width, int height, int stride,
                              int channelFormat) {
    static std::string resultStr;
    try {
        LOGI("[detect_table_bgra] Input: %dx%d, stride: %d, channelFormat: %d", width, height,
             stride, channelFormat);

        cv::Mat inputImage(height, width, CV_8UC4, (void*)imageBytes, stride);
        if (inputImage.empty()) {
            LOGE("[detect_table_bgra] Failed to create image from bytes.");
            resultStr = "{\"error\": \"Failed to create image from bytes\"}";
            return resultStr.c_str();
        }

        // Convert to BGRA if needed (channelFormat: 0=BGRA, 1=RGBA)
        cv::Mat bgraImageUnrotated;
        if (channelFormat == 1) {
            // Input is RGBA, convert to BGRA
            LOGI("[detect_table_bgra] Converting RGBA to BGRA");
            cv::cvtColor(inputImage, bgraImageUnrotated, cv::COLOR_RGBA2BGRA);
        } else {
            // Input is already BGRA
            bgraImageUnrotated = inputImage;
        }

        TableDetector tableDetector(RESIZE_MAX_SIZE, CELL_SIZE, DELTAE_THRESH);
        cv::Mat cellularMask, tableDetection;
        tableDetector.detect(bgraImageUnrotated, cellularMask, tableDetection, 0);

        std::vector<cv::Point2f> quadPoints = tableDetector.getQuadFromMask(cellularMask);

        float scaleX = (float)bgraImageUnrotated.cols / tableDetection.cols;
        float scaleY = (float)bgraImageUnrotated.rows / tableDetection.rows;

        std::vector<cv::Point2f> scaledQuadPoints;
        for (const auto& pt : quadPoints) {
            scaledQuadPoints.emplace_back(pt.x * scaleX, pt.y * scaleY);
        }

        // Analyze quad orientation
        QuadOrientation orientation = OTHER;
        std::string orientationStr = "OTHER";
        ViewValidation validation;

        if (scaledQuadPoints.size() == 4) {
            orientation = QuadAnalysis::orientation(scaledQuadPoints);
            orientationStr = QuadAnalysis::orientationToString(orientation);
            LOGI("[detect_table_bgra] Table orientation: %s", orientationStr.c_str());

            // Log image context for debugging coordinate system
            LOGI("[detect_table_bgra] === IMAGE CONTEXT ===");
            LOGI("[detect_table_bgra] Image dimensions: %dx%d (WxH), aspect=%.3f",
                 bgraImageUnrotated.cols, bgraImageUnrotated.rows,
                 (float)bgraImageUnrotated.cols / bgraImageUnrotated.rows);

            // Log quad points as percentage of image dimensions
            LOGI("[detect_table_bgra] Quad points as %% of image:");
            for (int i = 0; i < 4; i++) {
                float xPct = (scaledQuadPoints[i].x / bgraImageUnrotated.cols) * 100.0f;
                float yPct = (scaledQuadPoints[i].y / bgraImageUnrotated.rows) * 100.0f;
                LOGI("  [%d] %.1f%% width, %.1f%% height", i, xPct, yPct);
            }

            // Validate landscape 16:9 short-side view
            validation = QuadAnalysis::validateLandscapeShortSideView(
                scaledQuadPoints, cv::Size(bgraImageUnrotated.cols, bgraImageUnrotated.rows),
                orientation);
        } else {
            // No valid quad - set validation to invalid
            validation.isValid = false;
            validation.isLandscape = false;
            validation.isCorrectAspectRatio = false;
            validation.isShortSideView = false;
            validation.imageAspectRatio =
                (float)bgraImageUnrotated.cols / (float)bgraImageUnrotated.rows;
            validation.errorMessage = "No valid quad detected";
        }

        std::string json = "{\"quad_points\": [";
        for (size_t i = 0; i < scaledQuadPoints.size(); ++i) {
            json += "[" + std::to_string(scaledQuadPoints[i].x) + ", " +
                    std::to_string(scaledQuadPoints[i].y) + "]";
            if (i < scaledQuadPoints.size() - 1) json += ", ";
        }
        json += "], \"image_width\": " + std::to_string(bgraImageUnrotated.cols) +
                ", \"image_height\": " + std::to_string(bgraImageUnrotated.rows);

        json += ", \"orientation\": \"" + orientationStr + "\"";

        // Add validation results
        json += ", \"validation\": {";
        json += "\"is_valid\": " + std::string(validation.isValid ? "true" : "false") + ", ";
        json +=
            "\"is_landscape\": " + std::string(validation.isLandscape ? "true" : "false") + ", ";
        json += "\"is_correct_aspect_ratio\": " +
                std::string(validation.isCorrectAspectRatio ? "true" : "false") + ", ";
        json += "\"is_short_side_view\": " +
                std::string(validation.isShortSideView ? "true" : "false") + ", ";
        json += "\"image_aspect_ratio\": " + std::to_string(validation.imageAspectRatio);
        if (!validation.errorMessage.empty()) {
            json += ", \"error_message\": \"" + validation.errorMessage + "\"";
        }
        json += "}";

        // Add mask to JSON response if available
        if (!cellularMask.empty()) {
            try {
                // Scale mask to full resolution
                cv::Mat scaledMask;
                cv::resize(cellularMask, scaledMask,
                           cv::Size(bgraImageUnrotated.cols, bgraImageUnrotated.rows), 0, 0,
                           cv::INTER_NEAREST);

                // Encode mask to base64
                std::string maskBase64 = Base64Utils::encodeMat(scaledMask);
                json += ", \"mask\": {";
                json += "\"width\": " + std::to_string(scaledMask.cols) + ", ";
                json += "\"height\": " + std::to_string(scaledMask.rows) + ", ";
                json += "\"data\": \"" + maskBase64 + "\"";
                json += "}";
                LOGI("[detect_table_bgra] Added mask to response: %dx%d", scaledMask.cols,
                     scaledMask.rows);
            } catch (const std::exception& e) {
                LOGE("[detect_table_bgra] Failed to encode mask: %s", e.what());
            }
        }

        json += "}";

#if DEBUG_OUTPUT

        // Write debug image showing detected quad
        if (scaledQuadPoints.size() == 4) {
            cv::Mat debugImage;
            cv::cvtColor(bgraImageUnrotated, debugImage, cv::COLOR_BGRA2BGR);

            // Draw the quad with colored edges
            // Edge definitions: 0=TL, 1=TR, 2=BR, 3=BL
            cv::line(debugImage, scaledQuadPoints[0], scaledQuadPoints[1], cv::Scalar(0, 0, 255),
                     3);  // Top: Red
            cv::line(debugImage, scaledQuadPoints[1], scaledQuadPoints[2], cv::Scalar(0, 255, 0),
                     3);  // Right: Green
            cv::line(debugImage, scaledQuadPoints[2], scaledQuadPoints[3], cv::Scalar(255, 0, 0),
                     3);  // Bottom: Blue
            cv::line(debugImage, scaledQuadPoints[3], scaledQuadPoints[0], cv::Scalar(0, 255, 255),
                     3);  // Left: Yellow

            // Draw corner points with numbers
            for (int i = 0; i < 4; i++) {
                cv::circle(debugImage, scaledQuadPoints[i], 10, cv::Scalar(255, 255, 255), -1);
                cv::putText(debugImage, std::to_string(i), scaledQuadPoints[i] + cv::Point2f(-5, 5),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
            }

#if defined(PLATFORM_MACOS) || defined(PLATFORM_LINUX) || defined(PLATFORM_WINDOWS)
            cv::imwrite("/tmp/table_debug.jpg", debugImage);
            LOGI("[detect_table_bgra] Wrote debug image to /tmp/table_debug.jpg");
#else
            cv::imwrite("/sdcard/Download/table_debug.jpg", debugImage);
            LOGI("[detect_table_bgra] Wrote debug image to /sdcard/Download/table_debug.jpg");
#endif
        }
#endif

        resultStr = json;
        return resultStr.c_str();

    } catch (const std::exception& e) {
        LOGE("Error in detect_table_bgra: %s", e.what());
        resultStr =
            "{\"error\": \"Exception in detect_table_bgra: " + std::string(e.what()) + "\"}";
        return resultStr.c_str();
    }
}

const char* transform_points_using_quad(const float* pointsData, int pointsCount,
                                        const float* quadData, int quadCount, int imageWidth,
                                        int imageHeight, int displayWidth, int displayHeight,
                                        int inputRotationDegrees) {
    static std::string resultStr;

    try {
        if (pointsCount == 0 || quadCount != 4) {
            resultStr = "{\"error\": \"Invalid input parameters\"}";
            return resultStr.c_str();
        }

        std::vector<cv::Point2f> points;
        for (int i = 0; i < pointsCount; i++) {
            points.emplace_back(pointsData[i * 2], pointsData[i * 2 + 1]);
        }

        std::vector<cv::Point2f> quadPoints;
        for (int i = 0; i < quadCount; i++) {
            quadPoints.emplace_back(quadData[i * 2], quadData[i * 2 + 1]);
        }

        // Log input parameters received from Dart
        LOGI("[transform_points] === C++ TRANSFORM INPUT ===");
        LOGI("[transform_points] Received %d ball points, %d quad points", pointsCount, quadCount);
        LOGI("[transform_points] Image size: %dx%d, Display size: %dx%d", imageWidth, imageHeight,
             displayWidth, displayHeight);
        LOGI("[transform_points] Input rotation: %d degrees", inputRotationDegrees);
        LOGI("[transform_points] Quad points (in image %dx%d space):", imageWidth, imageHeight);
        for (int i = 0; i < quadCount; i++) {
            LOGI("  Quad[%d]: (%.1f, %.1f)", i, quadPoints[i].x, quadPoints[i].y);
        }
        LOGI("[transform_points] Ball positions (in image %dx%d space, before transform):",
             imageWidth, imageHeight);
        int ballsToLog = std::min(5, pointsCount);
        for (int i = 0; i < ballsToLog; i++) {
            LOGI("  Ball[%d]: (%.1f, %.1f)", i, points[i].x, points[i].y);
        }
        if (pointsCount > 5) {
            LOGI("  ... and %d more balls", pointsCount - 5);
        }

        TransformationResult transformation =
            transformPointsUsingQuad(points, quadPoints, cv::Size(imageWidth, imageHeight),
                                     cv::Size(displayWidth, displayHeight), inputRotationDegrees);

        // Log transformed output
        LOGI("[transform_points] === C++ TRANSFORM OUTPUT ===");
        LOGI("[transform_points] Orientation detected: %s",
             QuadAnalysis::orientationToString(transformation.orientation).c_str());
        LOGI("[transform_points] Needs rotation: %s",
             transformation.needsRotation ? "true" : "false");
        LOGI("[transform_points] Transformed ball positions (in display %dx%d space):",
             displayWidth, displayHeight);
        int transformedToLog = std::min(5, (int)transformation.transformedPoints.size());
        for (int i = 0; i < transformedToLog; i++) {
            LOGI("  Ball[%d]: (%.1f, %.1f)", i, transformation.transformedPoints[i].x,
                 transformation.transformedPoints[i].y);
        }
        if ((int)transformation.transformedPoints.size() > 5) {
            LOGI("  ... and %d more balls", (int)transformation.transformedPoints.size() - 5);
        }
        LOGI("[transform_points] === END TRANSFORM ===");

        std::string json = "{";
        json += "\"transformed_points\": [";
        for (size_t i = 0; i < transformation.transformedPoints.size(); ++i) {
            const auto& pt = transformation.transformedPoints[i];
            json += "{\"x\": " + std::to_string(pt.x) + ", \"y\": " + std::to_string(pt.y) + "}";
            if (i < transformation.transformedPoints.size() - 1) json += ", ";
        }
        json += "], ";
        json +=
            "\"needs_rotation\": " + std::string(transformation.needsRotation ? "true" : "false");
        json += ", \"orientation\": \"" +
                QuadAnalysis::orientationToString(transformation.orientation) + "\"";
        json += "}";

        resultStr = json;
        return resultStr.c_str();

    } catch (const std::exception& e) {
        resultStr = "{\"error\": \"" + std::string(e.what()) + "\"}";
        return resultStr.c_str();
    }
}

const char* normalize_image_bgra(const unsigned char* inputBytes, int inputWidth, int inputHeight,
                                 int inputStride, int rotationDegrees, unsigned char* outputBytes,
                                 int outputBufferSize, int channelFormat) {
    static std::string resultStr;

    try {
        LOGI("[normalize_image_bgra] Input: %dx%d, stride: %d, rotation: %d, channelFormat: %d",
             inputWidth, inputHeight, inputStride, rotationDegrees, channelFormat);

        // Create Mat from input bytes
        cv::Mat inputImage(inputHeight, inputWidth, CV_8UC4, (void*)inputBytes, inputStride);
        if (inputImage.empty()) {
            LOGE("[normalize_image_bgra] Failed to create Mat from input bytes");
            resultStr = "{\"error\": \"Failed to create image from input bytes\"}";
            return resultStr.c_str();
        }

        // Convert to BGRA if needed (channelFormat: 0=BGRA, 1=RGBA)
        cv::Mat bgraInputImage;
        if (channelFormat == 1) {
            // Input is RGBA, convert to BGRA
            LOGI("[normalize_image_bgra] Converting RGBA to BGRA");
            cv::cvtColor(inputImage, bgraInputImage, cv::COLOR_RGBA2BGRA);
        } else {
            // Input is already BGRA
            bgraInputImage = inputImage;
        }

        // Apply rotation if needed
        cv::Mat rotatedImage;
        bool wasRotated = false;
        int rotatedWidth = inputWidth;
        int rotatedHeight = inputHeight;

        if (rotationDegrees == 90) {
            cv::rotate(bgraInputImage, rotatedImage, cv::ROTATE_90_CLOCKWISE);
            wasRotated = true;
            rotatedWidth = inputHeight;
            rotatedHeight = inputWidth;
        } else if (rotationDegrees == 270 || rotationDegrees == -90) {
            cv::rotate(bgraInputImage, rotatedImage, cv::ROTATE_90_COUNTERCLOCKWISE);
            wasRotated = true;
            rotatedWidth = inputHeight;
            rotatedHeight = inputWidth;
        } else if (rotationDegrees == 180) {
            cv::rotate(bgraInputImage, rotatedImage, cv::ROTATE_180);
            wasRotated = true;
            rotatedWidth = inputWidth;
            rotatedHeight = inputHeight;
        } else {
            rotatedImage = bgraInputImage;
        }

        LOGI("[normalize_image_bgra] After rotation: %dx%d, wasRotated: %s", rotatedWidth,
             rotatedHeight, wasRotated ? "true" : "false");

        // Calculate 16:9 landscape canvas dimensions
        // Use the larger dimension as the height reference
        const double TARGET_ASPECT_RATIO = 16.0 / 9.0;
        int canvasHeight = std::max(rotatedWidth, rotatedHeight);
        int canvasWidth = static_cast<int>(std::round(canvasHeight * TARGET_ASPECT_RATIO));

        LOGI("[normalize_image_bgra] Canvas dimensions: %dx%d", canvasWidth, canvasHeight);

        // Check if output buffer is large enough
        int requiredSize = canvasWidth * canvasHeight * 4;
        if (outputBufferSize < requiredSize) {
            LOGE("[normalize_image_bgra] Output buffer too small: %d < %d", outputBufferSize,
                 requiredSize);
            resultStr = "{\"error\": \"Output buffer too small\"}";
            return resultStr.c_str();
        }

        // Create black canvas (BGRA format)
        cv::Mat canvas(canvasHeight, canvasWidth, CV_8UC4, cv::Scalar(0, 0, 0, 255));

        // Calculate centered position
        int offsetX = (canvasWidth - rotatedWidth) / 2;
        int offsetY = (canvasHeight - rotatedHeight) / 2;

        LOGI("[normalize_image_bgra] Image offset on canvas: (%d, %d)", offsetX, offsetY);

        // Copy rotated image onto canvas at centered position
        cv::Rect roi(offsetX, offsetY, rotatedWidth, rotatedHeight);
        rotatedImage.copyTo(canvas(roi));

        // Convert final canvas from BGRA to RGBA for a consistent pipeline output
        cv::Mat rgbaCanvas;
        cv::cvtColor(canvas, rgbaCanvas, cv::COLOR_BGRA2RGBA);

        // Copy canvas data to output buffer
        if (rgbaCanvas.isContinuous()) {
            std::memcpy(outputBytes, rgbaCanvas.data, requiredSize);
        } else {
            // Copy row by row if not continuous
            for (int y = 0; y < canvasHeight; ++y) {
                std::memcpy(outputBytes + y * canvasWidth * 4, rgbaCanvas.ptr(y), canvasWidth * 4);
            }
        }

        LOGI("[normalize_image_bgra] Normalization complete");

        // Build JSON result with metadata
        std::string json = "{";
        json += "\"width\": " + std::to_string(canvasWidth) + ", ";
        json += "\"height\": " + std::to_string(canvasHeight) + ", ";
        json += "\"stride\": " + std::to_string(canvasWidth * 4) + ", ";
        json += "\"offset_x\": " + std::to_string(offsetX) + ", ";
        json += "\"offset_y\": " + std::to_string(offsetY) + ", ";
        json += "\"was_rotated\": " + std::string(wasRotated ? "true" : "false");
        json += "}";

        resultStr = json;
        return resultStr.c_str();

    } catch (const std::exception& e) {
        LOGE("[normalize_image_bgra] Exception: %s", e.what());
        resultStr = "{\"error\": \"" + std::string(e.what()) + "\"}";
        return resultStr.c_str();
    }
}
}
