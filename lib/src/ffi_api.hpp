#pragma once

extern "C" {

void* initialize_detector(const char* modelPath);

const char* detect_balls_bgra(void* detectorPtr, const unsigned char* imageBytes, int width,
                              int height, int stride, const float* quadPoints, int quadPointsLength,
                              int channelFormat);

void release_detector(void* detectorPtr);

const char* detect_table_bgra(const unsigned char* imageBytes, int width, int height, int stride,
                              int channelFormat);

const char* transform_points_using_quad(const float* pointsData, int pointsCount,
                                        const float* quadData, int quadCount, int imageWidth,
                                        int imageHeight, int displayWidth, int displayHeight,
                                        int inputRotationDegrees);

const char* normalize_image_bgra(const unsigned char* inputBytes, int inputWidth, int inputHeight,
                                 int inputStride, int rotationDegrees, unsigned char* outputBytes,
                                 int outputBufferSize, int channelFormat);
}
