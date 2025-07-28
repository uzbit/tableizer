# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(Tableizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_DIR to the location of the OpenCV Android SDK
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/OpenCV-android-sdk/sdk/native/jni/abi-arm64-v8a)

# Find dependencies
find_package(OpenCV REQUIRED)

# Set ONNX Runtime include and library paths
set(ONNXRUNTIME_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/onnxruntime/include)
set(ONNXRUNTIME_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime/jniLibs/arm64-v8a/libonnxruntime.so)


# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

add_library(tableizer SHARED
    src/tableizer.cpp
    src/table_detector.cpp
    src/ball_detector.cpp
    src/utilities.cpp
    src/tableizer_ffi.cpp
)

target_include_directories(tableizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(tableizer
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    ${log-lib}
    ${android-lib}
)

install(TARGETS tableizer DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build)

# A message to the user
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNXRuntime library: ${ONNXRUNTIME_LIBRARY}")
