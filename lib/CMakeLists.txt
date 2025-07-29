cmake_minimum_required(VERSION 3.10)
project(Tableizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# You can customize this to your installation
set(ONNXRuntime_ROOT "/opt/homebrew")

# Add ONNX Runtime paths manually
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRuntime_ROOT}/include)
link_directories(${ONNXRuntime_ROOT}/lib)

# Find OpenCV (using pkg-config or Homebrew)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

add_executable(tableizer_app 
    src/main.cpp 
    src/tableizer.cpp 
    src/table_detector.cpp 
    src/ball_detector.cpp 
    src/utilities.cpp
)

# Link OpenCV and ONNX Runtime manually
target_link_libraries(tableizer_app
    ${OpenCV_LIBS}
    onnxruntime  # assumes libonnxruntime.dylib or libonnxruntime.so is in ONNXRuntime_ROOT/lib
)