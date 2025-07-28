# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(Tableizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path for Homebrew libraries
set(CMAKE_PREFIX_PATH "/opt/homebrew")

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(ONNXRuntime REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /opt/homebrew/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRuntime_INCLUDE_DIRS}
)

# Add executable
add_executable(tableizer_app
    src/main.cpp
    src/tableizer.cpp
    src/table_detector.cpp
    src/ball_detector.cpp
    src/utilities.cpp
)

# Link libraries
target_link_libraries(tableizer_app
    ${OpenCV_LIBS}
    /opt/homebrew/lib/libonnxruntime.dylib
)

# A message to the user
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "ONNXRuntime libraries: ${ONNXRuntime_LIBRARIES}")
