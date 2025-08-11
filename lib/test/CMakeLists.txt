cmake_minimum_required(VERSION 3.10)
project(TableizerTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version for test: ${OpenCV_VERSION}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${OpenCV_INCLUDE_DIRS}
)

# Add the test executable
add_executable(table_detection_test test_table_detection.cpp)

# Link libraries
target_link_libraries(table_detection_test PRIVATE
    tableizer_lib
    ${OpenCV_LIBS}
)
