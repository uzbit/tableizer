# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(Tableizer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Add executable
add_executable(tableizer_app src/main.cpp src/table_detector.cpp src/ball_detector.cpp)

# Link libraries
target_link_libraries(tableizer_app
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

# Ensure PyTorch libraries are found at runtime
set(CMAKE_INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")

# A message to the user
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
