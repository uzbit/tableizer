cmake_minimum_required(VERSION 3.10)
project(Tableizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Build Options ---
option(BUILD_SHARED_LIB "Build a shared library for cross-platform use" OFF)

# --- Source Files ---
set(LIB_SOURCES
    src/tableizer.cpp
    src/table_detector.cpp
    src/ball_detector.cpp
    src/utilities.cpp
)
set(APP_SOURCES
    ${LIB_SOURCES}
    src/main.cpp
)

# --- Platform-Specific Configuration ---
if(BUILD_SHARED_LIB)
    message(STATUS "Building shared library")
    add_definitions(-DBUILD_SHARED_LIB)
    # --- Cross-Platform Dependency Paths ---
    # export ANDROID_NDK_HOME=$ANDROID_SDK_ROOT/ndk/29.0.13599879 
    #     cmake -S . -B build-android \
    #   -DCMAKE_BUILD_TYPE=Release \
    #   -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    #   -DANDROID_ABI="arm64-v8a" \
    #   -DANDROID_PLATFORM="android-24" \
    #   -DBUILD_SHARED_LIB=ON \
    #   -DOpenCV_DIR="/Users/uzbit/Documents/projects/tableizer/lib/libs/OpenCV-android-sdk/sdk/native/jni" \
    #   -DONNXRUNTIME_INCLUDE_DIR="/Users/uzbit/Documents/projects/other/onnxruntime/include" \
    #   -DONNXRUNTIME_LIBRARY_PATH="/Users/uzbit/Documents/projects/other/onnxruntime/build-android/Release/libonnxruntime.so"

    if(ANDROID)
        message(STATUS "Configuring for Android")
        find_package(OpenCV REQUIRED)
        if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR OR NOT DEFINED ONNXRUNTIME_LIBRARY_PATH)
            message(FATAL_ERROR "For Android builds, you must define ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY_PATH")
        endif()
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=armv8-a")
    elseif(IOS)
        message(STATUS "Configuring for iOS")
        # Add logic for iOS build (e.g., find_package or manual paths)
        if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR OR NOT DEFINED ONNXRUNTIME_LIBRARY_PATH)
            message(FATAL_ERROR "For iOS builds, you must define ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY_PATH")
        endif()
    else()
        message(STATUS "Configuring for local library build (e.g., Linux/Windows)")
        set(ONNXRuntime_ROOT "/opt/homebrew") # Local dev path
        set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRuntime_ROOT}/include)
        set(ONNXRUNTIME_LIBRARY_PATH ${ONNXRuntime_ROOT}/lib/libonnxruntime.dylib) # Be specific
        find_package(OpenCV REQUIRED)
    endif()

    add_library(tableizer_lib SHARED ${LIB_SOURCES})
    target_compile_options(tableizer_lib PRIVATE
        $<$<CONFIG:Debug>:-g>
        $<$<CONFIG:Release>:-O2 -DNDEBUG -march=armv8-a -fno-tree-vectorize>
        -Wall
        -Wextra
    )

    target_include_directories(tableizer_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(tableizer_lib
        ${OpenCV_LIBS}
        ${ONNXRUNTIME_LIBRARY_PATH}
    )

else()
    message(STATUS "Building executable for macOS")

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

    add_library(tableizer_lib STATIC ${LIB_SOURCES})
    target_include_directories(tableizer_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRuntime_ROOT}/include
    )
    target_link_libraries(tableizer_lib PUBLIC
        ${OpenCV_LIBS}
        onnxruntime
    )

    add_executable(tableizer_app src/main.cpp)

    # Link OpenCV and ONNX Runtime manually
    target_link_libraries(tableizer_app PRIVATE
        tableizer_lib
    )
endif()

# Always add the test subdirectory
add_subdirectory(test)

