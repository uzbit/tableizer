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
    src/base64_utils.cpp
    src/quad_analysis.cpp
    src/ball_detection.cpp
    src/ffi_api.cpp
    src/image_processing.cpp
    src/json_parser.cpp
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
    # cmake -S . -B build-android \
    #   -DCMAKE_BUILD_TYPE=Release \
    #   -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    #   -DANDROID_ABI="arm64-v8a" \
    #   -DANDROID_PLATFORM="android-24" \
    #   -DBUILD_SHARED_LIB=ON \
    #   -DOpenCV_DIR="/Users/uzbit/Documents/projects/tableizer/lib/libs/opencv/build-android-opt" \
    #   -DONNXRUNTIME_INCLUDE_DIR="/Users/uzbit/Documents/projects/other/onnxruntime/include" \
    #   -DONNXRUNTIME_LIBRARY_PATH="/Users/uzbit/Documents/projects/other/onnxruntime/build-android/Release/libonnxruntime.so"

    if(ANDROID)
        message(STATUS "Configuring for Android")
        find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn imgcodecs)
        if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR OR NOT DEFINED ONNXRUNTIME_LIBRARY_PATH)
            message(FATAL_ERROR "For Android builds, you must define ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY_PATH")
        endif()
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=armv8-a")
    elseif(IOS)
        message(STATUS "Configuring for iOS")
        
        # Use locally built OpenCV static libraries
        # Use build-device for iOS device, build-sim for iOS simulator
        # Check for SDK to determine device vs simulator build
        if(CMAKE_OSX_SYSROOT MATCHES "iPhoneOS" OR CMAKE_OSX_SYSROOT MATCHES "iphoneos")
            set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/opencv/build-device")
            message(STATUS "Using OpenCV from build-device for iOS device (SDK: ${CMAKE_OSX_SYSROOT})")
        else()
            set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/opencv/build-sim")
            message(STATUS "Using OpenCV from build-sim for iOS simulator (SDK: ${CMAKE_OSX_SYSROOT})")
        endif()
        find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn imgcodecs photo features2d video calib3d)
        
        # Use CocoaPods-installed ONNX Runtime
        if(NOT DEFINED IOS_PODS_ROOT)
            set(IOS_PODS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../app/ios/Pods")
        endif()
        
        set(ONNXRUNTIME_INCLUDE_DIR "${IOS_PODS_ROOT}/onnxruntime-c/Headers")
        set(ONNXRUNTIME_XCFRAMEWORK_PATH "${IOS_PODS_ROOT}/onnxruntime-c/onnxruntime.xcframework")
        
        message(STATUS "Using OpenCV from: ${OpenCV_DIR}")
        message(STATUS "Using ONNX Runtime from CocoaPods at: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime XCFramework: ${ONNXRUNTIME_XCFRAMEWORK_PATH}")
    else()
        message(STATUS "Configuring for local library build (e.g., Linux/Windows)")
        set(ONNXRuntime_ROOT "/opt/homebrew") # Local dev path
        set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRuntime_ROOT}/include/onnxruntime)
        set(ONNXRUNTIME_LIBRARY_PATH ${ONNXRuntime_ROOT}/lib/libonnxruntime.dylib) # Be specific
        find_package(OpenCV REQUIRED NO_MODULE)
    endif()

    add_library(tableizer_lib SHARED ${LIB_SOURCES})
    set_source_files_properties(src/table_detector.cpp PROPERTIES COMPILE_FLAGS "-O0")
    add_compile_options(-O3 -fPIC -march=armv8-a -DOPENCV_DISABLE_TBB=ON -DWITH_PTHREADS_PF=OFF -DOPENCV_DISABLE_OPENMP_SUPPORT=ON)
    # target_compile_options(tableizer_lib PRIVATE
    #     $<$<CONFIG:Debug>:-g>
    #     $<$<CONFIG:Release>:-g -DNDEBUG>
    #     -Wall
    #     -Wextra
    # )

    target_include_directories(tableizer_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    if(IOS)
        target_link_libraries(tableizer_lib
            ${OpenCV_LIBS}
            "${ONNXRUNTIME_XCFRAMEWORK_PATH}"
            "-framework Foundation"
        )
    else()
        target_link_libraries(tableizer_lib
            ${OpenCV_LIBS}
            ${ONNXRUNTIME_LIBRARY_PATH}
            $<$<PLATFORM_ID:Android>:log>
        )
    endif()

else()
    message(STATUS "Building executable for macOS")

    # You can customize this to your installation
    set(ONNXRuntime_ROOT "/opt/homebrew")

    # Add ONNX Runtime paths manually
    include_directories(${CMAKE_SOURCE_DIR}/include)
    include_directories(${ONNXRuntime_ROOT}/include/onnxruntime)
    link_directories(${ONNXRuntime_ROOT}/lib)

    # Find OpenCV (using pkg-config or Homebrew)
    find_package(OpenCV REQUIRED)

    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

    add_library(tableizer_lib STATIC ${LIB_SOURCES})
    target_include_directories(tableizer_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRuntime_ROOT}/include/onnxruntime
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

