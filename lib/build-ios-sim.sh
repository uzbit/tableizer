#!/bin/bash

# Build script for iOS Simulator
# This builds both OpenCV and the tableizer library for iOS Simulator

set -e  # Exit on any error

echo "=== Building OpenCV and Tableizer for iOS Simulator ==="

# Set deployment target
export IPHONEOS_DEPLOYMENT_TARGET=18.5

# Navigate to OpenCV directory
cd /Users/uzbit/Documents/projects/tableizer/lib/libs/opencv

# Create symlink for path compatibility (if it doesn't exist)
if [ ! -L opencv ]; then
    echo "Creating opencv symlink..."
    ln -s . opencv
fi

echo "=== Step 1: Building OpenCV for iOS Simulator ==="

# Clean and rebuild OpenCV
rm -rf build-sim
mkdir -p build-sim && cd build-sim

# Configure OpenCV for iOS Simulator
IPHONEOS_DEPLOYMENT_TARGET=18.5 cmake -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=../platforms/ios/cmake/Toolchains/Toolchain-iPhoneSimulator_Xcode.cmake \
    -DIOS_ARCH=arm64 \
    -DBUILD_LIST=core,imgproc,photo,features2d,video,calib3d,dnn,imgcodecs \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_APPS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_objc=OFF \
    -DBUILD_objc_bindings_generator=OFF \
    -DBUILD_opencv_java=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_ANDROID_PROJECTS=OFF \
    -DWITH_TBB=OFF \
    -DWITH_OPENMP=OFF \
    -DWITH_IPP=OFF \
    -DWITH_ITT=OFF \
    -DWITH_EIGEN=ON \
    -DOPENCV_ENABLE_PARALLEL=OFF \
    -DENABLE_NEON=ON \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=18.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_FRAMEWORK=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IMGCODEC_IOS_CONVERSIONS=OFF \
    -DWITH_IMGCODEC_APPLE_CONVERSIONS=OFF \
    ..

# Build OpenCV
echo "Building OpenCV libraries..."
ninja

echo "OpenCV build complete! Libraries available at:"
find lib -name "*.a" | head -5

echo ""
echo "=== Step 2: Building Tableizer Library for iOS Simulator ==="

# Navigate back to lib directory
cd /Users/uzbit/Documents/projects/tableizer/lib

# Clean any previous iOS builds
rm -rf build-ios

# Configure tableizer library for iOS simulator
mkdir -p build-ios && cd build-ios

cmake -GXcode \
    -DCMAKE_TOOLCHAIN_FILE=../libs/opencv/platforms/ios/cmake/Toolchains/Toolchain-iPhoneSimulator_Xcode.cmake \
    -DIPHONEOS_DEPLOYMENT_TARGET=18.5 \
    -DIOS_ARCH=arm64 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DBUILD_SHARED_LIB=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ..

# Build the tableizer library
echo "Building tableizer library..."
xcodebuild -target tableizer_lib -configuration Release -sdk iphonesimulator ARCHS=arm64 ONLY_ACTIVE_ARCH=NO

echo ""
echo "=== Build Complete! ==="
echo "OpenCV libraries: $(pwd)/../libs/opencv/build-sim/lib/"
echo "Tableizer library: $(pwd)/lib/Release-iphonesimulator/"
echo ""
echo "Your iOS Simulator libraries are ready to use!"