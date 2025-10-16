#!/bin/bash

# Unified iOS Build Script
# Builds OpenCV and Tableizer libraries for both iOS Device and iOS Simulator

set -e  # Exit on any error

echo "=== Building OpenCV and Tableizer for iOS (Device + Simulator) ==="

# Set deployment target
export IPHONEOS_DEPLOYMENT_TARGET=18.5

# Navigate to OpenCV directory
cd /Users/uzbit/Documents/projects/tableizer/lib/libs/opencv

# Create symlink for path compatibility (if it doesn't exist)
if [ ! -L opencv ]; then
    echo "Creating opencv symlink..."
    ln -s . opencv
fi

echo ""
echo "=== Step 1: Building OpenCV for iOS Device ==="

# Clean and rebuild OpenCV for device
rm -rf build-device
mkdir -p build-device && cd build-device

# Configure OpenCV for iOS Device (actual iPhone/iPad)
IPHONEOS_DEPLOYMENT_TARGET=18.5 cmake -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=../platforms/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake \
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

# Build OpenCV for device
echo "Building OpenCV libraries for device..."
ninja

echo "OpenCV device build complete!"

# Navigate back to opencv directory
cd /Users/uzbit/Documents/projects/tableizer/lib/libs/opencv

echo ""
echo "=== Step 2: Building OpenCV for iOS Simulator ==="

# Clean and rebuild OpenCV for simulator
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

# Build OpenCV for simulator
echo "Building OpenCV libraries for simulator..."
ninja

echo "OpenCV simulator build complete!"

echo ""
echo "=== Step 3: Building Tableizer Library for iOS Device ==="

# Navigate back to lib directory
cd /Users/uzbit/Documents/projects/tableizer/lib

# Clean any previous iOS device builds
rm -rf build-ios-device

# Configure tableizer library for iOS device
mkdir -p build-ios-device && cd build-ios-device

# Get absolute paths to avoid CMake confusion
OPENCV_DEVICE_DIR="$(pwd)/../libs/opencv/build-device"
echo "Using OpenCV from: $OPENCV_DEVICE_DIR"

# Force CMake to ignore cached values and use only our specified OpenCV
cmake -GXcode \
    -DCMAKE_TOOLCHAIN_FILE=../libs/opencv/platforms/ios/cmake/Toolchains/Toolchain-iPhoneOS_Xcode.cmake \
    -DIPHONEOS_DEPLOYMENT_TARGET=18.5 \
    -DIOS_ARCH=arm64 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DBUILD_SHARED_LIB=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR="$OPENCV_DEVICE_DIR" \
    -DCMAKE_PREFIX_PATH="$OPENCV_DEVICE_DIR" \
    -DOpenCV_FOUND=OFF \
    -DCMAKE_FIND_ROOT_PATH="$OPENCV_DEVICE_DIR" \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO \
    -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
    ..

# Build the tableizer library for device (without code signing)
echo "Building tableizer library for device..."
xcodebuild -target tableizer_lib -configuration Release -sdk iphoneos ARCHS=arm64 ONLY_ACTIVE_ARCH=NO CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO

echo ""
echo "=== Step 4: Building Tableizer Library for iOS Simulator ==="

# Navigate back to lib directory
cd /Users/uzbit/Documents/projects/tableizer/lib

# Clean any previous iOS simulator builds
rm -rf build-ios-sim

# Configure tableizer library for iOS simulator
mkdir -p build-ios-sim && cd build-ios-sim

cmake -GXcode \
    -DCMAKE_TOOLCHAIN_FILE=../libs/opencv/platforms/ios/cmake/Toolchains/Toolchain-iPhoneSimulator_Xcode.cmake \
    -DIPHONEOS_DEPLOYMENT_TARGET=18.5 \
    -DIOS_ARCH=arm64 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DBUILD_SHARED_LIB=ON \
    -DCMAKE_BUILD_TYPE=Release \
    ..

# Build the tableizer library for simulator
echo "Building tableizer library for simulator..."
xcodebuild -target tableizer_lib -configuration Release -sdk iphonesimulator ARCHS=arm64 ONLY_ACTIVE_ARCH=NO

echo ""
echo "=== Step 5: Copy Libraries to Flutter App ==="

# Copy libraries to Flutter app with proper naming
FLUTTER_IOS_DIR="/Users/uzbit/Documents/projects/tableizer/app/ios"

# Device library
DEVICE_LIB_PATH="$(pwd)/../build-ios-device/Release-iphoneos/libtableizer_lib.dylib"
if [ -f "$DEVICE_LIB_PATH" ]; then
    echo "Copying iOS device library to Flutter app..."
    cp "$DEVICE_LIB_PATH" "$FLUTTER_IOS_DIR/libtableizer_lib_device.dylib"
    echo "Device library copied to: $FLUTTER_IOS_DIR/libtableizer_lib_device.dylib"
else
    echo "Warning: Device library not found at $DEVICE_LIB_PATH"
fi

# Simulator library
SIM_LIB_PATH="$(pwd)/Release-iphonesimulator/libtableizer_lib.dylib"
if [ -f "$SIM_LIB_PATH" ]; then
    echo "Copying iOS simulator library to Flutter app..."
    cp "$SIM_LIB_PATH" "$FLUTTER_IOS_DIR/libtableizer_lib_sim.dylib"
    echo "Simulator library copied to: $FLUTTER_IOS_DIR/libtableizer_lib_sim.dylib"
else
    echo "Warning: Simulator library not found at $SIM_LIB_PATH"
fi

# Copy device library as default
if [ -f "$DEVICE_LIB_PATH" ]; then
    echo "Copying device library as default..."
    cp "$DEVICE_LIB_PATH" "$FLUTTER_IOS_DIR/libtableizer_lib.dylib"
    echo "Default library set to: $FLUTTER_IOS_DIR/libtableizer_lib.dylib (device version)"
fi

echo ""
echo "=== Build Complete! ==="
echo "OpenCV device libraries:    $(pwd)/../libs/opencv/build-device/lib/"
echo "OpenCV simulator libraries: $(pwd)/../libs/opencv/build-sim/lib/"
echo "Tableizer device library:   $(pwd)/../build-ios-device/Release-iphoneos/"
echo "Tableizer simulator library: $(pwd)/Release-iphonesimulator/"
echo ""
echo "Libraries copied to Flutter app:"
echo "  Device:    $FLUTTER_IOS_DIR/libtableizer_lib_device.dylib"
echo "  Simulator: $FLUTTER_IOS_DIR/libtableizer_lib_sim.dylib"
echo "  Default:   $FLUTTER_IOS_DIR/libtableizer_lib.dylib (device version)"
echo ""
echo "Your iOS libraries are ready to use!"
echo ""
echo "Next steps:"
echo "  • For device: flutter run -d <ios-device-id> (uses libtableizer_lib.dylib)"
echo "  • For simulator: configure to use libtableizer_lib_sim.dylib"
echo "  • Or: flutter build ios --release"