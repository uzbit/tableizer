#!/bin/bash

# Build script for iOS Device
# This builds both OpenCV and the tableizer library for actual iOS devices

set -e  # Exit on any error

echo "=== Building OpenCV and Tableizer for iOS Device ==="

# Set deployment target
export IPHONEOS_DEPLOYMENT_TARGET=18.5

# Navigate to OpenCV directory
cd /Users/uzbit/Documents/projects/tableizer/lib/libs/opencv

# Create symlink for path compatibility (if it doesn't exist)
if [ ! -L opencv ]; then
    echo "Creating opencv symlink..."
    ln -s . opencv
fi

echo "=== Step 1: Building OpenCV for iOS Device ==="

# Clean and rebuild OpenCV
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

# Build OpenCV
echo "Building OpenCV libraries..."
ninja

echo "OpenCV build complete! Libraries available at:"
find lib -name "*.a" | head -5

echo ""
echo "=== Step 2: Building Tableizer Library for iOS Device ==="

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

# Build the tableizer library (without code signing for device)
echo "Building tableizer library..."
xcodebuild -target tableizer_lib -configuration Release -sdk iphoneos ARCHS=arm64 ONLY_ACTIVE_ARCH=NO CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO

echo ""
echo "=== Step 3: Copy Libraries to Flutter App ==="

# Copy the built library to Flutter app
FLUTTER_IOS_DIR="/Users/uzbit/Documents/projects/tableizer/app/ios"
BUILT_LIB_PATH="$(pwd)/Release-iphoneos/libtableizer_lib.dylib"

if [ -f "$BUILT_LIB_PATH" ]; then
    echo "Copying iOS device library to Flutter app..."
    cp "$BUILT_LIB_PATH" "$FLUTTER_IOS_DIR/"
    echo "Library copied to: $FLUTTER_IOS_DIR/libtableizer_lib.dylib"
else
    echo "Warning: Built library not found at $BUILT_LIB_PATH"
    echo "Available files in Release-iphoneos:"
    ls -la Release-iphoneos/ || echo "Directory not found"
fi

echo ""
echo "=== Build Complete! ==="
echo "OpenCV libraries: $(pwd)/../libs/opencv/build-device/lib/"
echo "Tableizer library: $(pwd)/Release-iphoneos/"
echo ""
echo "Your iOS Device libraries are ready to use!"
echo ""
echo "Next steps:"
echo "  1. Connect an iOS device via USB"
echo "  2. Run: flutter run -d <ios-device-id>"
echo "  3. Or: flutter build ios --release"