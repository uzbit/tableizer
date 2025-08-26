#!/bin/bash

# Complete Android Build Script for Tableizer
# Builds OpenCV + Tableizer native library for Android

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
echo_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PROJECT_ROOT="/Users/uzbit/Documents/projects/tableizer"
LIB_ROOT="$PROJECT_ROOT/lib"
OPENCV_SRC="$LIB_ROOT/libs/opencv"
OPENCV_BUILD="$OPENCV_SRC/build-android"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM="android-24"
BUILD_TYPE="Release"

# Check prerequisites
echo_info "Checking build prerequisites..."

if [[ -z "$ANDROID_NDK_HOME" ]]; then
    echo_error "ANDROID_NDK_HOME environment variable not set"
    echo "Please set it to your Android NDK installation path"
    echo "Example: export ANDROID_NDK_HOME=\$ANDROID_SDK_ROOT/ndk/29.0.13599879"
    exit 1
fi

if [[ ! -d "$ANDROID_NDK_HOME" ]]; then
    echo_error "Android NDK not found at: $ANDROID_NDK_HOME"
    exit 1
fi

if [[ ! -f "$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" ]]; then
    echo_error "Android NDK toolchain file not found"
    exit 1
fi

# Check for required tools
for tool in cmake make; do
    if ! command -v $tool &> /dev/null; then
        echo_error "$tool is required but not installed"
        exit 1
    fi
done

echo_success "All prerequisites met"

# Change to library directory
cd "$LIB_ROOT"

# Step 1: Build OpenCV for Android
echo_info "Building OpenCV for Android..."

if [[ ! -d "$OPENCV_SRC" ]]; then
    echo_error "OpenCV source not found at: $OPENCV_SRC"
    exit 1
fi

# Create OpenCV build directory
mkdir -p "$OPENCV_BUILD"
cd "$OPENCV_SRC"

echo_info "Configuring OpenCV build..."
cmake -S . -B "$OPENCV_BUILD" \
    -D CMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -D CMAKE_SYSTEM_NAME=Android \
    -D CMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -D ANDROID_ABI="$ANDROID_ABI" \
    -D ANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_LIST=core,imgproc,photo,features2d,video,calib3d,dnn,imgcodecs \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_java=OFF \
    -D BUILD_ANDROID_PROJECTS=OFF \
    -D WITH_TBB=OFF \
    -D WITH_OPENMP=OFF \
    -D WITH_IPP=ON \
    -D WITH_ITT=ON \
    -D WITH_EIGEN=ON \
    -D OPENCV_ENABLE_PARALLEL=OFF \
    -D ENABLE_NEON=ON \
    -D CMAKE_CXX_FLAGS="-march=armv8-a -mcpu=cortex-a55+crypto -O3 -ffast-math" \
    -D CMAKE_INSTALL_PREFIX="$OPENCV_BUILD/install"

echo_info "Building OpenCV (this may take several minutes)..."
cmake --build "$OPENCV_BUILD" --parallel $(nproc)

echo_info "Installing OpenCV..."
cmake --install "$OPENCV_BUILD"

echo_success "OpenCV build completed"

# Step 2: Build Tableizer library
echo_info "Building Tableizer native library..."

cd "$LIB_ROOT"

# Clean previous build
rm -rf build-android

echo_info "Configuring Tableizer library..."
cmake -S . -B build-android \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DBUILD_SHARED_LIB=ON \
    -DOpenCV_DIR="$OPENCV_BUILD" \
    -DONNXRUNTIME_INCLUDE_DIR="$PROJECT_ROOT/lib/libs/onnxruntime/onnxruntime" \
    -DONNXRUNTIME_LIBRARY_PATH="$PROJECT_ROOT/lib/libs/onnxruntime/jniLibs/$ANDROID_ABI/libonnxruntime.so"

echo_info "Building Tableizer library..."
cmake --build build-android --parallel $(nproc)

echo_success "Tableizer library build completed"

# Step 3: Copy libraries to Flutter app
echo_info "Copying libraries to Flutter app..."

FLUTTER_JNI_LIBS="$PROJECT_ROOT/app/android/app/src/main/jniLibs/$ANDROID_ABI"
mkdir -p "$FLUTTER_JNI_LIBS"

# Copy OpenCV libraries
echo_info "Copying OpenCV libraries..."
if [[ -d "$OPENCV_BUILD/lib" ]]; then
    cp "$OPENCV_BUILD/lib"/libopencv_*.so "$FLUTTER_JNI_LIBS/" 2>/dev/null || \
    cp "$OPENCV_BUILD/lib/$ANDROID_ABI"/libopencv_*.so "$FLUTTER_JNI_LIBS/" 2>/dev/null || \
    echo_warning "OpenCV libraries not found in expected locations"
else
    echo_warning "OpenCV build directory not found: $OPENCV_BUILD/lib"
fi

# Copy Tableizer library
echo_info "Copying Tableizer library..."
cp build-android/libtableizer_lib.so "$FLUTTER_JNI_LIBS/"

# Copy ONNX Runtime library (if it exists)
ONNX_LIB="$PROJECT_ROOT/lib/libs/onnxruntime/jniLibs/$ANDROID_ABI/libonnxruntime.so"
if [[ -f "$ONNX_LIB" ]]; then
    echo_info "Copying ONNX Runtime library..."
    cp "$ONNX_LIB" "$FLUTTER_JNI_LIBS/"
else
    echo_warning "ONNX Runtime library not found at: $ONNX_LIB"
fi

echo_success "All libraries copied to Flutter app"

# Step 4: Verification
echo_info "Verifying built libraries..."

echo_info "Libraries in Flutter app:"
ls -la "$FLUTTER_JNI_LIBS"

echo_info "Checking library symbols..."
if command -v objdump &> /dev/null; then
    echo_info "Tableizer library exports:"
    objdump -T "$FLUTTER_JNI_LIBS/libtableizer_lib.so" | grep -E "(detect_table_bgra|initialize_detector)" || echo "Symbols not found"
fi

echo_success "✅ Android build completed successfully!"
echo_info "Next steps:"
echo "  1. Run: flutter build apk --debug"
echo "  2. Or: flutter run --debug -d <android-device>"
echo ""
echo_info "Build artifacts:"
echo "  - OpenCV: $OPENCV_BUILD"
echo "  - Tableizer: $LIB_ROOT/build-android"
echo "  - Flutter libs: $FLUTTER_JNI_LIBS"