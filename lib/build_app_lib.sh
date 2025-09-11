#!/bin/bash

# Tableizer Build Script
# Builds both the C++ shared library and the macOS executable app

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}🏗️  Tableizer Build Script${NC}"
echo -e "Library directory: $LIB_DIR"
echo ""

# Function to print section headers
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_header "Checking Prerequisites"

if ! command_exists cmake; then
    echo -e "${RED}❌ CMake not found. Please install CMake.${NC}"
    exit 1
fi

if ! command_exists make; then
    echo -e "${RED}❌ Make not found. Please install build tools.${NC}"
    exit 1
fi


echo -e "${GREEN}✅ All prerequisites found${NC}"
echo ""

# Parse command line arguments
BUILD_SHARED_LIB=true
BUILD_MACOS_APP=true
BUILD_TYPE="Release"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --lib-only)
            BUILD_MACOS_APP=false
            shift
            ;;
        --app-only)
            BUILD_SHARED_LIB=false
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --lib-only    Build only the shared C++ library (.dylib)"
            echo "  --app-only    Build only the macOS executable app"
            echo "  --debug       Build in Debug mode (default: Release)"
            echo "  --clean       Clean build directories before building"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Build configuration:${NC}"
echo -e "  Build shared library: $BUILD_SHARED_LIB"
echo -e "  Build macOS app: $BUILD_MACOS_APP"
echo -e "  Build type: $BUILD_TYPE"
echo -e "  Clean build: $CLEAN"
echo ""

# Determine what to build based on flags
BUILD_COMMANDS=()
if [ "$BUILD_SHARED_LIB" = true ] && [ "$BUILD_MACOS_APP" = true ]; then
    print_header "Building C++ Shared Library and macOS App"
    BUILD_COMMANDS=("lib" "app")
elif [ "$BUILD_SHARED_LIB" = true ]; then
    print_header "Building C++ Shared Library Only"
    BUILD_COMMANDS=("lib")
elif [ "$BUILD_MACOS_APP" = true ]; then
    print_header "Building macOS App Only"
    BUILD_COMMANDS=("app")
else
    echo -e "${RED}❌ Nothing to build! Enable at least one build option.${NC}"
    exit 1
fi

cd "$LIB_DIR"

# Always clean when building both targets or when explicitly requested
if [ "$CLEAN" = true ] || [ ! -d "build" ] || ([ ${#BUILD_COMMANDS[@]} -gt 1 ]); then
    echo -e "${YELLOW}🧹 Cleaning build directory...${NC}"
    rm -rf build
fi

mkdir -p build
cd build

# Unset any iOS-related environment variables that might interfere
unset IPHONEOS_DEPLOYMENT_TARGET
unset IOS_PLATFORM
unset CMAKE_TOOLCHAIN_FILE

# Build shared library first (if requested)
if [[ " ${BUILD_COMMANDS[@]} " =~ " lib " ]]; then
    echo -e "${YELLOW}🔧 Configuring CMake for shared library...${NC}"
    cmake .. \
        -DBUILD_SHARED_LIB=ON \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_SYSTEM_NAME=Darwin \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
        -G "Unix Makefiles"
    
    echo -e "${YELLOW}🔨 Building shared library...${NC}"
    make -j$(sysctl -n hw.ncpu)
    
    # Check if library was built successfully
    if [ -f "libtableizer_lib.dylib" ]; then
        echo -e "${GREEN}✅ Shared library built successfully: libtableizer_lib.dylib${NC}"
        echo -e "${GREEN}   Size: $(du -h libtableizer_lib.dylib | cut -f1)${NC}"
        if [ -f "libtableizer_lib.a" ]; then
            echo -e "${YELLOW}ℹ️  Note: Static library also created: libtableizer_lib.a${NC}"
        fi
    else
        echo -e "${RED}❌ Shared library build failed!${NC}"
        echo -e "${RED}   Files found: $(ls -la libtableizer_lib.* 2>/dev/null || echo 'none')${NC}"
        exit 1
    fi
    echo ""
fi

# Build macOS app (if requested)
if [[ " ${BUILD_COMMANDS[@]} " =~ " app " ]]; then
    # Clean and reconfigure for macOS app if we also built the library
    if [[ " ${BUILD_COMMANDS[@]} " =~ " lib " ]]; then
        echo -e "${YELLOW}🧹 Preserving shared library and cleaning for macOS app build...${NC}"
        # Save the shared library if it exists
        if [ -f "libtableizer_lib.dylib" ]; then
            cp libtableizer_lib.dylib ../libtableizer_lib.dylib.backup
        fi
        rm -rf *
        # Restore the shared library
        if [ -f "../libtableizer_lib.dylib.backup" ]; then
            mv ../libtableizer_lib.dylib.backup ./libtableizer_lib.dylib
        fi
    fi
    
    echo -e "${YELLOW}🔧 Configuring CMake for macOS app...${NC}"
    cmake .. \
        -DBUILD_SHARED_LIB=OFF \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_SYSTEM_NAME=Darwin \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 \
        -G "Unix Makefiles"
    
    echo -e "${YELLOW}🔨 Building macOS app...${NC}"
    make -j$(sysctl -n hw.ncpu)
    
    # Check if app was built successfully
    if [ -f "tableizer_app" ]; then
        echo -e "${GREEN}✅ macOS app built successfully: tableizer_app${NC}"
    else
        echo -e "${RED}❌ macOS app build failed!${NC}"
        exit 1
    fi
    echo ""
fi


# Summary
print_header "Build Summary"

if [[ " ${BUILD_COMMANDS[@]} " =~ " lib " ]]; then
    LIB_PATH="$LIB_DIR/build/libtableizer_lib.dylib"
    if [ -f "$LIB_PATH" ]; then
        echo -e "${GREEN}📚 Shared Library: $LIB_PATH${NC}"
        echo -e "   Size: $(du -h "$LIB_PATH" | cut -f1)"
    else
        echo -e "${RED}❌ Shared library not found at: $LIB_PATH${NC}"
    fi
fi

if [[ " ${BUILD_COMMANDS[@]} " =~ " app " ]]; then
    APP_PATH="$LIB_DIR/build/tableizer_app"
    if [ -f "$APP_PATH" ]; then
        echo -e "${GREEN}🖥️  macOS App: $APP_PATH${NC}"
        echo -e "   Size: $(du -h "$APP_PATH" | cut -f1)"
    fi
fi

echo ""
echo -e "${GREEN}🎉 Build completed successfully!${NC}"

# Optional: Run basic tests
if [[ " ${BUILD_COMMANDS[@]} " =~ " lib " ]] || [[ " ${BUILD_COMMANDS[@]} " =~ " app " ]]; then
    echo ""
    echo -e "${YELLOW}🧪 Running basic library test...${NC}"
    cd "$LIB_DIR/build"
    if [ -f "test/test_tableizer" ]; then
        ./test/test_tableizer
    else
        echo -e "${YELLOW}⚠️  No test executable found${NC}"
    fi
fi