set -e
BUILD_DIR="build"

if [ -d "$BUILD_DIR" ]; then
    echo "--- Removing existing $BUILD_DIR directory ---"
    rm -rf "$BUILD_DIR"
fi

mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
