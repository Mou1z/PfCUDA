set -e
BUILD_DIR="build"

if [ -d "$BUILD_DIR" ]; then
    echo "--- Removing existing $BUILD_DIR directory ---"
    rm -rf "$BUILD_DIR"
fi

echo "--- Creating new $BUILD_DIR directory ---"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "--- Running CMake ---"
cmake ..

echo "--- Compiling with $(nproc) cores ---"
make -j$(nproc)

echo "--- Build Complete! ---"
echo "Output library: $(pwd)/libcupfaffian.so"