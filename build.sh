#!/bin/bash
# Build script for ResonanceNet with MSVC environment

MSVC_ROOT="/c/Program Files/Microsoft Visual Studio/18/Insiders/VC/Tools/MSVC/14.50.35717"
SDK_ROOT="/c/Program Files (x86)/Windows Kits/10"
SDK_VER="10.0.26100.0"

export PATH="${MSVC_ROOT}/bin/Hostx64/x64:${PATH}"

export INCLUDE="${MSVC_ROOT}/include;${SDK_ROOT}/Include/${SDK_VER}/ucrt;${SDK_ROOT}/Include/${SDK_VER}/shared;${SDK_ROOT}/Include/${SDK_VER}/um;${SDK_ROOT}/Include/${SDK_VER}/winrt;${SDK_ROOT}/Include/${SDK_VER}/cppwinrt"

export LIB="${MSVC_ROOT}/lib/x64;${SDK_ROOT}/Lib/${SDK_VER}/ucrt/x64;${SDK_ROOT}/Lib/${SDK_VER}/um/x64"

export LIBPATH="${MSVC_ROOT}/lib/x64"

# Find rc.exe
RC_PATH="${SDK_ROOT}/bin/${SDK_VER}/x64"
export PATH="${RC_PATH}:${PATH}"

cd "$(dirname "$0")"

cmake -B build -G Ninja \
    -DCMAKE_C_COMPILER="${MSVC_ROOT}/bin/Hostx64/x64/cl.exe" \
    -DCMAKE_CXX_COMPILER="${MSVC_ROOT}/bin/Hostx64/x64/cl.exe" \
    -DCMAKE_RC_COMPILER="${RC_PATH}/rc.exe" \
    "$@" 2>&1

ninja -C build "${NINJA_TARGET:-rnet_util}" 2>&1
