// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

#ifdef RNET_HAS_METAL

// Platform headers (Objective-C imports must precede own header).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Own header.
#include "metal_context.h"

// Project headers.
#include "../../core/logging.h"

namespace rnet::gpu {

// ---------------------------------------------------------------------------
// MetalContext::enumerate
// ---------------------------------------------------------------------------
// Discovers all Metal-capable GPUs on the system.  On macOS,
// MTLCopyAllDevices() returns discrete + integrated GPUs.  On iOS, only the
// system default device is available.  The Metal GPU family is mapped to a
// (major, minor) capability pair as a rough analogue of CUDA compute
// capability.
// ---------------------------------------------------------------------------
std::vector<GpuDeviceInfo> MetalContext::enumerate()
{
    std::vector<GpuDeviceInfo> devices;

    @autoreleasepool {
#if TARGET_OS_OSX
        NSArray<id<MTLDevice>>* mtl_devices = MTLCopyAllDevices();
#else
        NSMutableArray<id<MTLDevice>>* mtl_devices = [NSMutableArray array];
        id<MTLDevice> def = MTLCreateSystemDefaultDevice();
        if (def) [mtl_devices addObject:def];
#endif

        for (id<MTLDevice> dev in mtl_devices) {
            GpuDeviceInfo info;
            info.name = std::string([[dev name] UTF8String]);
            info.total_memory = [dev recommendedMaxWorkingSetSize];
            info.free_memory = info.total_memory; // No direct query available
            info.backend_type = GpuBackendType::METAL;

            // 1. Map Metal GPU family to capability pair.
            if ([dev supportsFamily:MTLGPUFamilyApple9]) {
                info.compute_capability_major = 9;
                info.compute_capability_minor = 0;
            } else if ([dev supportsFamily:MTLGPUFamilyApple8]) {
                info.compute_capability_major = 8;
                info.compute_capability_minor = 0;
            } else if ([dev supportsFamily:MTLGPUFamilyApple7]) {
                info.compute_capability_major = 7;
                info.compute_capability_minor = 0;
            } else if ([dev supportsFamily:MTLGPUFamilyApple6]) {
                info.compute_capability_major = 6;
                info.compute_capability_minor = 0;
            } else if ([dev supportsFamily:MTLGPUFamilyApple5]) {
                info.compute_capability_major = 5;
                info.compute_capability_minor = 0;
            } else {
                info.compute_capability_major = 1;
                info.compute_capability_minor = 0;
            }

            LogPrintf("Metal: found device '%s' (%.1f GB, family %d)",
                      info.name.c_str(),
                      static_cast<double>(info.total_memory) / (1024.0 * 1024.0 * 1024.0),
                      info.compute_capability_major);

            devices.push_back(std::move(info));
        }

#if TARGET_OS_OSX
        // MTLCopyAllDevices returns a retained NSArray -- release it.
        [mtl_devices release];
#endif
    }

    if (devices.empty()) {
        LogPrintf("Metal: no devices found");
    }

    return devices;
}

} // namespace rnet::gpu

#endif // RNET_HAS_METAL
