#pragma once

#include "backend.h"
#include <memory>
#include <string>
#include <vector>

namespace rnet::gpu {

/// Information about a single GPU device.
struct GpuDeviceInfo {
    std::string name;
    size_t total_memory = 0;
    size_t free_memory = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    GpuBackendType backend_type = GpuBackendType::CPU_FALLBACK;
};

/// GPU context — device enumeration and backend creation.
class GpuContext {
public:
    /// Enumerate all available GPU devices across all compiled backends.
    static std::vector<GpuDeviceInfo> enumerate_devices();

    /// Create a backend for the device at the given index (from enumerate_devices).
    /// Falls back to CPU if device_id is out of range.
    static std::unique_ptr<GpuBackend> create_for_device(int device_id);
};

}  // namespace rnet::gpu
