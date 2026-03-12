#ifdef RNET_HAS_METAL

#include "metal_context.h"
#include "../../core/logging.h"

namespace rnet::gpu {

std::vector<GpuDeviceInfo> MetalContext::enumerate() {
    // Stub: real implementation would use Metal API to enumerate devices.
    LogPrintf("Metal: device enumeration stub — returning empty list");
    return {};
}

}  // namespace rnet::gpu

#endif  // RNET_HAS_METAL
