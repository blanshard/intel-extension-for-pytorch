#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <mutex>

#include <core/AllocationInfo.h>
#include <core/Stream.h>
#include <utils/Macros.h>

namespace xpu {
namespace dpcpp {

/// Device Allocator
IPEX_API void emptyCacheInDevAlloc();

IPEX_API DeviceStats getDeviceStatsFromDevAlloc(DeviceIndex device_index);

IPEX_API void resetAccumulatedStatsInDevAlloc(DeviceIndex device_index);

IPEX_API void resetPeakStatsInDevAlloc(DeviceIndex device_index);

IPEX_API std::vector<SegmentInfo> snapshotOfDevAlloc();

at::Allocator* getDeviceAllocator();

void cacheInfoFromDevAlloc(
    DeviceIndex deviceIndex,
    size_t* cachedAndFree,
    size_t* largestBlock);

void* getBaseAllocationFromDevAlloc(void* ptr, size_t* size);

void recordStreamInDevAlloc(const DataPtr& ptr, DPCPPStream stream);

IPEX_API void dumpMemoryStatusFromDevAlloc(DeviceIndex device_index);

std::mutex* getFreeMutexOfDevAlloc();

/// Host Allocator
// Provide a caching allocator for host allocation by USM malloc_host
Allocator* getHostAllocator();

// Releases all cached host memory allocations
void emptyCacheInHostAlloc();

bool isAllocatedByHostAlloc(const void* ptr);

} // namespace dpcpp
} // namespace xpu
