#include <c10/util/Exception.h>
#include <runtime/CachingHostAllocator.h>
#include <runtime/Exception.h>
#include <utils/Macros.h>

#include <deque>
#include <mutex>
#include <set>
#include <unordered_map>

namespace xpu {
namespace dpcpp {

constexpr size_t kHostAlignment = 512;

CachingHostAllocator* CachingHostAllocator::Instance() {
  static CachingHostAllocator myInstance;
  return &myInstance;
}

void* CachingHostAllocator::Block::getPtr() const {
  return mPtr;
}

sycl::context& CachingHostAllocator::Block::getContext() const {
  return dpcppGetDeviceContext(mDevId);
}

bool CachingHostAllocator::BlockState::hasEvent() {
  return !mEvents.empty();
}

void CachingHostAllocator::BlockState::insertEvent(sycl::event& e) {
  mEvents.emplace_back(e);
}

void CachingHostAllocator::BlockState::processEvents() {
  while (hasEvent()) {
    auto& e = mEvents.front();
    bool completed =
        e.get_info<dpcpp_event_exec_stat>() == dpcpp_event_cmd_stat_complete;
    if (!completed) {
      return;
    }
    mEvents.pop_front();
  }
}

bool CachingHostAllocator::BlockState::isAllocated() {
  return mAllocated;
}

void CachingHostAllocator::BlockState::setAllocated(bool alloc) {
  mAllocated = alloc;
}

CachingHostAllocator::CachingHostAllocator() : mAvailable(Block::Comparator) {}

CachingHostAllocator::~CachingHostAllocator() {
  emptyCache();
}

void CachingHostAllocator::processEvents() {
  for (auto& mb : mBlocks) {
    auto& block_state = mb.second;
    block_state.processEvents();
    if (!block_state.isAllocated() && !block_state.hasEvent()) {
      mAvailable.insert(block_state);
    }
  }
}

bool CachingHostAllocator::isHostPtr(const void* ptr) {
#if defined(USE_MULTI_CONTEXT)
  int count;
  AT_DPCPP_CHECK(dpcppGetDeviceCount(&count));
  // We can NOT guarantee the ptr is allocated with the current device context.
  for (auto i = 0; i < count; i++) {
    if (sycl::usm::alloc::host ==
        sycl::get_pointer_type(ptr, dpcppGetDeviceContext(i))) {
      return true;
    }
  }
  return false;
#else
  return sycl::usm::alloc::host ==
      sycl::get_pointer_type(ptr, dpcppGetDeviceContext());
#endif
}

void CachingHostAllocator::emptyCache() {
  std::lock_guard<std::mutex> lock(mMutex);
  processEvents();

  for (auto& blk : mAvailable) {
    auto it = mBlocks.find(blk.getPtr());
    AT_ASSERT(it != mBlocks.end() && !it->second.isAllocated());
    sycl::free(blk.getPtr(), blk.getContext());
    mBlocks.erase(it);
  }

  mAvailable.clear();
}

void CachingHostAllocator::recordEvent(void* ptr, sycl::event& e) {
  std::lock_guard<std::mutex> lock(mMutex);

  auto it = mBlocks.find(ptr);
  if (it == mBlocks.end()) {
    return;
  }

  auto& block = it->second;
  block.insertEvent(e);
}

int CachingHostAllocator::malloc(void** ptr, size_t size) {
  std::lock_guard<std::mutex> lock(mMutex);
  processEvents();

  *ptr = nullptr;
  if (size <= 0) {
    return DPCPP_SUCCESS;
  }

  DeviceId curDevID;
  AT_DPCPP_CHECK(dpcppGetDevice(&curDevID));

  Block block_search(curDevID, size);
  auto it = mAvailable.lower_bound(block_search);
  if (it != mAvailable.end() && it->getContext() == dpcppGetDeviceContext()) {
    auto& block = mBlocks.at(it->getPtr());
    AT_ASSERT(!block.isAllocated() && !block.hasEvent());
    block.setAllocated(true);
    *ptr = it->getPtr();
    mAvailable.erase(it);
    return DPCPP_SUCCESS;
  }

  *ptr =
      sycl::aligned_alloc_host(kHostAlignment, size, dpcppGetDeviceContext());
  if (*ptr == NULL) {
    *ptr =
        sycl::aligned_alloc_host(kHostAlignment, size, dpcppGetDeviceContext());
    if (*ptr == NULL) {
      return DPCPP_FAILURE;
    }
  }

  mBlocks.insert({*ptr, {curDevID, size, *ptr, true}});
  return DPCPP_SUCCESS;
}

void CachingHostAllocator::release(void* ptr) {
  std::lock_guard<std::mutex> lock(mMutex);

  if (ptr == nullptr) {
    return;
  }

  auto it = mBlocks.find(ptr);
  AT_ASSERT(it != mBlocks.end());

  auto& block = it->second;
  AT_ASSERT(block.isAllocated());

  block.setAllocated(false);
  processEvents();
}

} // namespace dpcpp
} // namespace xpu
