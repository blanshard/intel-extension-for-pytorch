#include <core/Device.h>
#include <core/Generator.h>

namespace xpu {
namespace dpcpp {
namespace detail {

static std::once_flag num_gpu_init_flag;

static int64_t num_gpus;

static std::deque<std::once_flag> dpcpp_gens_init_flag;

static std::vector<Generator> default_gens_dpcpp;

static void initDPCPPGenVector() {
  num_gpus = device_count();
  dpcpp_gens_init_flag.resize(num_gpus);
  default_gens_dpcpp.resize(num_gpus);
}

const Generator& getDefaultDPCPPGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initDPCPPGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  std::call_once(dpcpp_gens_init_flag[idx], [&] {
    default_gens_dpcpp[idx] = make_generator<DPCPPGeneratorImpl>(idx);
    default_gens_dpcpp[idx].seed();
  });
  return default_gens_dpcpp[idx];
}

Generator createDPCPPGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initDPCPPGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = make_generator<DPCPPGeneratorImpl>(idx);
  auto dpcpp_gen = check_generator<DPCPPGeneratorImpl>(gen);
  dpcpp_gen->set_current_seed(default_rng_seed_val);
  dpcpp_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail

DPCPPGeneratorImpl::DPCPPGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl{
          Device(DeviceType::XPU, device_index),
          DispatchKeySet(c10::DispatchKey::XPU)} {}

void DPCPPGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

uint64_t DPCPPGeneratorImpl::current_seed() const {
  return seed_;
}

uint64_t DPCPPGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

void DPCPPGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  philox_offset_per_thread_ = offset;
}

uint64_t DPCPPGeneratorImpl::philox_offset_per_thread() const {
  return philox_offset_per_thread_;
}

std::pair<uint64_t, uint64_t> DPCPPGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  increment = ((increment + 3) / 4) * 4;
  TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

DeviceType DPCPPGeneratorImpl::device_type() {
  return DeviceType::XPU;
}

std::shared_ptr<DPCPPGeneratorImpl> DPCPPGeneratorImpl::clone() const {
  return std::shared_ptr<DPCPPGeneratorImpl>(this->clone_impl());
}

DPCPPGeneratorImpl* DPCPPGeneratorImpl::clone_impl() const {
  auto gen = new DPCPPGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

/**
 * DPCPPGeneratorImplState is a POD class containing
 * the state in the DPCPPGeneratorImpl.
 */
struct DPCPPGeneratorImplState {
  uint64_t seed;
  uint64_t philox_offset_per_thread;
};

/**
 * Sets the internal state of DPCPPGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of DPCPPGeneratorImplState for the layout.
 */
void DPCPPGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static_assert(
      std::is_pod<DPCPPGeneratorImplState>::value,
      "DPCPPGeneratorImplState is not a PODType");
  static const size_t size_current = sizeof(DPCPPGeneratorImplState);

  at::detail::check_rng_state(new_state);

  auto new_state_size = new_state.numel();

  if (new_state_size == size_current) {
    auto rng_state = (DPCPPGeneratorImplState*)new_state.data();
    this->set_current_seed(rng_state->seed);
    this->set_philox_offset_per_thread(rng_state->philox_offset_per_thread);
  } else {
    AT_ERROR(
        "Expected a DPCPPGeneratorImplState of size ",
        size_current,
        " but found the input RNG state size to be ",
        new_state_size);
  }
}

c10::intrusive_ptr<c10::TensorImpl> DPCPPGeneratorImpl::get_state() const {
  static const size_t size = sizeof(DPCPPGeneratorImplState);
  static_assert(
      std::is_pod<DPCPPGeneratorImplState>::value,
      "DPCPPGeneratorImplState is not a PODType");

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)size},
      ScalarType::Byte,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
  auto rng_state = state_tensor.data_ptr();

  // accumulate generator data to be copied into byte tensor
  auto accum_state = std::make_unique<DPCPPGeneratorImplState>();
  accum_state->seed = this->seed_;
  accum_state->philox_offset_per_thread = this->philox_offset_per_thread_;

  memcpy(rng_state, accum_state.get(), size);
  return state_tensor.getIntrusivePtr();
}

} // namespace dpcpp
} // namespace xpu
