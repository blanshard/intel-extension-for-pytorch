#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>

#include <core/Generator.h>
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/RegistrationDeclarations.h"

#include "DistributionTemplates.h"
#include "RandomEngine.h"

namespace at {
namespace AtenIpexTypeXPU {

void uniform_kernel(
    TensorIterator& iter,
    double from_,
    double to_,
    c10::optional<Generator> gen_) {
  auto gen = get_generator_or_default<xpu::dpcpp::DPCPPGeneratorImpl>(
      gen_, xpu::dpcpp::detail::getDefaultDPCPPGenerator());
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "uniform_dpcpp",
      [&] {
        auto from = static_cast<scalar_t>(from_);
        auto to = static_cast<scalar_t>(to_);
        using accscalar_t = acc_type<scalar_t>;
        auto range = static_cast<accscalar_t>(to - from);
        auto uniform_func = [range, from](accscalar_t rand) {
          auto reverse_bound_rand = rand == static_cast<accscalar_t>(1.0)
              ? static_cast<accscalar_t>(0.0)
              : rand;
          return static_cast<scalar_t>(reverse_bound_rand * range + from);
        };
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen, uniform_func);
      });
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name, " is out of bounds for ", dtype);

Tensor& uniform_(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> generator) {
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    uniform_(float_tensor, from, to, generator);
  } else {
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "check_uniform_bounds",
        [&] {
          const auto dtype = self.dtype();
          const auto min =
              static_cast<double>(std::numeric_limits<scalar_t>::lowest());
          const auto max =
              static_cast<double>(std::numeric_limits<scalar_t>::max());
          CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
          CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
          TORCH_CHECK(
              from <= to,
              "uniform_ expects to return a [from, to) range, but found from=",
              from,
              " > to=",
              to);
          TORCH_CHECK(
              (to - from) <= std::numeric_limits<scalar_t>::max(),
              "uniform_ expects to-from <= std::numeric_limits<",
              toString(self.scalar_type()),
              ">::max(), but found to=",
              to,
              " and from=",
              from,
              " which result in to-from to exceed the limit");
          from = std::min(std::max(from, min), max);
          to = std::max(std::min(to, max), min);
        });
    auto iter = at::TensorIterator::nullary_op(self);
    uniform_kernel(iter, from, to, generator);
  }
  return self;
}

} // namespace AtenIpexTypeXPU
} // namespace at
