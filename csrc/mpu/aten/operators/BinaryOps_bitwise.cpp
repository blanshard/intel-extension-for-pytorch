#include <ATen/Context.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <utils/DPCPP.h>
#include "comm/Pointwise.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
struct BitwiseAndFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a & b;
  }
};

template <>
struct BitwiseAndFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a && b;
  }
};

void and_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_INTEGRAL_TYPES_AND(
      kBool, iter.dtype(), "bitwise_and_xpu", [&]() {
        BitwiseAndFunctor<scalar_t> f;
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
      });
}

template <typename scalar_t>
struct BitwiseOrFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a | b;
  }
};

template <>
struct BitwiseOrFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a || b;
  }
};

void or_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_INTEGRAL_TYPES_AND(
      kBool, iter.dtype(), "bitwise_or_xpu", [&]() {
        BitwiseOrFunctor<scalar_t> f;
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
      });
}

template <typename scalar_t>
struct BitwiseXorFunctor {
  inline scalar_t operator()(scalar_t a, scalar_t b) const {
    return a ^ b;
  }
};

template <>
struct BitwiseXorFunctor<bool> {
  inline bool operator()(bool a, bool b) const {
    return a != b;
  }
};

void xor_kernel_dpcpp(TensorIterator& iter) {
  IPEX_DISPATCH_INTEGRAL_TYPES_AND(
      kBool, iter.dtype(), "bitwise_xor_xpu", [&]() {
        BitwiseXorFunctor<scalar_t> f;
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
      });
}

} // namespace impl

Tensor& bitwise_and_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::and_kernel_dpcpp(iter);
  return out;
}

Tensor& bitwise_or_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::or_kernel_dpcpp(iter);
  return out;
}

Tensor& bitwise_xor_out(const Tensor& self, const Tensor& other, Tensor& out) {
  auto iter = TensorIterator::binary_op(out, self, other);
  impl::xor_kernel_dpcpp(iter);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
