#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>

#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>

#include "Loops.h"
#include "ReduceOpsUtils.h"
#include "ScatterGather.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void Gather(
    Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  dpcpp_scatter_gather_base_kernel</*is_scatter_like=*/false>()(
      result, dim, index, self, "gather_out_dpcpp", tensor_assign);
}

template <typename scalar_t>
void Scatter(
    Tensor& result,
    int64_t dim,
    const Tensor& index,
    const Tensor& self) {
  dpcpp_scatter_gather_base_kernel</*is_scatter_like=*/true>()(
      result, dim, index, self, "scatter_out_dpcpp", tensor_assign);
}

template <typename scalar_t>
void ScatterFill(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    Scalar value_scalar) {
  dpcpp_scatter_fill_base_kernel<>()(
      self, dim, index, value_scalar, "scatter_fill_dpcpp_", tensor_assign);
}

template <typename scalar_t>
typename std::enable_if<
    IS_FLOAT32(scalar_t) || IS_BFLOAT16(scalar_t) || IS_INT(scalar_t) ||
        IS_INT8(scalar_t) || IS_INT16(scalar_t) || IS_UINT8(scalar_t) ||
        IS_INT64(scalar_t) || IS_DOUBLE(scalar_t) || IS_COMPLEX(scalar_t) ||
        IS_BOOL(scalar_t) || IS_HALF(scalar_t),
    void>::type
ScatterAdd(
    Tensor& result,
    int64_t dim,
    const Tensor& index,
    const Tensor& self) {
  globalContext().alertNotDeterministic("scatter_add_dpcpp_kernel");
  dpcpp_scatter_gather_base_kernel<
      /*is_scatter_like=*/true,
      /*cast_to_opaque=*/false>()(
      result, dim, index, self, "scatter_add_dpcpp", reduce_add);
}

template <typename scalar_t>
typename std::enable_if<
    !(IS_FLOAT32(scalar_t) || IS_BFLOAT16(scalar_t) || IS_INT(scalar_t) ||
      IS_INT8(scalar_t) || IS_INT16(scalar_t) || IS_UINT8(scalar_t) ||
      IS_INT64(scalar_t) || IS_DOUBLE(scalar_t) || IS_COMPLEX(scalar_t) ||
      IS_BOOL(scalar_t) || IS_HALF(scalar_t)),
    void>::type
ScatterAdd(
    Tensor& tensor,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  TORCH_CHECK(
      0,
      "scatter add only supports float, bfloat16, int, int64 and double type");
}
} // namespace impl

void scatter_dpcpp_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  dpcpp_scatter_gather_base_kernel<>()(
      self, dim, index, src, "scatter_dpcpp_", tensor_assign);
}

void scatter_reduce_dpcpp_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_dpcpp_add_", reduce_add);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self,
          dim,
          index,
          src,
          "scatter_reduce_dpcpp_multiply_",
          reduce_multiply);
      break;
    default:
      break;
  }
}

void scatter_fill_dpcpp_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& src) {
  dpcpp_scatter_fill_base_kernel<>()(
      self, dim, index, src, "scatter_fill_dpcpp_", tensor_assign);
}

void scatter_scalar_reduce_dpcpp_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Scalar& value,
    const SCATTER_GATHER_OP& reduce) {
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      dpcpp_scatter_fill_base_kernel<false>()(
          self, dim, index, value, "scatter_fill_dpcpp_add_", reduce_add);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      dpcpp_scatter_fill_base_kernel<false>()(
          self,
          dim,
          index,
          value,
          "scatter_fill_dpcpp_multiply_",
          reduce_multiply);
      break;
    default:
      break;
  }
}

// gather family
Tensor& gather_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    bool sparse_grad,
    Tensor& out) {
  if (index.numel() == 0)
    return out;
  dim = at::maybe_wrap_dim(dim, self.dim());
  out.resize_(index.sizes());

  bool check_result = out.defined();
  if (check_result) {
    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, self);
    at::assert_no_partial_overlap(out, index);
  }
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "Gather",
      [&]() { impl::Gather<scalar_t>(out, self, dim, index); });
  return out;
}

// scatter family
// scatter.value_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    Tensor& out) {
  at::assert_no_internal_overlap(self);
  if (!self.is_same(out)) {
    out.copy_(self);
  }
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "ScatterFill",
      [&]() { impl::ScatterFill<scalar_t>(out, dim, index, value); });
  return out;
}

Tensor& scatter_add_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  if (index.numel() == 0)
    return out;
  dim = maybe_wrap_dim(dim, self.dim());
  at::assert_no_internal_overlap(self);
  if (!self.is_same(out)) {
    out.copy_(self);
  }
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, src);
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "ScatterAdd",
      [&]() { impl::ScatterAdd<scalar_t>(out, dim, index, src); });
  return out;
}

// scatter.reduce_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    c10::string_view reduce,
    Tensor& out) {
  scatter_impl(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_dpcpp_kernel,
      scatter_dpcpp_kernel,
      reduce);
  return out;
}

// scatter.src_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    Tensor& out) {
  scatter_impl(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_dpcpp_kernel,
      scatter_dpcpp_kernel);
  return out;
}

// scatter.value_reduce_out
Tensor& scatter_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Scalar& value,
    c10::string_view reduce,
    Tensor& out) {
  scatter_impl(
      self,
      dim,
      index,
      value,
      out,
      scatter_scalar_reduce_dpcpp_kernel,
      scatter_fill_dpcpp_kernel,
      reduce);
  return out;
}

void scatter_reduce_two_dpcpp_kernel(
    const Tensor& self,
    const int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const SCATTER_GATHER_OP& reduce) {
  globalContext().alertNotDeterministic("scatter_reduce_dpcpp");
  switch (reduce) {
    case SCATTER_GATHER_OP::REDUCE_ADD:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_cuda_sum_", reduce_add);
      break;
    case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_cuda_prod_", reduce_multiply);
      break;
    case SCATTER_GATHER_OP::REDUCE_MAXIMUM:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_cuda_amax_", reduce_maximum);
      break;
    case SCATTER_GATHER_OP::REDUCE_MINIMUM:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_cuda_amin_", reduce_minimum);
      break;
    case SCATTER_GATHER_OP::REDUCE_MEAN:
      dpcpp_scatter_gather_base_kernel<true, false>()(
          self, dim, index, src, "scatter_reduce_cuda_mean_", reduce_mean);
      break;
  }
}

Tensor& scatter_reduce_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src,
    const c10::string_view reduce,
    bool include_self,
    Tensor& out) {
  // See issue https://github.com/pytorch/pytorch/issues/74770
  TORCH_WARN_ONCE(
      "scatter_reduce() is in beta and the API may change at any time.");

  scatter_impl</*use_new_options=*/true>(
      self,
      dim,
      index,
      src,
      out,
      scatter_reduce_two_dpcpp_kernel,
      scatter_dpcpp_kernel,
      reduce,
      include_self);

  if (get_operator_enum(reduce, true) == SCATTER_GATHER_OP::REDUCE_MEAN) {
    auto ones = at::ones_like(src);
    auto count = include_self ? at::ones_like(out) : at::zeros_like(out);
    count.scatter_add_(dim, index, ones);
    count.masked_fill_(count == 0, 1);

    if (out.is_floating_point() || out.is_complex()) {
      out.div_(count);
    } else {
      out.div_(count, "floor");
    }
  }
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
