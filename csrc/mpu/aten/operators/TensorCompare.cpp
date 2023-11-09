#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include "comm/AccumulateType.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/quantized/QTensorImpl.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include "PSTLFunctions.h"
#include "comm/ApplyUtils.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"
#include "comm/ScalarOps.h"

#include "comm/ATDispatch.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& where_out(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out);
namespace impl {

void where_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kHalf, kBFloat16, kBool, iter.dtype(), "where_dpcpp", [&] {
        dpcpp_kernel_for_tensor_iter(
            iter,
            [=](bool cond_val, scalar_t self_val, scalar_t other_val)
                -> scalar_t { return cond_val ? self_val : other_val; });
      });
}

// Composite op implementation for simplicity. This materializes the cross
// product of elements and test elements, so it is not very memory efficient
void isin_default_kernel(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    Tensor& out) {
  out.resize_as_(elements);
  std::vector<int64_t> bc_shape(elements.dim(), 1);
  bc_shape.push_back(-1);
  out.copy_(
      invert ? elements.unsqueeze(-1).ne(test_elements.view(bc_shape)).all(-1)
             : elements.unsqueeze(-1).eq(test_elements.view(bc_shape)).any(-1));
}

} // namespace impl

Tensor isnan(const Tensor& self) {
  return self != self;
}

template <typename scalar_t>
void _assert_async_kernel(scalar_t* input) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    cgf.single_task([=]() { SYCL_KERNEL_ASSERT(input[0] != 0); });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <>
void _assert_async_kernel(c10::complex<float>* input) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    cgf.single_task(
        [=]() { SYCL_KERNEL_ASSERT(input[0] != c10::complex<float>(0, 0)); });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <>
void _assert_async_kernel(c10::complex<double>* input) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgf) {
    cgf.single_task(
        [=]() { SYCL_KERNEL_ASSERT(input[0] != c10::complex<double>(0, 0)); });
  };

  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

void _assert_async(const Tensor& self) {
  auto n = self.numel();
  TORCH_CHECK(n != 0, "Boolean value of Tensor with no values is ambiguous");
  TORCH_CHECK(
      n < 2, "Boolean value of Tensor with more than one value is ambiguous");
  IPEX_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "_assert_async",
      [&]() { _assert_async_kernel<scalar_t>(self.data_ptr<scalar_t>()); });
}

Tensor& isneginf_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(!self.is_complex(), "isneginf does not support complex inputs.");
  TORCH_CHECK(
      out.dtype() == at::kBool,
      "isneginf does not support non-boolean outputs.");
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    out.fill_(false);
  } else {
    TensorIterator iter;
    iter.build_borrowing_unary_force_boolean_op(out, self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.input_dtype(),
        "isneginf",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> bool {
            return a == Numerics<scalar_t>::lower_bound();
          });
        });
  }
  return out;
}

Tensor& isposinf_out(const Tensor& self, Tensor& out) {
  TORCH_CHECK(!self.is_complex(), "isposinf does not support complex inputs.");
  TORCH_CHECK(
      out.dtype() == at::kBool,
      "isposinf does not support non-boolean outputs.");
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    out.fill_(false);
  } else {
    TensorIterator iter;
    iter.build_borrowing_unary_force_boolean_op(out, self);
    IPEX_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.input_dtype(),
        "isposinf",
        [&]() {
          dpcpp_kernel_for_tensor_iter(iter, [](scalar_t a) -> bool {
            return a == Numerics<scalar_t>::upper_bound();
          });
        });
  }
  return out;
}

static inline void check_for_unsupported_isin_dtype(const ScalarType type) {
  // Bail out for dtypes unsupported by the sorting algorithm to keep the
  // interface consistent.
  TORCH_CHECK(
      type != ScalarType::Bool && type != ScalarType::BFloat16 &&
          type != ScalarType::ComplexFloat && type != ScalarType::ComplexDouble,
      "Unsupported input type encountered for isin(): ",
      type);
}

// Sorting-based algorithm for isin(); used when the number of test elements is
// large.
static void isin_sorting(
    const Tensor& elements,
    const Tensor& test_elements,
    bool assume_unique,
    bool invert,
    Tensor& out) {
  // 1. Concatenate unique elements with unique test elements in 1D form. If
  //    assume_unique is true, skip calls to unique().
  Tensor elements_flat, test_elements_flat, unique_order;
  if (assume_unique) {
    elements_flat = elements.ravel();
    test_elements_flat = test_elements.ravel();
  } else {
    std::tie(elements_flat, unique_order) = at::_unique(elements, false, true);
    std::tie(test_elements_flat, std::ignore) =
        at::_unique(test_elements, false);
  }

  // 2. Stable sort all elements, maintaining order indices to reverse the
  //    operation. Stable sort is necessary to keep elements before test
  //    elements within the sorted list.
  Tensor all_elements = at::cat({elements_flat, test_elements_flat});
  // use pstl sort here, equals to "all_elements.sort(true, 0, false)"
  auto index_options = all_elements.options().dtype(kLong);
  Tensor sorted_elements = all_elements.clone().reshape(-1);
  int64_t num_inp = sorted_elements.numel();
  Tensor sorted_order = at::empty({num_inp}, index_options);
  auto sorted_indices_begin = sorted_order.data_ptr<int64_t>();
  xpu::pstl::iota(
      sorted_indices_begin, sorted_indices_begin + num_inp, (int64_t)0);
  auto scalar_t = all_elements.scalar_type();

  if (num_inp > 0) {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        all_elements.scalar_type(),
        "isin_sort",
        [&]() {
          xpu::pstl::sort<scalar_t, int64_t>(
              all_elements.data_ptr<scalar_t>(),
              sorted_elements.data_ptr<scalar_t>(),
              sorted_order.data_ptr<int64_t>(),
              num_inp,
              false);
        });
  }

  // 3. Create a mask for locations of adjacent duplicate values within the
  //    sorted list. Duplicate values are in both elements and test elements.
  Tensor duplicate_mask =
      at::empty_like(sorted_elements, TensorOptions(ScalarType::Bool));
  Tensor sorted_except_first = sorted_elements.slice(0, 1, at::indexing::None);
  Tensor sorted_except_last = sorted_elements.slice(0, 0, -1);
  duplicate_mask.slice(0, 0, -1).copy_(
      invert ? sorted_except_first.ne(sorted_except_last)
             : sorted_except_first.eq(sorted_except_last));
  duplicate_mask.index_put_({-1}, invert);

  // 4. Reorder the mask to match the pre-sorted element order.
  Tensor mask = at::empty_like(duplicate_mask);
  mask.index_copy_(0, sorted_order, duplicate_mask);

  // 5. Index the mask to match the pre-unique element order. If
  //    assume_unique is true, just take the first N items of the mask,
  //    where N is the original number of elements.
  if (assume_unique) {
    out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
  } else {
    out.copy_(at::index(mask, {c10::optional<Tensor>(unique_order)}));
  }
}

Tensor& isin_out(
    const Tensor& elements,
    const Tensor& test_elements,
    bool assume_unique,
    bool invert,
    Tensor& out) {
  check_for_unsupported_isin_dtype(elements.scalar_type());
  check_for_unsupported_isin_dtype(test_elements.scalar_type());
  if (elements.numel() == 0) {
    return out;
  }

  // Heuristic taken from numpy's implementation.
  // See
  // https://github.com/numpy/numpy/blob/fb215c76967739268de71aa4bda55dd1b062bc2e/numpy/lib/arraysetops.py#L575
  if (test_elements.numel() <
      static_cast<int64_t>(
          10.0f *
          Numerics<double>::pow(
              static_cast<double>(elements.numel()), 0.145))) {
    out.fill_(invert);
    impl::isin_default_kernel(elements, test_elements, invert, out);
  } else {
    isin_sorting(elements, test_elements, assume_unique, invert, out);
  }
  return out;
}

Tensor& isin_out(
    const Tensor& elements,
    const Scalar& test_elements,
    bool assume_unique,
    bool invert,
    Tensor& out) {
  check_for_unsupported_isin_dtype(elements.scalar_type());
  check_for_unsupported_isin_dtype(test_elements.type());

  // redispatch to eq / ne
  if (invert) {
    at::AtenIpexTypeXPU::ne_out(elements, test_elements, out);
  } else {
    at::AtenIpexTypeXPU::eq_out(elements, test_elements, out);
  }
  return out;
}

Tensor& isin_out(
    const Scalar& elements,
    const Tensor& test_elements,
    bool assume_unique,
    bool invert,
    Tensor& out) {
  check_for_unsupported_isin_dtype(elements.type());
  check_for_unsupported_isin_dtype(test_elements.scalar_type());
  // redispatch
  at::AtenIpexTypeXPU::isin_out(
      wrapped_scalar_tensor(elements, test_elements.device()),
      test_elements,
      assume_unique,
      invert,
      out);
  return out;
}

template <typename... Args>
Device out_device(Args&... inps) {
  for (const auto& i : {inps...}) {
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  return at::kCPU;
}

Tensor& where_out(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  Tensor self_, other_, condition_;
  if (self.dtype() != other.dtype()) {
    auto result_type = at::native::result_type(self, other);
    self_ = self.to(result_type);
    other_ = other.to(result_type);
  } else {
    self_ = self;
    other_ = other;
  }
  auto device = out_device(condition, self_, other_);
  condition_ = condition;
  if (device != at::kCPU) { // allow CPU scalars on non-cpu device
    if (condition.device() != device && condition.ndimension() == 0) {
      condition_ = condition.to(device);
    }
    if (self_.device() != device && self_.ndimension() == 0) {
      self_ = self_.to(device);
    }
    if (other_.device() != device && other_.ndimension() == 0) {
      other_ = other_.to(device);
    }
  }
  if (condition.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
  } else {
    TORCH_CHECK(
        condition.scalar_type() == ScalarType::Bool,
        "where expected condition to be a boolean tensor, but got a tensor with dtype ",
        condition.scalar_type());
  }
  condition_ = condition_.scalar_type() == ScalarType::Byte
      ? condition_.to(ScalarType::Bool)
      : condition_;
  // if there's still a device mismatch, let tensoriterator error out with it
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(out)
                  .add_input(condition_)
                  .add_input(self_)
                  .add_input(other_)
                  .build();
  impl::where_kernel(iter);
  return out;
}

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  auto device = out_device(condition, self, other);
  auto result_type = at::native::result_type(self, other);
  Tensor ret = at::empty({0}, self.options().dtype(result_type).device(device));
  at::AtenIpexTypeXPU::where_out(condition, self, other, ret);
  return ret;
}

} // namespace AtenIpexTypeXPU
} // namespace at
