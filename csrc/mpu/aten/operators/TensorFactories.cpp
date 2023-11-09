#include <ATen/ATen.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorFactories.h>
#include <c10/util/Exception.h>
#include <core/Allocator.h>
#include <core/Device.h>
#include <core/Generator.h>
#include <core/detail/ListUtils.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include "BitonicMergeSort.h"
#include "Loops.h"
#include "PSTLFunctions.h"
#include "ReduceOpStdVar.h"
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace at::native;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

Tensor empty_dpcpp(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  TORCH_INTERNAL_ASSERT(
      options.backend() == at::Backend::XPU ||
      options.backend() == at::Backend::QuantizedXPU);
  // TORCH_INTERNAL_ASSERT(!options.is_variable()); // is_variable should have
  // been
  // "unpacked"

  auto* allocator = xpu::dpcpp::getDeviceAllocator();
  int64_t nelements = xpu::dpcpp::detail::prod_intlist(size);
  auto dtype = options.dtype();
  int64_t size_bytes = nelements * dtype.itemsize();
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizeable=*/true);
  auto tensor = detail::make_tensor<TensorImpl>(
      storage_impl, options.computeDispatchKey(), dtype);
  // Default TensorImpl has size [0]
  if (size.size() != 1 || size[0] != 0) {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  }

  TORCH_CHECK(
      !(options.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  auto memory_format = options.memory_format_opt().value_or(
      optional_memory_format.value_or(MemoryFormat::Contiguous));

  tensor.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  return tensor;
}

Tensor empty_quantized(
    IntArrayRef size,
    const Tensor& qtensor,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  TensorOptions specified_options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  TORCH_CHECK(
      !(specified_options.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  TensorOptions options = qtensor.options()
                              .merge_in(specified_options)
                              .merge_memory_format(memory_format);

  Tensor output;
  if (qtensor.qscheme() == kPerTensorAffine) {
    output = at::_empty_affine_quantized(
        size, options, qtensor.q_scale(), qtensor.q_zero_point());
  } else if (
      qtensor.qscheme() == kPerChannelAffine ||
      qtensor.qscheme() == kPerChannelAffineFloatQParams) {
    output = at::_empty_per_channel_affine_quantized(
        size,
        qtensor.q_per_channel_scales(),
        qtensor.q_per_channel_zero_points(),
        qtensor.q_per_channel_axis(),
        options);
  } else {
    TORCH_CHECK(
        false,
        "QScheme not supported by empty_quantized:",
        toString(qtensor.qscheme()));
  }
  return output;
}

Tensor empty_strided_dpcpp(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  check_size_nonnegative(size);
  auto t = empty_dpcpp({0}, options, c10::nullopt);
  resize_impl(t.unsafeGetTensorImpl(), size, stride);
  return t;
}

Tensor& eye_out_dpcpp(Tensor& result, int64_t n, int64_t m) {
  TORCH_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
  TORCH_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

  result.resize_({n, m});
  result.zero_();

  int64_t sz = std::min<int64_t>(n, m);
  int64_t stride = result.stride(0) + result.stride(1);

  Tensor diag = result.as_strided({sz}, {stride});
  diag.fill_(1);
  return result;
}

Tensor& eye_out_dpcpp(Tensor& result, int64_t n) {
  return eye_out_dpcpp(result, n, n);
}

template <typename scalar_t>
Tensor randperm_dpcpp(
    Tensor& result,
    int64_t n,
    c10::optional<Generator> generator) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto keys = at::empty(result.sizes(), result.options()).random_(generator);
  scalar_t* keys_data = keys.data_ptr<scalar_t>();

  Tensor shuffled;
  scalar_t* shuffled_data;
  if (result.is_contiguous()) {
    shuffled = result;
    shuffled_data = result.data_ptr<scalar_t>();
  } else {
    shuffled = at::empty(n, result.options());
    shuffled_data = shuffled.data_ptr<scalar_t>();
  }

  xpu::pstl::iota(shuffled_data, shuffled_data + n, scalar_t(0));
  xpu::pstl::merge_sort<scalar_t, scalar_t>(
      keys_data, shuffled_data, keys.size(0), [](scalar_t a, scalar_t b) {
        return Numerics<scalar_t>::lt(a, b);
      });

  if (!result.is_contiguous()) {
    result.copy_(shuffled);
  }
  return result;
}

namespace triangle_dpcpp {
// To find the max integer that does not exceed the root of an int64_t variable,
// we could use a loop to test one bit at a time, which takes up to 31
// iterations. This would give the accurate result, but is relatively slow and
// is an overkill for most cases where double's precision suffice.
//
// If we directly use sqrt to calculate the root, the convertion from int64_t
// to double would lose 11 bits precision.
//
// The following solution uses sqrt directly for most cases, and would only
// special handle it if there is indeed precision loss.
inline int64_t resolve_root_int(
    int64_t b,
    int64_t cX4,
    int64_t x,
    int32_t sign) {
  int64_t bXb_cX4 = b * b - cX4;
  // potential precision loss could occur here when casting int64_t (63 bits
  // precision) to double (52 bits precision)
  double sr = Numerics<double>::sqrt((double)bXb_cX4);
  //
  // TODO: PyTorch uses ::__double2ll_rd. No corresponding API in DPCPP.
  // uses std::llround or std::ceil or std::float will cause error:
  // terminate called after throwing an instance of
  // 'sycl::compile_program_error'.
  //
  int64_t res = static_cast<int64_t>((-b + sign * sr) / 2);

  // have to cast double to int64_t, otherwise it would only compare up to the
  // precision of a double variable, ignoring the precision loss
  if (bXb_cX4 != (int64_t)(sr * sr)) {
    // TODO:PyTorch uses ::__double2ll_rd && ::__double2ll_ru. No corresponding
    // API in DPCPP.
  }

  return res;
}

inline void get_coordinate_in_triu_trapezoid(
    int64_t f,
    int64_t x,
    int64_t& row,
    int64_t& col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = -1 - f;
  auto cX4 = x << 3; // 4 * c = 4 * (2x) = 8x;
  row = resolve_root_int(b, cX4, x, -1);
  col = x - ((f - row + 1) * row >> 1) + row;
}

inline void get_coordinate_in_tril_trapezoid(
    int64_t f,
    int64_t x,
    int64_t& row,
    int64_t& col) {
  f <<= 1; // all statements use 2f, so only calculate it once here.
  auto b = f - 1;
  auto cX4 = -(x << 3); // 4 * c = 4 * (-2x) = -8x;
  row = resolve_root_int(b, cX4, x, 1);
  col = x - ((f + row - 1) * row >> 1);
}

} // namespace triangle_dpcpp

template <typename scalar_t>
void triu_indices_dpcpp_kernel(
    scalar_t* tensor,
    int64_t col_offset,
    int64_t m_first_row,
    int64_t col,
    int64_t rectangle_size,
    int64_t triu_size) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dev_id);
  auto totalElements = triu_size;
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto data = tensor;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto tensor_ptr = data;
      int64_t r, c;
      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements;
           linearIndex += item.get_global_range()[0]) {
        if (linearIndex < rectangle_size) {
          // the coordinate is within the top rectangle
          r = linearIndex / col;
          c = linearIndex % col;
        } else {
          // the coordinate falls in the bottom trapezoid
          triangle_dpcpp::get_coordinate_in_triu_trapezoid(
              m_first_row, linearIndex - rectangle_size, r, c);
          r += rectangle_size / col;
        }
        c += col_offset;
        tensor_ptr[linearIndex] = r;
        tensor_ptr[linearIndex + triu_size] = c;
      }
    };
    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void tril_indices_dpcpp_kernel(
    scalar_t* tensor,
    int64_t row_offset,
    int64_t m_first_row,
    int64_t col,
    int64_t trapezoid_size,
    int64_t tril_size) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dev_id);
  auto totalElements = tril_size;
  auto num_groups = CeilDiv(totalElements, group_size);
  auto total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto data = tensor;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto tensor_ptr = data;
      int64_t r, c;
      for (int64_t linearIndex = item.get_global_id(0);
           linearIndex < totalElements;
           linearIndex += item.get_global_range()[0]) {
        if (linearIndex < trapezoid_size) {
          // the coordinate is within the top trapezoid
          triangle_dpcpp::get_coordinate_in_tril_trapezoid(
              m_first_row, linearIndex, r, c);
        } else {
          // the coordinate falls in the bottom rectangle
          auto surplus = linearIndex - trapezoid_size;
          // add the height of trapezoid: m_last_row (col) - m_first_row + 1
          r = surplus / col + col - m_first_row + 1;
          c = surplus % col;
        }
        r += row_offset;
        tensor_ptr[linearIndex] = r;
        tensor_ptr[linearIndex + tril_size] = c;
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(total_items), sycl::range<1>(group_size)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

Tensor triu_indices_dpcpp(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options) {
  check_args(row, col, options.layout());

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);
  auto tensor = empty_dpcpp({2, triu_size}, options, c10::nullopt);

  if (triu_size > 0) {
    // # of triu elements in the first row
    auto m_first_row = (offset > 0) ? std::max<int64_t>(col - offset, 0)
                                    : // upper bounded by col
        col;

    // size of the top rectangle
    int64_t rectangle_size = 0;
    if (offset < 0) {
      rectangle_size = std::min<int64_t>(row, -offset) * col;
    }

    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, tensor.scalar_type(), "triu_indices_dpcpp", [&] {
          triu_indices_dpcpp_kernel<scalar_t>(
              tensor.data_ptr<scalar_t>(),
              Numerics<int64_t>::max(0, offset),
              m_first_row,
              col,
              rectangle_size,
              triu_size);
        });
  }

  return tensor;
}

Tensor tril_indices_dpcpp(
    int64_t row,
    int64_t col,
    int64_t offset,
    const TensorOptions& options) {
  check_args(row, col, options.layout());

  auto tril_size = get_tril_size(row, col, offset);
  auto tensor = empty_dpcpp({2, tril_size}, options, c10::nullopt);

  if (tril_size > 0) {
    auto m_first_row = (offset > 0) ? std::min<int64_t>(col, 1 + offset)
                                    : // upper bounded by col
        (row + offset > 0); // either 0 or 1
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
    auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;

    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row) {
      rectangle_size = (row - rectangle_row_offset) * col;
    }

    IPEX_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, tensor.scalar_type(), "tril_indices_dpcpp", [&] {
          tril_indices_dpcpp_kernel<scalar_t>(
              tensor.data_ptr<scalar_t>(),
              trapezoid_row_offset,
              m_first_row,
              col,
              tril_size - rectangle_size,
              tril_size);
        });
  }

  return tensor;
}

} // namespace impl

Tensor _efficientzerotensor(
    IntArrayRef size,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory) {
  auto device_ = device_or_default(device);
  if (!device_.has_index()) {
    device_.set_index(current_device());
  }
  auto allocator = at::native::ZeroTensorAllocator(device_);
  auto dtype_ = dtype_or_default(dtype);
  auto zero_ks = at::DispatchKeySet(c10::DispatchKey::XPU) |
      at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
  auto out = at::detail::empty_generic(
      size, &allocator, zero_ks, dtype_, c10::nullopt);
  return out;
}

Tensor empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  return AtenIpexTypeXPU::impl::empty_dpcpp(
      size, options, optional_memory_format);
}

Tensor empty(
    IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return empty(size, options, optional_memory_format);
}

Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return AtenIpexTypeXPU::impl::empty_strided_dpcpp(size, stride, options);
}

Tensor& eye_out(int64_t n, Tensor& out) {
  AtenIpexTypeXPU::impl::eye_out_dpcpp(out, n);
  return out;
}

Tensor& eye_out(int64_t n, int64_t m, Tensor& out) {
  AtenIpexTypeXPU::impl::eye_out_dpcpp(out, n, m);
  return out;
}

Tensor& randperm_out(
    int64_t n,
    c10::optional<Generator> generator,
    Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  check_supported_max_int_with_precision(n, result);
  result.resize_({n});
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "randperm", [&]() -> void {
        AtenIpexTypeXPU::impl::randperm_dpcpp<scalar_t>(result, n, generator);
      });

  return result;
}

Tensor tril_indices(
    int64_t row,
    int64_t col,
    int64_t offset,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return AtenIpexTypeXPU::impl::tril_indices_dpcpp(row, col, offset, options);
}

Tensor triu_indices(
    int64_t row,
    int64_t col,
    int64_t offset,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return AtenIpexTypeXPU::impl::triu_indices_dpcpp(row, col, offset, options);
}

Tensor var(
    const Tensor& self,
    c10::OptionalIntArrayRef _dim,
    c10::optional<int64_t> _correction,
    bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  auto dim = _dim.value_or(IntArrayRef{});
  auto correction = _correction.value_or(1);
  return at::AtenIpexTypeXPU::std_var_out(
      result, self, dim, correction, keepdim, false);
}

Tensor _var(const Tensor& self, bool unbiased) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::std_var_out(
      result, self, IntArrayRef{}, int64_t{unbiased ? 1 : 0}, false, false);
}

at::Tensor& var_out(
    const at::Tensor& self,
    c10::OptionalArrayRef<int64_t> opt_dim,
    c10::optional<int64_t> _correction,
    bool keepdim,
    at::Tensor& out) {
  auto correction = _correction.value_or(1);
  return at::AtenIpexTypeXPU::std_var_out(
      out, self, opt_dim, correction, keepdim, false);
}

Tensor _std(const Tensor& self, bool unbiased) {
  Tensor result = at::empty({0}, self.options());
  return at::AtenIpexTypeXPU::std_var_out(
      result, self, IntArrayRef{}, int64_t{unbiased ? 1 : 0}, false, true);
}

Tensor std(
    const Tensor& self,
    c10::OptionalArrayRef<int64_t> _dim,
    c10::optional<int64_t> _correction,
    bool keepdim) {
  Tensor result = at::empty({0}, self.options());
  auto correction = _correction.value_or(1);
  auto dim = _dim.value_or(IntArrayRef{});
  return at::AtenIpexTypeXPU::std_var_out(
      result, self, dim, correction, keepdim, true);
}

Tensor& std_out(
    const Tensor& self,
    c10::OptionalArrayRef<int64_t> _dim,
    c10::optional<int64_t> _correction,
    bool keepdim,
    Tensor& out) {
  auto correction = _correction.value_or(1);
  auto dim = _dim.value_or(IntArrayRef{});
  return at::AtenIpexTypeXPU::std_var_out(
      out, self, dim, correction, keepdim, true);
}

std::tuple<Tensor, Tensor> var_mean(
    const Tensor& self,
    c10::OptionalArrayRef<int64_t> _dim,
    c10::optional<int64_t> _correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  auto dim = _dim.value_or(IntArrayRef{});
  auto correction = _correction.value_or(1);
  return at::AtenIpexTypeXPU::std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> std_mean(
    const Tensor& self,
    c10::OptionalArrayRef<int64_t> _dim,
    c10::optional<int64_t> _correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, self.options());
  Tensor result2 = at::empty({0}, self.options());
  auto dim = _dim.value_or(IntArrayRef{});
  auto correction = _correction.value_or(1);
  return at::AtenIpexTypeXPU::std_var_mean_out(
      "std_mean", result1, result2, self, dim, correction, keepdim, true);
}

Tensor view_as_real(const at::Tensor& self) {
  return at::native::view_as_real(self);
}

Tensor view_as_complex(const Tensor& self) {
  return at::native::view_as_complex(self);
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
Tensor empty(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format) {
  return AtenIpexTypeXPU::impl::empty_dpcpp(
      size, options, optional_memory_format);
}

Tensor empty_strided(
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return AtenIpexTypeXPU::impl::empty_strided_dpcpp(size, stride, options);
}

Tensor empty(
    IntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<MemoryFormat> optional_memory_format) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  return empty(size, options, optional_memory_format);
}

Tensor empty_like(
    const Tensor& self,
    c10::optional<ScalarType> dtype = c10::nullopt,
    c10::optional<Layout> layout = c10::nullopt,
    c10::optional<Device> device = c10::nullopt,
    c10::optional<bool> pin_memory = c10::nullopt,
    c10::optional<c10::MemoryFormat> optional_memory_format = c10::nullopt) {
  return at::native::empty_like_quantized(
      self, dtype, layout, device, pin_memory, optional_memory_format);
}

Tensor empty_quantized(
    IntArrayRef size,
    const Tensor& qtensor,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  return AtenIpexTypeXPU::impl::empty_quantized(
      size, qtensor, dtype, layout, device, pin_memory, memory_format);
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
