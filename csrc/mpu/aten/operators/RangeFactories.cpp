#include <ATen/NativeFunctions.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include "comm/AccumulateType.h"

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include <cmath>
#include <limits>

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename T, typename accT = T>
class LinspaceOp {
 public:
  LinspaceOp(accT start, accT step) : start_(start), step_(step) {}
  T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT value = start_ + increment;
    return static_cast<T>(value);
  }

  const accT start_, step_;
};

template <typename T, typename accT = T>
struct LogspaceOp {
  LogspaceOp(accT start, accT step, accT base)
      : start_(start), step_(step), base_(base) {}
  T operator()(ptrdiff_t index) {
    accT increment = step_ * static_cast<accT>(index);
    accT value = Numerics<accT>::pow(base_, start_ + increment);
    return static_cast<T>(value);
  }

  const accT start_, step_, base_;
};

// TODO: move it to the loops for more generic supporting.
template <typename scalar_t, typename func_t>
void dpcpp_elementwise_kernel_with_index_impl(
    scalar_t* out_ptr,
    int64_t N,
    func_t f) {
  if (N > std::numeric_limits<int>::max()) {
    dpcpp_elementwise_kernel_with_index_impl(
        out_ptr + std::numeric_limits<int>::max(),
        N - std::numeric_limits<int>::max(),
        f);
  }
  int64_t thread_number = std::min(N, (int64_t)std::numeric_limits<int>::max());
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto kfn = DPCPP_Q_KFN(sycl::item<1> item_id) {
      auto idx = item_id.get_linear_id();
      out_ptr[idx] = f(idx);
    };
    cgh.parallel_for(sycl::range</*dim=*/1>(thread_number), kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

template <typename func_t>
void dpcpp_elementwise_kernel_with_index(at::Tensor& output, func_t f) {
  using scalar_t = typename function_traits<func_t>::result_type;
  int64_t N = output.numel();
  if (N == 0) {
    return;
  }

  return dpcpp_elementwise_kernel_with_index_impl(
      output.data_ptr<scalar_t>(), N, f);
}

Tensor& linspace_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    c10::optional<int64_t> optional_steps) {
  const auto steps = optional_steps.value_or(100);
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (!optional_steps.has_value()) {
    TORCH_WARN_ONCE(
        "Not providing a value for linspace's steps is deprecated and will "
        "throw a runtime error in a future release. This warning will appear "
        "only once per process.");
  }

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous
      ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else if (isIntegralType(r.scalar_type(), 0)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "linspace_xpu", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // Cast `end` and `start` to `float`, since range can be larger than
      // scalar_t for integral types
      float step =
          (static_cast<float>(scalar_end) - static_cast<float>(scalar_start)) /
          (steps - 1);
      const int64_t halfway = steps / 2;
      dpcpp_elementwise_kernel_with_index(
          r,
          [scalar_start, scalar_end, steps, step, halfway](
              int64_t ind) -> scalar_t {
            if (ind < halfway) {
              return scalar_start + (step * ind);
            }

            return scalar_end - step * (steps - ind - 1);
          });
    });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, r.scalar_type(), "linspace_xpu", [&]() {
          scalar_t scalar_start = start.to<scalar_t>();
          scalar_t scalar_end = end.to<scalar_t>();
          scalar_t step =
              (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
          const int64_t halfway = steps / 2;
          dpcpp_elementwise_kernel_with_index(
              r,
              [scalar_start, scalar_end, steps, step, halfway](
                  int64_t ind) -> scalar_t {
                if (ind < halfway) {
                  return scalar_start + (step * ind);
                }

                return scalar_end - step * (steps - ind - 1);
              });
        });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

Tensor& logspace_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    c10::optional<int64_t> optional_steps,
    double base) {
  const auto steps = optional_steps.value_or(100);
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (!optional_steps.has_value()) {
    TORCH_WARN_ONCE(
        "Not providing a value for logspace's steps is deprecated and will "
        "throw a runtime error in a future release. This warning will appear "
        "only once per process.");
  }

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous
      ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())) {
      r.fill_(Numerics<c10::complex<double>>::pow(
          base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(Numerics<double>::pow(base, start.to<double>()));
    }
  } else if (isIntegralType(r.scalar_type(), /*includeBool=*/false)) {
    IPEX_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "logspace_xpu", [&]() {
      float scalar_base =
          static_cast<float>(base); // Use float to avoid promotion to double
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      float step = static_cast<float>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      dpcpp_elementwise_kernel_with_index(
          r,
          [scalar_start, scalar_end, scalar_base, steps, step, halfway](
              int64_t ind) -> scalar_t {
            if (ind < halfway) {
              return std::pow(scalar_base, scalar_start + step * ind);
            }
            return std::pow(scalar_base, scalar_end - step * (steps - ind - 1));
          });
    });
  } else {
    IPEX_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, r.scalar_type(), "logspace_xpu", [&]() {
          scalar_t scalar_base = static_cast<scalar_t>(base);
          scalar_t scalar_start = start.to<scalar_t>();
          scalar_t scalar_end = end.to<scalar_t>();
          scalar_t step =
              (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
          const int64_t halfway = steps / 2;
          dpcpp_elementwise_kernel_with_index(
              r,
              [scalar_start, scalar_end, scalar_base, steps, step, halfway](
                  int64_t ind) -> scalar_t {
                if (ind < halfway) {
                  return std::pow(scalar_base, scalar_start + step * ind);
                }
                return std::pow(
                    scalar_base, scalar_end - step * (steps - ind - 1));
              });
        });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

Tensor& range_dpcpp_out(Tensor& result, Scalar start, Scalar end, Scalar step) {
  IPEX_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "range_dpcpp", [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");
        int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
        if (result.numel() != size) {
          result.resize_({size});
        }
        Tensor r = result.is_contiguous() ? result : result.contiguous();
        LinspaceOp<scalar_t, accscalar_t> linspace_method(xstart, xstep);
        auto& dpcpp_queue = dpcppGetCurrentQueue();

        dpcpp_elementwise_kernel_with_index(
            r, [xstart, xstep](int64_t ind) -> scalar_t {
              accscalar_t inc = xstep * static_cast<accscalar_t>(ind);
              accscalar_t val = xstart + inc;
              return static_cast<scalar_t>(val);
            });

        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      });

  return result;
}

Tensor& arange_dpcpp_out(
    Tensor& result,
    Scalar start,
    Scalar end,
    Scalar step) {
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      result.scalar_type(),
      "arange_dpcpp",
      [&]() {
        using accscalar_t = acc_type<scalar_t>;
        auto xstart = start.to<accscalar_t>();
        auto xend = end.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        double size_d;
        if (std::is_same<scalar_t, int64_t>::value) {
          size_d = std::ceil(
              static_cast<double>(
                  end.to<accscalar_t>() - start.to<accscalar_t>()) /
              step.to<accscalar_t>());
        } else {
          size_d = std::ceil(
              static_cast<double>(end.to<double>() - start.to<double>()) /
              step.to<double>());
        }

        TORCH_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        TORCH_CHECK(
            std::isfinite(static_cast<double>(xstart)) &&
                std::isfinite(static_cast<double>(xend)),
            "unsupported range: ",
            xstart,
            " -> ",
            xend);
        TORCH_CHECK(
            ((xstep > 0) && (xend >= xstart)) ||
                ((xstep < 0) && (xend <= xstart)),
            "upper bound and larger bound inconsistent with step sign");

        TORCH_CHECK(
            size_d >= 0 &&
                size_d <=
                    static_cast<double>(std::numeric_limits<int64_t>::max()),
            "invalid size, possible overflow?");
        int64_t size = static_cast<int64_t>(size_d);
        int64_t numel = result.numel();

        if (size == 0) {
          return;
        } else if (numel != size) {
          if (numel > 0) {
            TORCH_WARN(
                "The number of elements in the out tensor of shape ",
                result.sizes(),
                " is ",
                numel,
                " which does not match the computed number of elements ",
                size,
                ". Note that this may occur as a result of rounding error. "
                "The out tensor will be resized to a tensor of shape (",
                size,
                ",).");
          }
          result.resize_({size});
        }
        auto& dpcpp_queue = dpcppGetCurrentQueue();
        const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
        auto max_wgroup_size = dpcppMaxWorkGroupSize(dev_id);
        const auto wgroup_size_col =
            (size > max_wgroup_size) ? max_wgroup_size : size;
        const auto wgroup_num_col =
            (size + wgroup_size_col - 1) / wgroup_size_col;

        // command group functions
        auto cgf = DPCPP_Q_CGF(cgh) {
          auto acc = result.data_ptr<scalar_t>();
          dpcpp_local_acc_t<accscalar_t> slm_xstart(1, cgh);
          dpcpp_local_acc_t<accscalar_t> slm_xstep(1, cgh);

          // kernel function per work-item
          auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item_id) {
            slm_xstart[0] = xstart;
            slm_xstep[0] = xstep;
            auto ptr = acc;
            auto global_id_col = item_id.get_global_id(0);
            if (global_id_col < size) {
              ptr[global_id_col] = slm_xstart[0] + global_id_col * slm_xstep[0];
            }
          };
          // kick off kernel
          cgh.parallel_for(
              sycl::nd_range</*dim=*/1>(
                  sycl::range<1>(wgroup_num_col * wgroup_size_col),
                  sycl::range<1>(wgroup_size_col)),
              kfn);
        };

        // submit to DPCPP queue
        DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
      });
  return result;
}

} // namespace impl

Tensor& linspace_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& out) {
  impl::linspace_dpcpp_out(out, start, end, steps);
  return out;
}

Tensor& logspace_out(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    double base,
    Tensor& out) {
  impl::logspace_dpcpp_out(out, start, end, steps, base);
  return out;
}

Tensor& range_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  impl::range_dpcpp_out(out, start, end, step);
  return out;
}
Tensor& arange_out(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  impl::arange_dpcpp_out(out, start, end, step);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
