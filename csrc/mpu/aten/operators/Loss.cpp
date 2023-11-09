#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/native/TensorIterator.h>

#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "comm/ATDispatch.h"
#include "comm/Numerics.h"

#include "Loops.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

Tensor& sum_out(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<at::ScalarType> opt_dtype,
    Tensor& result);

Tensor& mean_out(
    const Tensor& self,
    c10::OptionalArrayRef<int64_t> opt_dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype,
    Tensor& result);

static inline at::Tensor apply_loss_reduction(
    const at::Tensor& unreduced,
    int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

namespace impl {

void bce_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "bce_kernel",
      [&iter]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t input, scalar_t target) -> scalar_t {
              SYCL_KERNEL_ASSERT(input >= 0 && input <= 1);
              return (target - scalar_t(1)) *
                  Numerics<scalar_t>::max(
                         scalar_t(Numerics<scalar_t>::log(scalar_t(1) - input)),
                         scalar_t(-100)) -
                  target *
                  Numerics<scalar_t>::max(
                      scalar_t(Numerics<scalar_t>::log(input)), scalar_t(-100));
            });
      });
}

void bce_backward_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, iter.dtype(), "bce_backward_kernel", [&iter] {
        dpcpp_kernel_for_tensor_iter(
            iter,
            [](scalar_t input,
               scalar_t target,
               scalar_t grad_output) -> scalar_t {
              return grad_output * (input - target) /
                  (scalar_t(std::max(
                      (scalar_t(1) - input) * input, scalar_t(1e-12))));
            });
      });
}

void soft_margin_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "soft_margin_kernel",
      [&iter]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t input, scalar_t target) -> scalar_t {
              return Numerics<scalar_t>::log(
                  scalar_t(1.) + Numerics<scalar_t>::exp(-input * target));
            });
      });
}

void soft_margin_backward_kernel(TensorIterator& iter, Scalar norm) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "soft_margin_backward_kernel",
      [&iter, &norm] {
        auto norm_val = norm.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter,
            [norm_val](scalar_t input, scalar_t target, scalar_t grad_output)
                -> scalar_t {
              auto z = Numerics<scalar_t>::exp(-target * input);
              return -norm_val * target * z / (scalar_t(1.) + z) * grad_output;
            });
      });
}

void smooth_l1_kernel(TensorIterator& iter, double beta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "smooth_l1_kernel",
      [&iter, beta]() {
        scalar_t beta_val(beta);
        dpcpp_kernel_for_tensor_iter(
            iter, [beta_val](scalar_t input, scalar_t target) -> scalar_t {
              auto z = Numerics<scalar_t>::abs(input - target);
              return z < beta_val ? scalar_t(0.5) * z * z / beta_val
                                  : z - scalar_t(0.5) * beta_val;
            });
      });
}

void smooth_l1_backward_kernel(TensorIterator& iter, Scalar norm, double beta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "smooth_l1_backward_kernel",
      [&iter, &norm, beta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t beta_val(beta);
        dpcpp_kernel_for_tensor_iter(
            iter,
            [norm_val, beta_val](
                scalar_t input,
                scalar_t target,
                scalar_t grad_output) -> scalar_t {
              const auto x = input - target;
              if (x < -beta_val)
                return -norm_val * grad_output;
              else if (x > beta_val)
                return norm_val * grad_output;
              else
                return norm_val * x * grad_output / beta_val;
            });
      });
}

void mse_kernel(TensorIterator& iter) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_kernel",
      [&iter]() {
        dpcpp_kernel_for_tensor_iter(
            iter, [](scalar_t input, scalar_t target) -> scalar_t {
              return (input - target) * (input - target);
            });
      });
}

void mse_backward_kernel(TensorIterator& iter, Scalar norm) {
  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_backward_kernel",
      [&iter, &norm] {
        auto norm_val = norm.to<scalar_t>();
        dpcpp_kernel_for_tensor_iter(
            iter,
            [norm_val](scalar_t input, scalar_t target, scalar_t grad_output)
                -> scalar_t {
              return norm_val * (input - target) * grad_output;
            });
      });
}

void huber_kernel(TensorIterator& iter, double delta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "huber", [&iter, delta] {
        scalar_t delta_val(delta);
        dpcpp_kernel_for_tensor_iter(
            iter, [delta_val](scalar_t a, scalar_t b) -> scalar_t {
              auto z = Numerics<scalar_t>::abs(a - b);
              return z < delta_val
                  ? scalar_t(0.5) * z * z
                  : delta_val * (z - scalar_t(0.5) * delta_val);
            });
      });
}

void huber_backward_kernel(
    TensorIterator& iter,
    const Scalar& norm,
    double delta) {
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, iter.dtype(), "huber_backward", [&iter, &norm, delta] {
        auto norm_val = norm.to<scalar_t>();
        scalar_t delta_val(delta);
        dpcpp_kernel_for_tensor_iter(
            iter,
            [norm_val, delta_val](
                scalar_t input,
                scalar_t target,
                scalar_t grad_output) -> scalar_t {
              const auto x = input - target;
              if (x < -delta_val) {
                return -norm_val * grad_output * delta_val;
              } else if (x > delta_val) {
                return norm_val * grad_output * delta_val;
              } else {
                return norm_val * x * grad_output;
              }
            });
      });
}

} // namespace impl

Tensor binary_cross_entropy(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor loss = at::empty_like(self);
  auto iter = TensorIterator::binary_op(loss, self, target);
  impl::bce_kernel(iter);
  if (weight.defined()) {
    loss.mul_(weight);
  }
  return apply_loss_reduction(loss, reduction);
}

Tensor& binary_cross_entropy_out(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    Tensor& out) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  if (reduction != Reduction::None) {
    Tensor loss = at::empty_like(self);
    auto iter = TensorIterator::binary_op(loss, self, target);
    impl::bce_kernel(iter);
    if (weight.defined()) {
      loss.mul_(weight);
    }
    if (reduction == Reduction::Mean) {
      at::AtenIpexTypeXPU::mean_out(
          loss, OptionalIntArrayRef{IntArrayRef{}}, false, c10::nullopt, out);
    } else {
      at::AtenIpexTypeXPU::sum_out(
          loss, OptionalIntArrayRef{IntArrayRef{}}, false, c10::nullopt, out);
    }
  } else {
    auto iter = TensorIterator::binary_op(out, self, target);
    impl::bce_kernel(iter);
    if (weight.defined()) {
      out.mul_(weight);
    }
  }
  return out;
}

Tensor& binary_cross_entropy_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    Tensor& grad_input) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .build();
  impl::bce_backward_kernel(iter);
  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  if (reduction == Reduction::Mean) {
    grad_input.div_(self.numel());
  }
  return grad_input;
}

Tensor binary_cross_entropy_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction) {
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  // XXX: Replace zeros_like with empty_like to avoid unnecessary zero fills
  Tensor grad_input = at::empty_like(
      self, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  return at::AtenIpexTypeXPU::binary_cross_entropy_backward_out(
      grad_output, self, target, weight, reduction, grad_input);
}

Tensor soft_margin_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, self, target);
  impl::soft_margin_kernel(iter);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& soft_margin_loss_out(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& out) {
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, self, target);
    impl::soft_margin_kernel(iter);
    if (reduction == Reduction::Mean) {
      at::AtenIpexTypeXPU::mean_out(
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt,
          out);
    } else {
      at::AtenIpexTypeXPU::sum_out(
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt,
          out);
    }
  } else {
    auto iter = TensorIterator::binary_op(out, self, target);
    impl::soft_margin_kernel(iter);
  }
  return out;
}

Tensor& soft_margin_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / self.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .build();
  impl::soft_margin_backward_kernel(iter, norm);
  return grad_input;
}

Tensor soft_margin_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::zeros_like(
      self, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  return at::AtenIpexTypeXPU::soft_margin_loss_backward_out(
      grad_output, self, target, reduction, grad_input);
}

Tensor l1_loss(const Tensor& self, const Tensor& target, int64_t reduction) {
  return apply_loss_reduction((self - target).abs(), reduction);
}

Tensor smooth_l1_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  if (beta == 0) {
    return at::AtenIpexTypeXPU::l1_loss(self, target, reduction);
  }
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, self, target);
  impl::smooth_l1_kernel(iter, beta);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& smooth_l1_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta,
    Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / self.numel() : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .promote_inputs_to_common_dtype(true)
                  .cast_common_dtype_to_outputs(true)
                  .enforce_safe_casting_to_output(true)
                  .build();
  impl::smooth_l1_backward_kernel(iter, norm, beta);
  return grad_input;
}

Tensor& smooth_l1_loss_out(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta,
    Tensor& out) {
  TORCH_CHECK(
      beta >= 0, "smooth_l1_loss does not support negative values for beta.")
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, self, target);
    impl::smooth_l1_kernel(iter, beta);
    if (reduction == Reduction::Mean) {
      at::AtenIpexTypeXPU::mean_out(
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt,
          out);
    } else {
      at::AtenIpexTypeXPU::sum_out(
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt,
          out);
    }
  } else {
    auto iter = TensorIterator::binary_op(out, self, target);
    impl::smooth_l1_kernel(iter, beta);
  }
  return out;
}

Tensor smooth_l1_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  Tensor grad_input = at::zeros_like(
      self, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  return at::AtenIpexTypeXPU::smooth_l1_loss_backward_out(
      grad_output, self, target, reduction, beta, grad_input);
}

Tensor mse_loss(const Tensor& self, const Tensor& target, int64_t reduction) {
  Tensor loss;
  auto iter = TensorIterator::binary_op(loss, self, target);
  impl::mse_kernel(iter);
  return apply_loss_reduction(iter.output(), reduction);
}

Tensor& mse_loss_out(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& out) {
  if (reduction != Reduction::None) {
    Tensor loss;
    auto iter = TensorIterator::binary_op(loss, self, target);
    impl::mse_kernel(iter);
    if (reduction == Reduction::Mean) {
      at::AtenIpexTypeXPU::mean_out(
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt,
          out);
    } else {
      at::AtenIpexTypeXPU::sum_out(
          iter.output(),
          OptionalIntArrayRef{IntArrayRef{}},
          false,
          c10::nullopt,
          out);
    }
  } else {
    auto iter = TensorIterator::binary_op(out, self, target);
    impl::mse_kernel(iter);
  }
  return out;
}

Tensor& mse_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 2. / self.numel() : 2.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .build();
  impl::mse_backward_kernel(iter, norm);
  return grad_input;
}

Tensor mse_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  Tensor grad_input = at::zeros_like(
      self, self.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));
  return at::AtenIpexTypeXPU::mse_loss_backward_out(
      grad_output, self, target, reduction, grad_input);
}

Tensor huber_loss(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double delta) {
  TORCH_CHECK(
      delta > 0, "huber_loss does not support non-positive values for delta.")
  Tensor loss = at::empty_like(self);
  auto iter = TensorIterator::binary_op(loss, self, target);
  impl::huber_kernel(iter, delta);
  return apply_loss_reduction(loss, reduction);
}

Tensor& huber_loss_out(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double delta,
    Tensor& out) {
  TORCH_CHECK(
      delta > 0, "huber_loss does not support non-positive values for delta.")
  auto iter = TensorIterator::borrowing_binary_op(out, self, target);
  impl::huber_kernel(iter, delta);
  if (reduction != Reduction::None) {
    auto reduced = apply_loss_reduction(out, reduction);
    out.resize_({});
    out.copy_(reduced);
  }
  return out;
}

Tensor huber_loss_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double delta) {
  auto grad_input = at::zeros_like(self, MemoryFormat::Contiguous);
  return at::huber_loss_backward_out(
      grad_input, grad_output, self, target, reduction, delta);
}

Tensor& huber_loss_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double delta,
    Tensor& grad_input) {
  auto norm = (reduction == Reduction::Mean) ? (1. / self.numel()) : 1.;
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_input(self)
                  .add_input(target)
                  .add_input(grad_output)
                  .build();
  impl::huber_backward_kernel(iter, norm, delta);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
