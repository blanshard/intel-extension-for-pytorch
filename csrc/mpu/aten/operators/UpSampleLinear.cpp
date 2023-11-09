#include <ATen/native/UpSample.h>
#include <tensor/Tensor.h>

#include "UpSample.h"
#include "comm/RegistrationDeclarations.h"

#include <core/MemoryFormat.h>
#include <oneDNN/oneDNN.h>
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"

using namespace dnnl;
using namespace at::native;
using namespace at::AtenIpexTypeXPU;
using namespace xpu::dpcpp;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t, typename accscalar_t>
void upsample_bilinear2d_out_frame(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* idata,
    scalar_t* odata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();

  int num_group = CeilDiv((int64_t)n, (int64_t)1024);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = idata;
    auto out_data = odata;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto in_ptr = idata;
      auto out_ptr = out_data;
      int index = item.get_global_linear_id();

      if (index < n) {
        const int output_x = index % output_width;
        const int output_y = index / output_width;

        const accscalar_t h1r = area_pixel_compute_source_index<scalar_t>(
            rheight, output_y, align_corners, /*cubic=*/false);
        const int h1 = h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const accscalar_t h1lambda = h1r - h1;
        const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

        const accscalar_t w1r = area_pixel_compute_source_index<scalar_t>(
            rwidth, output_x, align_corners, /*cubic=*/false);
        const int w1 = w1r;
        const int w1p = (w1 < input_width - 1) ? 1 : 0;
        const accscalar_t w1lambda = w1r - w1;
        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; ++c) {
            auto val = h0lambda *
                    (w0lambda *
                         in_ptr
                             [n * input_height * input_width * channels +
                              c * input_height * input_width +
                              h1 * input_width + w1] +

                     w1lambda *
                         in_ptr
                             [n * input_height * input_width * channels +
                              c * input_height * input_width +
                              h1 * input_width + w1 + w1p]) +

                h1lambda *
                    (w0lambda *
                         in_ptr
                             [n * input_height * input_width * channels +
                              c * input_height * input_width +
                              (h1 + h1p) * input_width + w1] +

                     w1lambda *
                         in_ptr
                             [n * input_height * input_width * channels +
                              c * input_height * input_width +
                              (h1 + h1p) * input_width + w1 + w1p]);

            out_ptr
                [n * output_height * output_width * channels +
                 c * output_height * output_width + output_y * output_width +
                 output_x] = val;
          }
        }
      }
    };

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(num_group * 1024), sycl::range<1>(1024)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

size_t idx(
    const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
void upsample_bilinear2d_backward_out_frame(
    const size_t nc,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    scalar_t* idata,
    const scalar_t* odata) {
  const size_t o_numel = nc * output_width * output_height;
  const size_t i_numel = nc * input_width * input_height;

  const size_t num_kernels = nc * output_width * output_height;
  int num_groups = CeilDiv((int64_t)num_kernels, (int64_t)1024);

  auto& dpcpp_queue = dpcppGetCurrentQueue();

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = idata;
    auto out_data = odata;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto in_ptr = in_data;
      auto out_ptr = out_data;

      for (size_t index = item.get_local_id(0) +
               item.get_group(0) * item.get_local_range(0);
           index < o_numel;
           index += item.get_local_range(0) * item.get_group_range(0)) {
        size_t index_temp = index;
        const int w2 = index_temp % output_width;
        index_temp /= output_width;
        const int h2 = index_temp % output_height;
        const size_t nc = index_temp / output_height;

        const accscalar_t h1r = area_pixel_compute_source_index<scalar_t>(
            rheight, h2, align_corners, /*cubic=*/false);
        const int h1 = h1r;
        const int h1p = (h1 < input_height - 1) ? 1 : 0;
        const accscalar_t h1lambda = h1r - h1;
        const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

        const accscalar_t w1r = area_pixel_compute_source_index<scalar_t>(
            rwidth, w2, align_corners, /*cubic=*/false);
        const int w1 = w1r;
        const int w1p = (w1 < input_width - 1) ? 1 : 0;
        const accscalar_t w1lambda = w1r - w1;
        const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

        const scalar_t d2val = odata[index];

        atomicAdd(
            (dpcpp_global_ptr_pt<
                scalar_t>)(in_ptr + idx(nc, input_height, input_width, h1, w1)),
            static_cast<scalar_t>(h0lambda * w0lambda * d2val));

        atomicAdd(
            (dpcpp_global_ptr_pt<
                scalar_t>)(in_ptr + idx(nc, input_height, input_width, h1, w1 + w1p)),
            static_cast<scalar_t>(h0lambda * w1lambda * d2val));

        atomicAdd(
            (dpcpp_global_ptr_pt<
                scalar_t>)(in_ptr + idx(nc, input_height, input_width, h1 + h1p, w1)),
            static_cast<scalar_t>(h1lambda * w0lambda * d2val));

        atomicAdd(
            (dpcpp_global_ptr_pt<
                scalar_t>)(in_ptr + idx(nc, input_height, input_width, h1 + h1p, w1 + w1p)),
            static_cast<scalar_t>(h1lambda * w1lambda * d2val));
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(num_groups * 1024), sycl::range<1>(1024)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

static void upsample_bilinear2d_out_dpcpp_template(
    const Tensor& output,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  output.resize_({nbatch, channels, output_height, output_width});

  if (input.sizes() == output.sizes()) {
    output.copy_(input);
    return;
  }
  const int num_kernels = output_height * output_width;

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "upsample_bilinear2d_out_frame",
      [&] {
        using accscalar_t = acc_type<scalar_t>;

        auto* idata = input.data_ptr<scalar_t>();
        auto* odata = output.data_ptr<scalar_t>();

        const accscalar_t rheight = area_pixel_compute_scale<scalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<scalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_bilinear2d_out_frame<scalar_t, accscalar_t>(
            num_kernels,
            rheight,
            rwidth,
            align_corners,
            idata,
            odata,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels);
      });
}

static void upsample_bilinear2d_backward_out_dpcpp_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input_size[0];
  int channels = input_size[1];
  int input_height = input_size[2];
  int input_width = input_size[3];

  Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_height, input_width});
  if (grad_input.numel() == 0) {
    return;
  }

  grad_input.contiguous();
  grad_input.zero_();

  if (grad_output.sizes() == grad_input.sizes()) {
    grad_input.copy_(grad_output_);
    return;
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad_output.scalar_type(),
      "upsample_bilinear2d_backward_out_frame",
      [&] {
        using accscalar_t = acc_type<scalar_t>;

        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();

        const accscalar_t rheight = area_pixel_compute_scale<scalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t rwidth = area_pixel_compute_scale<scalar_t>(
            input_width, output_width, align_corners, scales_w);

        upsample_bilinear2d_backward_out_frame<scalar_t, accscalar_t>(
            nbatch * channels,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            rheight,
            rwidth,
            align_corners,
            idata,
            odata);
      });
}

} // namespace impl

using namespace impl;
using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor& upsample_trilinear3d_out(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  TORCH_CHECK(
      align_corners == false,
      "we don't support align_cornser path by currently as oneDNN don't support this "
      "algorithm!\n")
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_linear,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return output;
}

Tensor upsample_trilinear3d(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_linear,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return output;
}

Tensor upsample_trilinear3d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  auto output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  xpu::oneDNN::resample(
      input,
      output,
      osize,
      algorithm::resampling_linear,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0f,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0f,
      scale_d.has_value() ? static_cast<double>(scale_d.value()) : 0.0f);
  return output;
}

Tensor& upsample_trilinear3d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_linear,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return grad_input;
}

Tensor upsample_trilinear3d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  auto ndim = grad_output.ndimension();
  Tensor grad_input;
  if (is_smf_channels_last(grad_output)) {
    grad_input =
        at::empty(input_size, grad_output.options(), get_cl_tag_by_ndim(ndim));
  } else {
    grad_input = at::empty(input_size, grad_output.options());
  }

  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_linear,
      scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
      scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f,
      scales_d.has_value() ? static_cast<double>(scales_d.value()) : 0.0f);
  return grad_input;
}

Tensor upsample_trilinear3d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_d = get_scale_value(scale_factors, 0);
  auto scale_h = get_scale_value(scale_factors, 1);
  auto scale_w = get_scale_value(scale_factors, 2);
  auto ndim = grad_output.ndimension();

  Tensor grad_input;

  if (is_smf_channels_last(grad_output)) {
    grad_input =
        at::empty(input_size, grad_output.options(), get_cl_tag_by_ndim(ndim));
  } else {
    grad_input = at::empty(input_size, grad_output.options());
  }

  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      osize,
      algorithm::resampling_linear,
      scale_w.has_value() ? static_cast<double>(scale_w.value()) : 0.0f,
      scale_h.has_value() ? static_cast<double>(scale_h.value()) : 0.0f,
      scale_d.has_value() ? static_cast<double>(scale_d.value()) : 0.0f);
  return grad_input;
}

Tensor& upsample_bilinear2d_out(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& output) {
  if (align_corners) {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    upsample_bilinear2d_out_dpcpp_template(
        output, input, output_size, true, scales_h, scales_w);
  } else {
    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_linear,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  }
  return output;
}

Tensor upsample_bilinear2d(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, input.options());
  if (align_corners) {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    upsample_bilinear2d_out_dpcpp_template(
        output, input, output_size, true, scales_h, scales_w);
  } else {
    xpu::oneDNN::resample(
        input,
        output,
        output_size,
        algorithm::resampling_linear,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  }
  return output;
}

Tensor upsample_bilinear2d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scales_h = get_scale_value(scale_factors, 0);
  auto scales_w = get_scale_value(scale_factors, 1);
  if (align_corners) {
    at::AtenIpexTypeXPU::to_plain_if_needed_(input);
    upsample_bilinear2d_out_dpcpp_template(
        output, input, osize, true, scales_h, scales_w);
  } else {
    xpu::oneDNN::resample(
        input,
        output,
        osize,
        algorithm::resampling_linear,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  }
  return output;
}

Tensor& upsample_bilinear2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  if (align_corners) {
    at::AtenIpexTypeXPU::to_plain_if_needed_(grad_output);
    upsample_bilinear2d_backward_out_dpcpp_template(
        grad_input,
        grad_output,
        output_size,
        input_size,
        true, // align_corners
        scales_h,
        scales_w);
  } else {
    xpu::oneDNN::resample_backward(
        grad_input,
        grad_output,
        input_size,
        output_size,
        algorithm::resampling_linear,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  }
  return grad_input;
}

Tensor upsample_bilinear2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto ndim = grad_output.ndimension();
  Tensor grad_input;
  if (is_smf_channels_last(grad_output)) {
    grad_input =
        at::empty(input_size, grad_output.options(), get_cl_tag_by_ndim(ndim));
  } else {
    grad_input = at::empty(input_size, grad_output.options());
  }

  if (align_corners) {
    at::AtenIpexTypeXPU::to_plain_if_needed_(grad_output);
    upsample_bilinear2d_backward_out_dpcpp_template(
        grad_input,
        grad_output,
        output_size,
        input_size,
        true, // align_corners
        scales_h,
        scales_w);
  } else {
    xpu::oneDNN::resample_backward(
        grad_input,
        grad_output,
        input_size,
        output_size,
        algorithm::resampling_linear,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  }
  return grad_input;
}

Tensor upsample_bilinear2d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scales_h = get_scale_value(scale_factors, 0);
  auto scales_w = get_scale_value(scale_factors, 1);
  auto ndim = grad_output.ndimension();
  Tensor grad_input;
  if (is_smf_channels_last(grad_output)) {
    grad_input =
        at::empty(input_size, grad_output.options(), get_cl_tag_by_ndim(ndim));
  } else {
    grad_input = at::empty(input_size, grad_output.options());
  }

  if (align_corners) {
    at::AtenIpexTypeXPU::to_plain_if_needed_(grad_output);
    upsample_bilinear2d_backward_out_dpcpp_template(
        grad_input,
        grad_output,
        osize,
        input_size,
        true, // align_corners
        scales_h,
        scales_w);
  } else {
    xpu::oneDNN::resample_backward(
        grad_input,
        grad_output,
        input_size,
        osize,
        algorithm::resampling_linear,
        scales_w.has_value() ? static_cast<double>(scales_w.value()) : 0.0f,
        scales_h.has_value() ? static_cast<double>(scales_h.value()) : 0.0f);
  }
  return grad_input;
}

Tensor& upsample_linear1d_out(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    Tensor& output) {
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  xpu::oneDNN::resample(
      input,
      output,
      output_size,
      algorithm::resampling_linear,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
  return output;
}

Tensor& upsample_linear1d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales,
    Tensor& grad_input) {
  TORCH_CHECK(
      align_corners == false,
      "We don't support align_corners currently as oneDNN don't support this "
      "algorithm!\n");
  xpu::oneDNN::resample_backward(
      grad_input,
      grad_output,
      input_size,
      output_size,
      algorithm::resampling_linear,
      scales.has_value() ? static_cast<double>(scales.value()) : 0.0f);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
