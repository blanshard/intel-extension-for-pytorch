#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>

#include <core/Memory.h>
#include <utils/DPCPP.h>
#include <utils/Helpers.h>
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

#include "UpSample.h"
#include "comm/Numerics.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
static void upsample_bicubic2d_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    int64_t onum,
    bool align_corners) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t local_range = static_cast<int64_t>(1);
  int64_t global_range = static_cast<int64_t>(1);
  if (onum != 0) {
    int64_t wg_size = dpcppMaxWorkGroupSize();
    local_range = onum < wg_size ? onum : wg_size;
    global_range = ((onum + local_range - 1) / local_range) * local_range;
  }

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = idata;
    auto out_data = odata;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto in_ptr = in_data;
      auto out_ptr = out_data;
      int global_id = item.get_global_linear_id();

      if (global_id < output_height * output_width) {
        const int output_x = global_id % output_width;
        const int output_y = global_id / output_width;
        if (input_height == output_height && input_width == output_width) {
          for (int n = 0; n < nbatch; n++) {
            for (int c = 0; c < channels; c++) {
              auto val = in_ptr
                  [n * input_height * input_width * channels +
                   c * input_height * input_width + output_y * input_width +
                   output_x];
              out_ptr
                  [n * output_height * output_width * channels +
                   c * output_height * output_width + output_y * output_width +
                   output_x] = val;
            }
          }
          return;
        }

        const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
            input_height, output_height, align_corners);
        const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
            input_width, output_width, align_corners);

        // Interpolation kernel
        scalar_t real_x = area_pixel_compute_source_index(
            width_scale, output_x, align_corners, /*cubic=*/true);
        // TODO: floor should support at dispatch to half type
        int in_x = Numerics<scalar_t>::floor(real_x);
        scalar_t t_x = real_x - in_x;

        scalar_t real_y = area_pixel_compute_source_index(
            height_scale, output_y, align_corners, /*cubic=*/true);
        int in_y = Numerics<scalar_t>::floor(real_y);
        scalar_t t_y = real_y - in_y;
        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; c++) {
            scalar_t coefficients[4];
            for (int k = 0; k < 4; k++) {
              coefficients[k] = cubic_interp1d(
                  upsample_get_value_bounded<scalar_t>(
                      in_ptr,
                      n,
                      c,
                      input_width,
                      input_height,
                      in_x - 1,
                      in_y - 1 + k),
                  upsample_get_value_bounded<scalar_t>(
                      in_ptr,
                      n,
                      c,
                      input_width,
                      input_height,
                      in_x + 0,
                      in_y - 1 + k),
                  upsample_get_value_bounded<scalar_t>(
                      in_ptr,
                      n,
                      c,
                      input_width,
                      input_height,
                      in_x + 1,
                      in_y - 1 + k),
                  upsample_get_value_bounded<scalar_t>(
                      in_ptr,
                      n,
                      c,
                      input_width,
                      input_height,
                      in_x + 2,
                      in_y - 1 + k),
                  t_x);
            }

            out_ptr
                [n * output_height * output_width * channels +
                 c * output_height * output_width + output_y * output_width +
                 output_x] =
                    static_cast<scalar_t>(cubic_interp1d(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                        t_y));
          }
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename scalar_t>
static void upsample_bicubic2d_backward_out_frame(
    scalar_t* odata,
    scalar_t* idata,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t nbatch,
    int64_t channels,
    int64_t onum,
    bool align_corners) {
  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t local_range = static_cast<int64_t>(1);
  int64_t global_range = static_cast<int64_t>(1);
  if (onum != 0) {
    int64_t wg_size = dpcppMaxWorkGroupSize();
    local_range = onum < wg_size ? onum : wg_size;
    global_range = ((onum + local_range - 1) / local_range) * local_range;
  }
  auto cgf = DPCPP_Q_CGF(cgh) {
    auto in_data = idata;
    auto out_data = odata;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto in_ptr = in_data;
      auto out_ptr = out_data;
      int global_id = item.get_global_linear_id();

      if (global_id < output_height * output_width) {
        const int output_x = global_id % output_width;
        const int output_y = global_id / output_width;
        // special case: output_xust copy
        if (input_height == output_height && input_width == output_width) {
          for (int n = 0; n < nbatch; n++) {
            for (int c = 0; c < channels; ++c) {
              auto val = out_ptr
                  [n * output_height * output_width * channels +
                   c * output_height * output_width + output_y * output_width +
                   output_x];
              in_ptr
                  [n * input_height * input_width * channels +
                   c * input_height * input_width + output_y * input_width +
                   output_x] = val;
            }
          }
          return;
        }

        const scalar_t height_scale = area_pixel_compute_scale<scalar_t>(
            input_height, output_height, align_corners);
        const scalar_t width_scale = area_pixel_compute_scale<scalar_t>(
            input_width, output_width, align_corners);

        scalar_t real_x = area_pixel_compute_source_index(
            width_scale, output_x, align_corners, /*cubic=*/true);
        int input_x = Numerics<scalar_t>::floor(real_x);
        scalar_t t_x = real_x - input_x;

        scalar_t real_y = area_pixel_compute_source_index(
            height_scale, output_y, align_corners, /*cubic=*/true);
        int input_y = Numerics<scalar_t>::floor(real_y);
        scalar_t t_y = real_y - input_y;

        scalar_t x_coeffs[4];
        scalar_t y_coeffs[4];

        get_cubic_upsample_coefficients(x_coeffs, t_x);
        get_cubic_upsample_coefficients(y_coeffs, t_y);

        for (int n = 0; n < nbatch; n++) {
          for (int c = 0; c < channels; ++c) {
            scalar_t out_value = out_ptr
                [n * output_height * output_width * channels +
                 c * output_height * output_width + output_y * output_width +
                 output_x];
            for (int i = 0; i < 4; i++) {
              for (int j = 0; j < 4; j++) {
                upsample_increment_value_bounded<scalar_t>(
                    in_ptr,
                    n,
                    c,
                    input_width,
                    input_height,
                    input_x - 1 + j,
                    input_y - 1 + i,
                    out_value * y_coeffs[i] * x_coeffs[j]);
              }
            }
          }
        }
      }
    };
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(global_range), sycl::range<1>(local_range)),
        kfn);
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

static void upsample_bicubic2d_out_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    bool align_corners) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_.size(0);
  int64_t channels = input_.size(1);
  int64_t input_height = input_.size(2);
  int64_t input_width = input_.size(3);

  upsample_2d_shape_check(
      input_,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  auto input = input_.contiguous();

  output.resize_({nbatch, channels, output_height, output_width});
  output.zero_();

  IPEX_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "upsample_bicubic2d", [&] {
        auto* idata = input.data_ptr<scalar_t>();
        auto* odata = output.data_ptr<scalar_t>();
        auto onum = output.numel();

        upsample_bicubic2d_out_frame<scalar_t>(
            odata,
            idata,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            onum,
            align_corners);
      });
}

static void upsample_bicubic2d_backward_out_template(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  upsample_2d_shape_check(
      Tensor(),
      grad_output_,
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  Tensor grad_output = grad_output_.contiguous();

  grad_input.resize_({nbatch, channels, input_height, input_width});
  grad_input.zero_();

  IPEX_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "upsample_bicubic2d_backward", [&] {
        scalar_t* idata = grad_input.data_ptr<scalar_t>();
        scalar_t* odata = grad_output.data_ptr<scalar_t>();
        auto onum = grad_output.numel();

        upsample_bicubic2d_backward_out_frame<scalar_t>(
            odata,
            idata,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            onum,
            align_corners);
      });
}
} // namespace impl

using at::native::upsample::compute_output_size;
using at::native::upsample::get_scale_value;

Tensor& upsample_bicubic2d_out(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& out) {
  impl::upsample_bicubic2d_out_template(out, self, output_size, align_corners);
  return out;
}

Tensor upsample_bicubic2d(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto output = at::empty({0}, self.options());
  impl::upsample_bicubic2d_out_template(
      output, self, output_size, align_corners);
  return output;
}

Tensor upsample_bicubic2d(
    const Tensor& input,
    c10::optional<IntArrayRef> output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto output = at::empty({0}, input.options());
  auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  impl::upsample_bicubic2d_out_template(output, input, osize, align_corners);
  return output;
}

Tensor& upsample_bicubic2d_backward_out(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  impl::upsample_bicubic2d_backward_out_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

Tensor upsample_bicubic2d_backward(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::zeros(input_size, grad_output.options());
  impl::upsample_bicubic2d_backward_out_template(
      grad_input, grad_output, output_size, input_size, align_corners);
  return grad_input;
}

Tensor upsample_bicubic2d_backward(
    const Tensor& grad_output,
    c10::optional<IntArrayRef> output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scale_factors) {
  auto osize = compute_output_size(input_size, output_size, scale_factors);
  auto scale_h = get_scale_value(scale_factors, 0);
  auto scale_w = get_scale_value(scale_factors, 1);
  auto grad_input = at::zeros(input_size, grad_output.options());
  impl::upsample_bicubic2d_backward_out_template(
      grad_input, grad_output, osize, input_size, align_corners);
  return grad_input;
}

} // namespace AtenIpexTypeXPU
} // namespace at
