#pragma once

#include <oneDNN/oneDNN.h>

using namespace dnnl;
using namespace xpu::oneDNN;

namespace at {
namespace AtenIpexTypeXPU {

void inner_product(
    int M,
    int N,
    int K,
    at::Tensor& output,
    const at::Tensor& input_,
    const at::Tensor& weight,
    Tensor bias,
    bool use_bias) {
  Tensor input = input_.is_quantized() ? at::dequantize(input_) : input_;
  Device curDevice = Device(kXPU, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto strm = GpuStreamManager::Instance().get_stream();

  int32_t n = M;
  int32_t ic = K;
  int32_t oc = N;
  auto data_t = memory::data_type::f32;
  auto format_any = memory::format_tag::any;
  auto format_nc = memory::format_tag::nc;
  auto format_oi = memory::format_tag::oi;
  auto format_x = memory::format_tag::x;

  memory::dims input_tz = {n, ic};
  memory::dims weight_tz = {oc, ic};
  memory::dims bias_tz = {oc};
  memory::dims output_tz = {n, oc};

  auto input_md = memory::desc(input_tz, data_t, format_any);
  auto weight_md = memory::desc(weight_tz, data_t, format_any);
  auto output_md = memory::desc(output_tz, data_t, format_any);
  auto bias_md =
      use_bias ? memory::desc(bias_tz, data_t, format_any) : memory::desc();

  auto ip_forward_pd = inner_product_forward::primitive_desc(
      engine,
      prop_kind::forward_inference,
      input_md,
      weight_md,
      bias_md,
      output_md);

  auto input_memory = dpcpp_onednn_memory(
      {input_tz, data_t, format_nc}, engine, input.data_ptr());
  auto weight_memory = dpcpp_onednn_memory(
      {weight_tz, data_t, format_oi}, engine, weight.data_ptr());
  auto output_memory = dpcpp_onednn_memory(
      {output_tz, data_t, format_nc}, engine, output.data_ptr());

  memory bias_memory = memory({{}, data_t, format_x}, engine);
  if (use_bias) {
    bias_memory = dpcpp_onednn_memory(
        {bias_tz, data_t, format_x}, engine, bias.data_ptr());
  }

  auto ip_forward = inner_product_forward(ip_forward_pd);
  DPCPP_ONEDNN_EXEC(
      ip_forward,
      strm,
      {{DNNL_ARG_SRC, input_memory},
       {DNNL_ARG_WEIGHTS, weight_memory},
       {DNNL_ARG_BIAS, bias_memory},
       {DNNL_ARG_DST, output_memory}});
}

} // namespace AtenIpexTypeXPU
} // namespace at
