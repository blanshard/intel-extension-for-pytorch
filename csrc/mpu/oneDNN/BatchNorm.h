#pragma once

#include <ATen/ATen.h>

#include <ATen/core/TensorAccessor.h>
#include <ATen/record_function.h>
#include <core/MemoryFormat.h>
#include <oneDNN/Runtime.h>
#include <operators/Utils.h>
#include <operators/comm/ATDispatch.h>
#include <quantized/Quantizer.h>
#include <runtime/Utils.h>
#include <tensor/Tensor.h>
#include <utils/LRUCache.h>
#include "Reorder.h"
#include "Utils.h"

#include <oneapi/dnnl/dnnl.hpp>

using namespace dnnl;
using namespace xpu::dpcpp;
using namespace at::AtenIpexTypeXPU;

namespace xpu {
namespace oneDNN {

static inline memory::format_tag bn_src_format(const at::Tensor& t) {
  auto is_channels_last = using_channels_last_for_onednn_op(t);
  auto ndim = t.ndimension();
  if (ndim == 2) {
    return memory::format_tag::nc;
  } else if (ndim == 3) {
    return is_channels_last ? memory::format_tag::nwc : memory::format_tag::ncw;
  } else if (ndim == 4) {
    return is_channels_last ? memory::format_tag::nhwc
                            : memory::format_tag::nchw;
  } else if (ndim == 5) {
    return is_channels_last ? memory::format_tag::ndhwc
                            : memory::format_tag::ncdhw;
  } else {
    std::stringstream ss;
    ss << "SYCL batch_norm backend got shape=" << t.sizes()
       << ", expected input with rank 2 [n, c], rank 3 [n, c, l], rank 4 [n, "
          "c, h, w] or rank 5 [n, c, d, h, w] shape ";
    AT_ERROR(ss.str());
  }
}

static std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> batch_normalization(
    const at::Tensor& src,
    const at::Tensor& wgh_option,
    const at::Tensor& bia_option,
    const at::Tensor& running_mean_option,
    const at::Tensor& running_var_option,
    bool training,
    double momentum,
    double epsilon,
    at::Tensor& dst,
    at::Tensor& save_mean,
    at::Tensor& save_var) {
  auto engine =
      GpuEngineManager::Instance().get_engine({at::kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  at::Tensor wgh = wgh_option;
  at::Tensor bia = bia_option;
  at::Tensor running_mean = running_mean_option;
  at::Tensor running_var = running_var_option;

  auto prop =
      training ? prop_kind::forward_training : prop_kind::forward_inference;
  auto flag = normalization_flags::use_scale | normalization_flags::use_shift;

  auto feature_num = src.size(1);
  auto feature_size = src.numel() / feature_num;

  if (!wgh.defined())
    wgh = at::ones(feature_num, wgh.options());

  if (!bia.defined())
    bia = at::zeros(feature_num, wgh.options());

  if (!training && running_mean.defined() && running_var.defined())
    flag |= normalization_flags::use_global_stats;

  auto src_tz = get_onednn_dims(src);
  auto src_dt = get_onednn_dtype(src);
  auto src_fmt = bn_src_format(src);

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_md = src_ctx.is_plain() ? memory::desc({src_tz}, src_dt, src_fmt)
                                   : src_ctx.meta();
  auto scl_md = memory::desc(
      {feature_num}, memory::data_type::f32, memory::format_tag::a);
  auto sft_md = memory::desc(
      {feature_num}, memory::data_type::f32, memory::format_tag::a);

  primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  auto bn_fwd_pd = batch_normalization_forward::primitive_desc(
      engine, prop, src_md, src_md, epsilon, flag, pattr);

  auto ndim = src.ndimension();
  auto src_cl_mfmt = at::MemoryFormat::ChannelsLast;
  if (3 == ndim || 4 == ndim || 5 == ndim) {
    src_cl_mfmt = get_cl_tag_by_ndim(ndim);
  }
  auto dst_md = bn_fwd_pd.dst_desc();
  if (!dst.defined()) {
    if (!src_ctx.is_plain()) {
      auto dst_fmt = using_channels_last_for_onednn_op(src)
          ? src_cl_mfmt
          : at::MemoryFormat::Contiguous;
      dst = empty_opaque_tensor(dst_md, src.options(), dst_fmt);
    } else {
      dst = using_channels_last_for_onednn_op(src)
          ? xpu::dpcpp::empty_like_dpcpp(src, src.options(), src_cl_mfmt)
          : at::empty_like(src);
    }
  }

  auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());
  auto dst_m = dpcpp_onednn_memory(dst_md, engine, dst.data_ptr());

  auto bn_fwd = batch_normalization_forward(bn_fwd_pd);

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DST, dst_m},
  };

  if (wgh.scalar_type() == ScalarType::Half ||
      wgh.scalar_type() == ScalarType::BFloat16) {
    wgh = wgh.to(at::kFloat);
  }

  if (bia.scalar_type() == ScalarType::Half ||
      bia.scalar_type() == ScalarType::BFloat16) {
    bia = bia.to(at::kFloat);
  }

  auto scl_m = dpcpp_onednn_memory(scl_md, engine, wgh.data_ptr());
  auto sft_m = dpcpp_onednn_memory(sft_md, engine, bia.data_ptr());
  args.insert({DNNL_ARG_SCALE, scl_m});
  args.insert({DNNL_ARG_SHIFT, sft_m});

  if (!save_mean.defined()) {
    save_mean = at::empty(feature_num, wgh.options().dtype(at::kFloat));
  }
  if (!save_var.defined()) {
    save_var = at::empty(feature_num, wgh.options().dtype(at::kFloat));
  }

  void* mean_data = nullptr;
  void* var_data = nullptr;
  if ((bool)(flag & normalization_flags::use_global_stats)) {
    if (running_mean.scalar_type() == ScalarType::Half ||
        running_mean.scalar_type() == ScalarType::BFloat16)
      running_mean = running_mean.to(ScalarType::Float);

    if (running_var.scalar_type() == ScalarType::Half ||
        running_var.scalar_type() == ScalarType::BFloat16)
      running_var = running_var.to(ScalarType::Float);

    mean_data = running_mean.data_ptr();
    var_data = running_var.data_ptr();
  } else {
    mean_data = save_mean.data_ptr();
    var_data = save_var.data_ptr();
  }

  auto mean_m = dpcpp_onednn_memory(bn_fwd_pd.mean_desc(), engine, mean_data);
  auto var_m = dpcpp_onednn_memory(bn_fwd_pd.variance_desc(), engine, var_data);

  args.insert({DNNL_ARG_MEAN, mean_m});
  args.insert({DNNL_ARG_VARIANCE, var_m});

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = bn_fwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad_tensor = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_memory = dpcpp_onednn_memory(
      bn_fwd_pd.scratchpad_desc(), engine, scratchpad_tensor.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_memory});
#endif

  DPCPP_ONEDNN_EXEC(bn_fwd, strm, args);

  return {dst, save_mean, save_var};
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_normalization_backward(
    const at::Tensor& diff_dst,
    const at::Tensor& src,
    const at::Tensor& wgh,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    bool training,
    double epsilon,
    std::array<bool, 3> diff_src_mask) {
  auto engine =
      GpuEngineManager::Instance().get_engine({at::kXPU, current_device()});
  auto strm = GpuStreamManager::Instance().get_stream();

  at::Tensor diff_src, diff_wgh, diff_bia;
  if (diff_src_mask[0])
    diff_src = at::empty_like(src);
  if (diff_src_mask[1])
    diff_wgh = at::empty(wgh.sizes(), wgh.options().dtype(ScalarType::Float));
  if (diff_src_mask[2])
    diff_bia = at::empty(wgh.sizes(), wgh.options().dtype(ScalarType::Float));

  auto flags = normalization_flags::use_scale | normalization_flags::use_shift;

  if (!(diff_src_mask[1] && diff_src_mask[2])) {
    flags &= ~normalization_flags::use_scale;
    flags &= ~normalization_flags::use_shift;
  }

  auto src_tz = get_onednn_dims(src);
  auto src_dt = get_onednn_dtype(src);
  auto src_fmt = bn_src_format(src);

  auto src_ctx = DPCPPTensorContext::get_tensor_ctx(src);
  auto src_md = src_ctx.is_plain() ? memory::desc({src_tz}, src_dt, src_fmt)
                                   : src_ctx.meta();
  auto src_m = dpcpp_onednn_memory(src_md, engine, src.data_ptr());

  auto diff_dst_ctx = DPCPPTensorContext::get_tensor_ctx(diff_dst);
  auto diff_dst_md = diff_dst_ctx.is_plain()
      ? memory::desc({src_tz}, src_dt, src_fmt)
      : diff_dst_ctx.meta();
  auto diff_dst_usr_m =
      dpcpp_onednn_memory(diff_dst_md, engine, diff_dst.data_ptr());

  primitive_attr pattr;
#ifdef USE_SCRATCHPAD_MODE
  pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
#endif
  auto bn_fwd_pd = batch_normalization_forward::primitive_desc(
      engine,
      prop_kind::forward_training,
      src_md,
      src_md,
      epsilon,
      flags,
      pattr);

  at::Tensor diff_dst_;
  auto diff_dst_m = diff_dst_usr_m;
  if (diff_dst_ctx.is_plain() && (!src_ctx.is_plain())) {
    auto expected_dst_md = bn_fwd_pd.dst_desc();
    diff_dst_ =
        empty_opaque_tensor(expected_dst_md, src.options(), c10::nullopt);
    diff_dst_m =
        dpcpp_onednn_memory(expected_dst_md, engine, diff_dst_.data_ptr());
    diff_dst_md = expected_dst_md;
    xpu::oneDNN::reorder(diff_dst, diff_dst_);
  }

  prop_kind p_kind;
  if ((bool)(flags & normalization_flags::use_scale)) {
    p_kind = prop_kind::backward;
  } else {
    p_kind = prop_kind::backward_data;
  }

  auto bn_bwd_pd = batch_normalization_backward::primitive_desc(
      engine,
      p_kind,
      diff_dst_md,
      diff_dst_md,
      src_md,
      epsilon,
      flags,
      bn_fwd_pd,
      pattr);

  memory mean_m, var_m;
  if (training) {
    mean_m = dpcpp_onednn_memory(
        bn_fwd_pd.mean_desc(), engine, save_mean.data_ptr());
    var_m = dpcpp_onednn_memory(
        bn_fwd_pd.variance_desc(), engine, save_var.data_ptr());
  } else {
    mean_m = dpcpp_onednn_memory(
        bn_fwd_pd.mean_desc(), engine, running_mean.data_ptr());
    var_m = dpcpp_onednn_memory(
        bn_fwd_pd.variance_desc(), engine, running_var.data_ptr());
  }

  auto diff_src_md = memory::desc({src_tz, src_dt, src_fmt});
  auto expected_diff_src_md = bn_bwd_pd.diff_src_desc();
  if (diff_src_md != expected_diff_src_md) {
    diff_src = empty_opaque_tensor(
        expected_diff_src_md, diff_dst.options(), c10::nullopt);
  }
  auto diff_src_m =
      dpcpp_onednn_memory(expected_diff_src_md, engine, diff_src.data_ptr());

  auto bn_bwd = dnnl::batch_normalization_backward(bn_bwd_pd);

  std::unordered_map<int, memory> args = {
      {DNNL_ARG_SRC, src_m},
      {DNNL_ARG_DIFF_DST, diff_dst_m},
      {DNNL_ARG_MEAN, mean_m},
      {DNNL_ARG_VARIANCE, var_m},
      {DNNL_ARG_DIFF_SRC, diff_src_m},
  };

#ifdef USE_SCRATCHPAD_MODE
  size_t scratchpad_size = bn_bwd_pd.scratchpad_desc().get_size();
  Tensor scratchpad = at::AtenIpexTypeXPU::empty(
      {scratchpad_size}, src.options().dtype(at::kByte), c10::nullopt);
  auto scratchpad_m = dpcpp_onednn_memory(
      bn_bwd_pd.scratchpad_desc(), engine, scratchpad.data_ptr());
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad_m});
#endif

  size_t feature_num = src.size(1);

  at::Tensor wgh_f32, bia_f32;
  if ((bool)(flags & normalization_flags::use_scale)) {
    wgh_f32 = wgh.to(ScalarType::Float);
    auto wgh_m = dpcpp_onednn_memory(
        bn_bwd_pd.weights_desc(), engine, wgh_f32.data_ptr());

    if (!diff_wgh.has_storage())
      diff_wgh = at::empty_like(wgh_f32);
    auto diff_wgh_m = dpcpp_onednn_memory(
        bn_bwd_pd.diff_weights_desc(), engine, diff_wgh.data_ptr());

    if (!diff_bia.has_storage())
      diff_bia = at::empty_like(wgh_f32);
    auto diff_bia_m = dpcpp_onednn_memory(
        bn_bwd_pd.diff_weights_desc(), engine, diff_bia.data_ptr());

    args.insert({DNNL_ARG_SCALE, wgh_m});
    args.insert({DNNL_ARG_DIFF_SCALE, diff_wgh_m});
    args.insert({DNNL_ARG_DIFF_SHIFT, diff_bia_m});
    DPCPP_ONEDNN_EXEC(bn_bwd, strm, args);
  } else {
    DPCPP_ONEDNN_EXEC(bn_bwd, strm, args);
  }

  return {diff_src, diff_wgh, diff_bia};
}

} // namespace oneDNN
} // namespace xpu
