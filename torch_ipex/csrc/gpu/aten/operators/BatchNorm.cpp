#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

#include <core/Memory.h>
#include <core/Runtime.h>
#include <utils/Math.h>

using namespace mkldnn;
using namespace at::dpcpp;

namespace at {
namespace AtenIpexTypeDPCPP {
namespace impl {

void get_dnnl_format(
    const Tensor& input,
    memory::format_tag& dnnl_format,
    memory::dims& input_tz) {
  auto input_sizes = input.sizes();
  auto input_ndim = input_sizes.size();

  if (input_ndim == 2) {
    dnnl_format = memory::format_tag::nc;
    input_tz = {input.size(0), input.size(1)};
  } else if (input_ndim == 3) {
    /* Map the rank3 batch norm to rank4 batch norm*/
    dnnl_format = memory::format_tag::nchw;
    input_tz = {
        /*n*/ input.size(0), /*c*/ input.size(1), /*h*/ input.size(2), /*w*/ 1};
  } else if (input_ndim == 4) {
    dnnl_format = memory::format_tag::nchw;
    input_tz = {/*n*/ input.size(0),
                /*c*/ input.size(1),
                /*h*/ input.size(2),
                /*w*/ input.size(3)};
  } else if (input_ndim == 5) {
    dnnl_format = memory::format_tag::ncdhw;
    input_tz = {/*n*/ input.size(0),
                /*c*/ input.size(1),
                /*d*/ input.size(2),
                /*h*/ input.size(3),
                /*w*/ input.size(4)};
  } else {
    std::stringstream ss;
    ss << "SYCL batch_norm backend got shape=" << input_sizes
       << ", expected input with rank 2 [n, c], rank 3 [n, c, l], rank 4 [n, "
          "c, h, w] or rank 5 [n, c, d, h, w] shape ";
    AT_ERROR(ss.str());
  }
}

Tensor condition_contiguous(const Tensor& t) {
  if (t.defined())
    return t.contiguous();
  return t;
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_template(
    const Tensor& input_,
    const Tensor& weight_,
    const Tensor& bias_,
    const Tensor& running_mean_ /* optional */,
    const Tensor& running_var_ /* optional */,
    bool training,
    double momentum,
    double epsilon) {
  Tensor input = condition_contiguous(input_);
  Tensor weight = condition_contiguous(weight_);
  Tensor bias = condition_contiguous(bias_);
  Tensor running_mean = condition_contiguous(running_mean_);
  Tensor running_var = condition_contiguous(running_var_);
  auto output = at::empty_like(input);

  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  auto propagation =
      training ? prop_kind::forward_training : prop_kind::forward_inference;
  normalization_flags flag = normalization_flags::use_scale_shift;

  if (!weight.defined() || !bias.defined()) {
    flag &= ~normalization_flags::use_scale_shift;
    //        if (mkldnn_use_scaleshift)
    //            flag |= normalization_flags::use_scale_shift;
    //        else
    //            TODO: Add warning
  }

  if (!training) {
    if (running_mean.defined() && running_var.defined())
      // Only inference and the running_xxx is defined.
      flag |= normalization_flags::use_global_stats;
    //    Some normalizatoins are re-dispatched to batch norm. Do not report
    //    error.
    //  else
    //    AT_ERROR("The running mean or running var is not defined in batch norm
    //    trainning");
  }

  int64_t feature_num = input.size(1);
  int64_t feature_size = input.numel() / feature_num;
  memory::format_tag dnnl_format;
  memory::dims input_tz;
  get_dnnl_format(input, dnnl_format, input_tz);

  auto data_t = dt_to_dnnl(input.type().scalarType());

  auto input_md = memory::desc({input_tz}, data_t, dnnl_format);

  batch_normalization_forward::desc batch_norm_forward_desc(
      propagation, input_md, epsilon, flag);
  auto batch_norm_forward_pd = batch_normalization_forward::primitive_desc(
      batch_norm_forward_desc, engine);

  auto input_usr_memory = memory({{{input_tz}, data_t, dnnl_format}, engine});
  dpcpp_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto output_usr_memory = memory({{{input_tz}, data_t, dnnl_format}, engine});
  dpcpp_set_mkldnn_buffer(output.data_ptr(), output_usr_memory);

  std::shared_ptr<mkldnn::primitive> bn_fwd;
  auto strm = GpuStreamManager::Instance().get_stream();

  bn_fwd.reset(new batch_normalization_forward(batch_norm_forward_pd));
  std::unordered_map<int, memory> args = {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_DST, output_usr_memory},
  };

  if ((bool)(flag & normalization_flags::use_scale_shift)) {
    auto weight_bias_memory =
        memory(batch_norm_forward_pd.weights_desc(), engine);
    auto weight_bias =
        at::empty(2 * feature_num, weight.options().dtype(ScalarType::Float));
    Tensor _weight = weight.to(ScalarType::Float);
    Tensor _bias = bias.to(ScalarType::Float);

    if (mkldnn_use_scaleshift) {
      dpcppMemcpyAsync(
          weight_bias.data_ptr(),
          _weight.data_ptr(),
          feature_num * sizeof(float),
          DeviceToDevice);
      dpcppMemcpyAsync(
          static_cast<uint8_t*>(weight_bias.data_ptr()) +
              feature_num * sizeof(float),
          _bias.data_ptr(),
          feature_num * sizeof(float),
          DeviceToDevice);
      dpcpp_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);
    }

    args.insert({MKLDNN_ARG_SCALE_SHIFT, weight_bias_memory});
  } else {
    // TODO: Add warning
  }

  Tensor save_mean = at::empty({feature_num}, input.options());
  Tensor save_var = at::empty({feature_num}, input.options());

  auto mean_memory = memory(batch_norm_forward_pd.mean_desc(), engine);
  auto var_memory = memory(batch_norm_forward_pd.variance_desc(), engine);

  if ((bool)(flag & normalization_flags::use_global_stats)) {
    dpcpp_set_mkldnn_buffer(running_mean.data_ptr(), mean_memory);
    dpcpp_set_mkldnn_buffer(running_var.data_ptr(), var_memory);
  } else {
    dpcpp_set_mkldnn_buffer(save_mean.data_ptr(), mean_memory);
    dpcpp_set_mkldnn_buffer(save_var.data_ptr(), var_memory);
  }

  args.insert({MKLDNN_ARG_MEAN, mean_memory});
  args.insert({MKLDNN_ARG_VARIANCE, var_memory});

  bn_fwd->execute(strm, args);

  if (training && running_mean.defined() && running_var.defined()) {
    dpcppMemoryScale1(
        running_mean.data_ptr(), save_mean.data_ptr(), feature_num, momentum);
    size_t orig_size = feature_size;
    size_t adjust_size = orig_size - 1;
    float adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
    dpcppMemoryScale2(
        running_var.data_ptr(),
        save_var.data_ptr(),
        feature_num,
        adjust_factor,
        momentum);
  }

  return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

} // namespace impl

std::tuple<Tensor, Tensor, Tensor> native_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double epsilon) {
  checkBackend(
      "batch_norm",
      {input, weight, bias, running_mean, running_var},
      Backend::DPCPP);

  if (input.scalar_type() != at::ScalarType::Float &&
      input.scalar_type() != at::ScalarType::Half) {
    // TODO: add more scalar type support.
    std::stringstream ss;
    ss << "DPCPP batch_norm backend got unsupported type="
       << input.scalar_type();
    TORCH_CHECK(0, ss.str());
  } else
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "batch_norm", [&] {
          return impl::batch_norm_template<scalar_t>(
              input,
              weight,
              bias,
              running_mean,
              running_var,
              training,
              momentum,
              epsilon);
        });
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward(
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& weight_,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const Tensor& running_mean_,
    const Tensor& running_var_,
    const Tensor& save_mean_,
    const Tensor& save_var_,
    bool training,
    double epsilon,
    std::array<bool, 3> grad_input_mask) {
  Tensor grad_output = impl::condition_contiguous(grad_output_);
  Tensor input = impl::condition_contiguous(input_);
  Tensor weight = impl::condition_contiguous(weight_);
  Tensor running_mean = impl::condition_contiguous(running_mean_);
  Tensor running_var = impl::condition_contiguous(running_var_);
  Tensor save_mean = impl::condition_contiguous(save_mean_);
  Tensor save_var = impl::condition_contiguous(save_var_);

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (grad_input_mask[0]) {
    grad_input = at::empty_like(input);
  }
  if (grad_input_mask[1]) {
    grad_weight = at::empty_like(weight);
  }
  if (grad_input_mask[2]) {
    grad_bias = at::empty_like(weight);
  }

  Device curDevice = Device(kDPCPP, current_device());
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);
  auto flags = normalization_flags::use_scale_shift; // backward only support
                                                     // training mode

  if (!(grad_input_mask[1] && grad_input_mask[2])) {
    flags &=
        ~normalization_flags::use_scale_shift; // No grad_weight and grad_bias
    //        if (mkldnn_use_scaleshift)
    //            flag |= normalization_flags::use_scale_shift;
    //        else
    //            TODO: Add warning
  }

  size_t feature_num = input.size(1);
  memory::format_tag dnnl_format;
  memory::dims input_tz;
  impl::get_dnnl_format(input, dnnl_format, input_tz);
  auto data_t = memory::data_type::f32;

  auto input_md = memory::desc({input_tz}, data_t, dnnl_format);
  auto grad_output_md = input_md;
  batch_normalization_forward::desc batch_norm_forward_desc(
      prop_kind::forward_training, input_md, epsilon, flags);
  auto batch_norm_forward_pd = batch_normalization_forward::primitive_desc(
      batch_norm_forward_desc, engine);

  prop_kind p_kind;

  //  The check is not nnecessarybecause the use_scale_shift defined as:
  // If not specified:
  //  - on backward propagation
  //    prop_kind == #dnnl::prop_kind::backward_data has the
  //    same behavior as prop_kind == #dnnl::prop_kind::backward

  // But the dnnl wrapper returns the output number without checking the flags:
  //  virtual int n_outputs() const override {
  //          return 1 + (desc_.prop_kind == prop_kind::backward);
  //  }
  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    p_kind = prop_kind::backward;
  } else {
    p_kind = prop_kind::backward_data;
  }

  auto bwd_desc = batch_normalization_backward::desc(
      p_kind, grad_output_md, input_md, epsilon, flags);
  auto bn_bwd_pd = batch_normalization_backward::primitive_desc(
      bwd_desc, engine, batch_norm_forward_pd);

  auto input_usr_memory = memory({{{input_tz}, data_t, dnnl_format}, engine});
  dpcpp_set_mkldnn_buffer(input.data_ptr(), input_usr_memory);

  auto grad_output_memory = memory({{{input_tz}, data_t, dnnl_format}, engine});
  dpcpp_set_mkldnn_buffer(grad_output.data_ptr(), grad_output_memory);

  auto mean_memory = memory(batch_norm_forward_pd.mean_desc(), engine);
  auto var_memory = memory(batch_norm_forward_pd.variance_desc(), engine);
  if (training) {
    dpcpp_set_mkldnn_buffer(save_mean.data_ptr(), mean_memory);
    dpcpp_set_mkldnn_buffer(save_var.data_ptr(), var_memory);
  } else {
    dpcpp_set_mkldnn_buffer(running_mean.data_ptr(), mean_memory);
    dpcpp_set_mkldnn_buffer(running_var.data_ptr(), var_memory);
  }

  auto grad_input_memory = memory({{{input_tz}, data_t, dnnl_format}, engine});
  dpcpp_set_mkldnn_buffer(grad_input.data_ptr(), grad_input_memory);

  auto strm = GpuStreamManager::Instance().get_stream();
  std::shared_ptr<mkldnn::primitive> bn_bwd;

  bn_bwd.reset(new batch_normalization_backward(bn_bwd_pd));

  std::unordered_map<int, memory> args = {
      {MKLDNN_ARG_SRC, input_usr_memory},
      {MKLDNN_ARG_DIFF_DST, grad_output_memory},
      {MKLDNN_ARG_MEAN, mean_memory},
      {MKLDNN_ARG_VARIANCE, var_memory},
      {MKLDNN_ARG_DIFF_SRC, grad_input_memory},
  };

  Tensor grad_weight_bias;

  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    auto weight_bias_memory =
        memory(batch_norm_forward_pd.weights_desc(), engine);
    auto weight_bias = at::empty(2 * feature_num, weight.options());
    dpcppMemcpyAsync(
        weight_bias.data_ptr(),
        weight.data_ptr(),
        feature_num * sizeof(float),
        DeviceToDevice);
    dpcppMemsetAsync(
        static_cast<uint8_t*>(weight_bias.data_ptr()) +
            feature_num * sizeof(float),
        0,
        feature_num * sizeof(float));
    dpcpp_set_mkldnn_buffer(weight_bias.data_ptr(), weight_bias_memory);

    auto grad_weight_bias_memory =
        memory(bn_bwd_pd.diff_weights_desc(), engine);
    grad_weight_bias = at::empty(2 * feature_num, weight.options());
    dpcpp_set_mkldnn_buffer(
        grad_weight_bias.data_ptr(), grad_weight_bias_memory);

    args.insert({MKLDNN_ARG_SCALE_SHIFT, weight_bias_memory});
    args.insert({MKLDNN_ARG_DIFF_SCALE_SHIFT, grad_weight_bias_memory});
  }

  bn_bwd->execute(strm, args);

  if ((bool)(flags & normalization_flags::use_scale_shift)) {
    dpcppMemcpyAsync(
        grad_weight.data_ptr(),
        grad_weight_bias.data_ptr(),
        feature_num * sizeof(float),
        DeviceToDevice);
    dpcppMemcpyAsync(
        grad_bias.data_ptr(),
        static_cast<uint8_t*>(grad_weight_bias.data_ptr()) +
            feature_num * sizeof(float),
        feature_num * sizeof(float),
        DeviceToDevice);
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

} // namespace AtenIpexTypeDPCPP
} // namespace at
