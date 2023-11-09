#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <core/Memory.h>
#include <core/MemoryFormat.h>
#include <runtime/Utils.h>
#include "comm/ATDispatch.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

template <typename scalar_t>
void max_unpooling2d_forward_kernel(
    const int64_t numInputElements,
    const scalar_t* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    scalar_t* output,
    const bool is_channels_last) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t num_groups = CeilDiv(numInputElements, group_size);
  int64_t total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto output_data = output;
    auto input_data = input;
    auto indices_data = indices;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto output_ptr = output_data;
      auto input_ptr = input_data;
      auto indices_ptr = indices_data;
      for (int linearIndex = item.get_global_id(0);
           linearIndex < numInputElements;
           linearIndex += item.get_global_range()[0]) {
        int c = is_channels_last
            ? linearIndex % numChannels
            : (linearIndex / inputWidth / inputHeight) % numChannels;
        int n = linearIndex / inputWidth / inputHeight / numChannels;
        int maxind = indices_ptr[linearIndex];
        int offset = is_channels_last
            ? n * numChannels * outputHeight * outputWidth + c
            : (n * numChannels + c) * outputHeight * outputWidth;
        output_ptr += offset;
        if (is_channels_last) {
          output_ptr[maxind * numChannels] = input_ptr[linearIndex];
        } else {
          output_ptr[maxind] = input_ptr[linearIndex];
        }
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
void max_unpooling3d_forward_kernel(
    scalar_t* input,
    int64_t* indices,
    scalar_t* output,
    const int64_t batchSize,
    const int64_t inputSlices,
    const int64_t iT,
    const int64_t iH,
    const int64_t iW,
    const int64_t oT,
    const int64_t oH,
    const int64_t oW,
    const int64_t offsetZ) {
  auto& queue = dpcppGetCurrentQueue();
  int64_t totalZ = batchSize * inputSlices * iT;
  int64_t num_groups_0 = CeilDiv(iW, (int64_t)32);
  int64_t num_groups_1 = CeilDiv(iH, (int64_t)8);

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto output_data = output;
    auto input_data = input;
    auto indices_data = indices;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<3> item) {
      auto output_ptr = output_data;
      auto input_ptr = input_data;
      auto indices_ptr = indices_data;

      int64_t iColumn = item.get_global_id(0);
      int64_t iRow = item.get_global_id(1);
      int64_t iFrame = (item.get_group()[2] + offsetZ) % iT; // input frame/time
      int64_t slice =
          (item.get_group()[2] + offsetZ) / iT; // input slice/feature
      if (iRow < iH && iColumn < iW) {
        scalar_t val = input_ptr
            [slice * iT * iH * iW + iFrame * iH * iW + iRow * iW +
             iColumn] /*[slice][iFrame][iRow][iColumn]*/;
        int64_t index = indices_ptr
            [slice * iT * iH * iW + iFrame * iH * iW + iRow * iW +
             iColumn] /*[slice][iFrame][iRow][iColumn]*/;
        output_ptr[slice * oT * oH * oW + index] = val;
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(32 * num_groups_0, 8 * num_groups_1, totalZ),
            sycl::range<3>(32, 8, 1)),
        kfn);
  };

  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t>
void max_unpooling3d_cl_forward_kernel(
    const int64_t numInputElements,
    const scalar_t* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputDepth,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputDepth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    scalar_t* output) {
  auto& queue = dpcppGetCurrentQueue();
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t group_size = dpcppMaxWorkGroupSize(dev_id);
  int64_t num_groups = CeilDiv(numInputElements, group_size);
  int64_t total_items = num_groups * group_size;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto output_data = output;
    auto input_data = input;
    auto indices_data = indices;
    auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {
      auto output_ptr = output_data;
      auto input_ptr = input_data;
      auto indices_ptr = indices_data;
      for (int linearIndex = item.get_global_id(0);
           linearIndex < numInputElements;
           linearIndex += item.get_global_range()[0]) {
        int c = linearIndex % numChannels;
        int n =
            linearIndex / inputDepth / inputWidth / inputHeight / numChannels;
        int maxind = indices_ptr[linearIndex];
        int offset =
            n * numChannels * outputDepth * outputHeight * outputWidth + c;
        output_ptr += offset;
        output_ptr[maxind * numChannels] = input_ptr[linearIndex];
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

Tensor& max_unpooling2d_forward_template(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size) {
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  auto oheight = output_size[0];
  auto owidth = output_size[1];

  TORCH_CHECK(self_.numel() > 0, "Input must be non-empty tensor");

  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor",
      self_.sizes());
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Shape of input must match shape of indices");
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (width, height) in output_size");

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t numBatch = 1;

  int64_t numChannels;
  int64_t inputHeight;
  int64_t inputWidth;

  auto fmt = is_smf_channels_last(self_)
      ? get_cl_tag_by_ndim(self_.ndimension())
      : at::MemoryFormat::Contiguous;
  auto self = self_.contiguous(fmt);
  auto indices = indices_.contiguous(fmt);

  if (self.ndimension() == 4) {
    numBatch = self.size(0);
    dimw++;
    dimh++;
  }
  numChannels = self.size(dimh - 1);
  inputHeight = self.size(dimh);
  inputWidth = self.size(dimw);

  output.resize_({numBatch, numChannels, oheight, owidth}, fmt);

  output.zero_();

  auto count = self.numel();
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "max_unpooling2d_forward_kernel",
      ([&] {
        max_unpooling2d_forward_kernel(
            self.numel(),
            self.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            numChannels,
            inputHeight,
            inputWidth,
            oheight,
            owidth,
            output.data_ptr<scalar_t>(),
            is_smf_channels_last(self_));
      }));

  if (self.ndimension() == 3) {
    output.resize_({numChannels, oheight, owidth});
  }
  return output;
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor",
      input.sizes());
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in "
      "output_size");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in "
      "stride");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in "
      "padding");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");

  TORCH_CHECK(input.numel() > 0, "Input must be non-empty");

  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);

  if (gradOutput.defined()) {
    if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
        oW != gradOutput.size(dimw)) {
      AT_ERROR(
          "Inconsistent gradOutput size. oT= ",
          oT,
          ", oH= ",
          oH,
          ", oW= ",
          oW,
          ". gradOutput: ",
          gradOutput.size(dimt),
          "x",
          gradOutput.size(dimh),
          "x",
          gradOutput.size(dimw));
    }
    TORCH_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
            gradOutput.size(dimn) == nslices,
        "gradOutput and input Tensors should have same number of dimensions "
        "and also the same number of channels/slices");
  }
}

Tensor& max_unpooling3d_forward_template(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  max_unpooling3d_shape_check(
      self_, Tensor(), indices_, output_size, stride, padding);

  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  auto fmt = is_smf_channels_last(self_)
      ? get_cl_tag_by_ndim(self_.ndimension())
      : at::MemoryFormat::Contiguous;
  auto self = self_.contiguous(fmt);
  auto indices = indices_.contiguous(fmt);

  int64_t batchSize;
  int64_t inputSlices;
  int64_t inputTime;
  int64_t inputHeight;
  int64_t inputWidth;

  if (self.ndimension() == 4) {
    batchSize = 1;
    inputSlices = self.size(0);
    inputTime = self.size(1);
    inputHeight = self.size(2);
    inputWidth = self.size(3);
    output.resize_({inputSlices, oT, oH, oW}, fmt);
  } else {
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
    output.resize_({batchSize, inputSlices, oT, oH, oW}, fmt);
  }

  output.zero_();

  if (is_smf_channels_last(self_)) {
    IPEX_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "max_unpooling3d_cl_forward_kernel",
        ([&] {
          max_unpooling3d_cl_forward_kernel(
              self.numel(),
              self.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              inputSlices,
              inputTime,
              inputHeight,
              inputWidth,
              oT,
              oH,
              oW,
              output.data_ptr<scalar_t>());
        }));

    return output;
  }

  // Collapse batch and feature dimensions if needed
  if (self.ndimension() == 5) {
    self = self.reshape(
        {self.size(0) * self.size(1),
         self.size(2),
         self.size(3),
         self.size(4)});
    indices = indices.reshape(
        {indices.size(0) * indices.size(1),
         indices.size(2),
         indices.size(3),
         indices.size(4)});
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  IPEX_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "max_unpooling3d_forward_kernel",
      ([&] {
        while (totalZ > 0) {
          max_unpooling3d_forward_kernel(
              self.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              batchSize,
              inputSlices,
              inputTime,
              inputHeight,
              inputWidth,
              oT,
              oH,
              oW,
              offsetZ);
          totalZ -= 65535;
          offsetZ += 65535;
        }
      }));
  return output;
}
} // namespace impl

Tensor& max_unpool2d_out(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    Tensor& out) {
  impl::max_unpooling2d_forward_template(out, self, indices, output_size);
  return out;
}

Tensor max_unpool2d(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto out = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::max_unpool2d_out(self, indices, output_size, out);
  return out;
}

Tensor& max_unpool3d_out(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& out) {
  impl::max_unpooling3d_forward_template(
      out, self, indices, output_size, stride, padding);
  return out;
}

Tensor max_unpool3d(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto out = at::empty({0}, self.options());
  at::AtenIpexTypeXPU::max_unpool3d_out(
      self, indices, output_size, stride, padding, out);
  return out;
}

} // namespace AtenIpexTypeXPU
} // namespace at
