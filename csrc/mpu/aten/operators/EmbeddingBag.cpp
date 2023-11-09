#include <ATen/ATen.h>
#include <torch/torch.h>

#include <core/Device.h>
#include <core/Memory.h>
#include <runtime/Utils.h>
#include <utils/DPCPP.h>

#include "BitonicMergeSort.h"
#include "MemoryAccess.h"
#include "PSTLFunctions.h"
#include "comm/ATDispatch.h"
#include "comm/AccumulateType.h"
#include "comm/Atomics.h"
#include "comm/Numerics.h"

#include <aten/operators/MemoryAccess.h>
#include "EmbeddingBackwardKernel.h"
#include "EmbeddingBagKernel.h"

using namespace xpu::dpcpp;
using namespace xpu::dpcpp::detail;

namespace at {
namespace AtenIpexTypeXPU {
namespace impl {

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

template <
    int vec_size,
    typename vec_t,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
void vec_chunk_kernel_embeddingbag(
    const int64_t mode,
    index_t* input,
    index_t* offset,
    scalar_t* weight,
    scalar_t* output,
    index_t* offset2bag,
    index_t* bag_size,
    bool per_sample_weights_defined,
    scalar_t* per_sample_weights,
    int64_t per_sample_weights_stride,
    index_t* max_indices,
    int64_t WGNumber,
    int64_t numBags,
    int64_t weight_total_elem,
    int64_t chunk_size,
    int64_t bag_chunk_num,
    int64_t bag_wi_num,
    int64_t bagsPerLoop,
    int64_t input_length,
    int64_t weight_stride0,
    int64_t weight_stride1,
    const bool include_last_offset,
    const index_t padding_idx,
    const bool if_align_vector,
    sycl::nd_item<1> item) {
  auto globalId = item.get_global_linear_id();

  // global chunk id
  auto globalChunkId = globalId / chunk_size;

  // which initial bag this work item is in
  auto bagId = globalChunkId / bag_chunk_num;

  // work item id inside one bag
  auto insideBagId = globalId % bag_wi_num;

  constexpr int align_bytes = alignof(vec_t);

  // outer bag loop
  for (auto bag = bagId; bag < numBags; bag += bagsPerLoop) {
    auto begin = offset[bag];

    // TODO: Here need a check for begin and end that end must >= begin.
    auto end = (bag < (numBags - 1))
        ? (offset[bag + 1])
        : (include_last_offset ? offset[bag + 1] : input_length);

    // for mean mode's backward
    index_t bag_size_ = 0;

    // In single_bag situation, embeddingbag is like embedding, no
    // per_sample_weight, mode is not max and not padding entry and 2D weight,
    // pure vec copy is used to achieve most memory bandwidth.
    auto single_bag = bool(
        (end == (begin + 1)) && (!per_sample_weights_defined) &&
        (mode != MODE_MAX) && (input[begin] != padding_idx));

    if (single_bag) {
      auto input_single_elem = input[begin];

      // for checking alignment with vector
      auto shift = ((uint64_t)(weight + input_single_elem * weight_stride0)) %
          align_bytes / sizeof(scalar_t);

      // here the shift elements need to be individually dealed with
      for (auto mis_idx = 0; mis_idx < shift; ++mis_idx) {
        if (insideBagId == 0) {
          if (mis_idx < weight_stride0) {
            output[bag * weight_stride0 + mis_idx] = weight
                [input_single_elem * weight_stride0 + mis_idx * weight_stride1];
          }
        }
      }

      if (((shift + input_single_elem * weight_stride0) < weight_total_elem) &&
          (shift < weight_stride0)) {
        vec_t* weight_vec = reinterpret_cast<vec_t*>(
            shift + weight + input_single_elem * weight_stride0);
        // vector load
        auto weightSingleValue = weight_vec[insideBagId];
        vec_t* output_vec =
            reinterpret_cast<vec_t*>(shift + output + bag * weight_stride0);
#pragma unroll
        for (auto id = 0; id < vec_size; id++) {
          if ((shift + insideBagId * vec_size + id) < weight_stride0) {
            output_vec[insideBagId][id] =
                weightSingleValue[id * weight_stride1];
          }
        }
      }

      if (insideBagId == 0) {
        offset2bag[begin] = bag;
        bag_size[bag] = static_cast<index_t>(1);
      }
    } else {
      // not single bag mode
      index_t maxWord[vec_size];
      accscalar_t weightFeatSum[vec_size];
      scalar_t weightFeatMax[vec_size];

#pragma unroll
      for (auto id = 0; id < vec_size; id++) {
        maxWord[id] = -1;
        weightFeatSum[id] = static_cast<accscalar_t>(0.0);
        weightFeatMax[id] = static_cast<scalar_t>(0.0);
      }

      // alignment with vector load
      if (if_align_vector) {
        for (auto emb = begin; emb < end; emb++) {
          auto input_elem = input[emb];

          // if this bag copes with multi embeddings and one of these embeddings
          // is padding_idx, this embedding is ignored for reduction because
          // embedding vector at padding_idx is excluded from the reduction
          bool pad = (input_elem == padding_idx);

          // vector process remaining
          vec_t* weight_vec =
              reinterpret_cast<vec_t*>(weight + input_elem * weight_stride0);
          auto weightValue = weight_vec[insideBagId];

#pragma unroll
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_stride0) {
              if (mode == MODE_MAX) {
                // static_cast to scalar_t is used because vec_t contains
                // uint dtype
                auto val = weightValue[id];
                auto max_val = weightFeatMax[id];
                // bag_size_ == 0 means it first come
                if (bag_size_ == 0 || val > max_val) {
                  // padded entry will not be included output
                  weightFeatMax[id] = pad ? weightFeatMax[id] : weightValue[id];
                  maxWord[id] = pad ? maxWord[id] : input_elem;
                }
              } else {
                // 1. for scalar type fma/add, accscalar_t is needed to keep
                // accurate. Vec is stored uint value, whose size is same
                // as sizeof(scalar_t), when computing, uint value should
                // be casted to floating value, after computation,
                // write-back needs casting to uint value.
                // 2. if this entry is padded, 0 value is prepared for
                // reduce(sum/mean)
                auto val = pad ? static_cast<scalar_t>(0.0) : weightValue[id];
                auto acc_val = static_cast<accscalar_t>(val);
                auto acc_sum = weightFeatSum[id];
                if (per_sample_weights_defined) {
                  auto scaleWeightBy = static_cast<accscalar_t>(
                      per_sample_weights[emb * per_sample_weights_stride]);
                  acc_sum += acc_val * scaleWeightBy;
                } else {
                  acc_sum += acc_val;
                }
                weightFeatSum[id] = acc_sum;
              }
            }
          }

          // if this entry is padded, it will not contribute to bag size
          bag_size_ += pad ? 0 : 1;

          // avoid compete write in and padded entry also needs to be recorded
          // to offset2bag
          if (insideBagId == 0) {
            offset2bag[emb] = bag;
          }
        }
      } else {
        // exist misalignment, back to single point processing
        for (auto emb = begin; emb < end; emb++) {
          auto input_elem = input[emb];
          // if this bag copes with multi embeddings and one of these embeddings
          // is padding_idx, this embedding is ignored for reduction because
          // embedding vector at padding_idx is excluded from the reduction
          bool pad = (input_elem == padding_idx);

#pragma unroll
          for (auto id = 0; id < vec_size; id++) {
            if ((insideBagId * vec_size + id) < weight_stride0) {
              auto weight_idx = input_elem * weight_stride0 +
                  insideBagId * vec_size + id * weight_stride1;
              if (mode == MODE_MAX) {
                // static_cast to scalar_t is used because vec_t contains
                // uint dtype
                auto val = weight[weight_idx];
                auto max_val = weightFeatMax[id];
                // bag_size_ == 0 means it first come
                if (bag_size_ == 0 || val > max_val) {
                  // padded entry will not be included output
                  weightFeatMax[id] = pad ? weightFeatMax[id] : val;
                  maxWord[id] = pad ? maxWord[id] : input_elem;
                }
              } else {
                // 1. for scalar type fma/add, accscalar_t is needed to keep
                // accurate. Vec is stored uint value, whose size is same
                // as sizeof(scalar_t), when computing, uint value should
                // be casted to floating value, after computation,
                // write-back needs casting to uint value.
                // 2. if this entry is padded, 0 value is prepared for
                // reduce(sum/mean)
                auto val =
                    pad ? static_cast<scalar_t>(0.0) : weight[weight_idx];
                auto acc_val = static_cast<accscalar_t>(val);
                if (per_sample_weights_defined) {
                  auto scaleWeightBy = static_cast<accscalar_t>(
                      per_sample_weights[emb * per_sample_weights_stride]);
                  weightFeatSum[id] += acc_val * scaleWeightBy;
                } else {
                  weightFeatSum[id] += acc_val;
                }
              }
            }
          }

          // if this entry is padded, it will not contribute to bag size
          bag_size_ += pad ? 0 : 1;

          // avoid compete write in and padded entry also needs to be recorded
          // to offset2bag
          if (insideBagId == 0) {
            offset2bag[emb] = bag;
          }
        }
      }

      // calculate average for mean mode
      if (mode == MODE_MEAN) {
#pragma unroll
        for (auto id = 0; id < vec_size; id++) {
          if ((insideBagId * vec_size + id) < weight_stride0) {
            auto acc_sum = weightFeatSum[id];
            if (bag_size_ != 0) {
              acc_sum /= static_cast<accscalar_t>(bag_size_);
            }
            weightFeatSum[id] = acc_sum;
          }
        }
      }

      // output
#pragma unroll
      for (auto id = 0; id < vec_size; id++) {
        if ((insideBagId * vec_size + id) < weight_stride0) {
          auto output_idx = bag * weight_stride0 + insideBagId * vec_size +
              id * weight_stride1;
          if (mode == MODE_MEAN || mode == MODE_SUM) {
            output[output_idx] = static_cast<scalar_t>(weightFeatSum[id]);
          } else if (mode == MODE_MAX) {
            output[output_idx] = weightFeatMax[id];
            max_indices[output_idx] = maxWord[id];
          }
        }
      }

      if (insideBagId == 0) {
        bag_size[bag] = static_cast<index_t>(bag_size_);
      }
    }
  }
}

/*
  The kernel EmbeddingBag is optimized for memory coleascing and thread
  efficiency. Vec design and chunk design are deployed for this kernel. In
  additional, single bag is specifically considered.(for example, when
  offset_data=0,1,2,3,4,5,...).
  Thought:
  0. Principle: One or multi chunks work for one Bag. One loop at least solves
  one bag.
  1. Implementation: Use vec<scalar_t, vec_size> to achieve higher bandwidth
  both in ATS and PVC, because it is a memory bound kernel. Use chunk design,
  chunk splitted from different WG to reach high occupancy especially when bag
  dim is much larger. The vec size is determined by device. The chunk size is
  determined by workload amounts and device resource.
  2. If it is single bag specific situation, pure copy is done for kernel.
  Single bag means offset is linear increase by 1.
  3. Passing vec size as template to kernel.

  Shortcoming:
  1. Chunk design may cause some resource waste when work items is handling
  the tail of last bag in one loop.
*/
template <typename scalar_t, typename index_t>
void EmbeddingBag_updateOutputKernel(
    const int64_t mode,
    index_t* input_data,
    index_t* offset_data,
    scalar_t* weight_data,
    scalar_t* output_data,
    index_t* offset2bag_data,
    int64_t weight_total_elem,
    int64_t input_length,
    int64_t numBags,
    int64_t weight_stride0,
    int64_t weight_stride1,
    index_t* bag_size_data,
    index_t* max_indices_data,
    scalar_t* per_sample_weights_data,
    int64_t per_sample_weights_stride,
    const bool include_last_offset,
    const index_t padding_idx,
    const bool ignore_offsets) {
  using accscalar_t = acc_type<scalar_t>;

  // vector size, query it according to machine, scalar_t and weight_data
  auto& queue = dpcppGetCurrentQueue();
  auto vec_size = at::native::Memory::can_vectorize_up_to<scalar_t>(
      dpcppGetDeviceIdOfCurrentQueue(), reinterpret_cast<char*>(weight_data));

  // determine per sample weights should be in calculation or not
  bool per_sample_weights_defined = per_sample_weights_data ? true : false;

  auto maxWGSize = dpcppMaxWorkGroupSize();

  auto gpuEuCount = dpcppGpuEuCount();

  // how many work items serve for one bag in vector sight
  auto bag_wi_num = (weight_stride0 % vec_size == 0)
      ? (weight_stride0 / vec_size)
      : (weight_stride0 / vec_size + 1);

  auto chunk_size = 32;

  // how many chunks serve for one bag
  auto bag_chunk_num = (bag_wi_num % chunk_size == 0)
      ? (bag_wi_num / chunk_size)
      : (bag_wi_num / chunk_size + 1);

  // how many work items serve for one bag in chunk sight
  bag_wi_num = bag_chunk_num * chunk_size;

  // how many chunks serve for all bag
  auto all_chunk_num = numBags * bag_chunk_num;

  // how many wi serve for all bag
  auto all_wi_num = all_chunk_num * chunk_size;

  // For huge bags number, limited wg number is set to avoid overhead of
  // groups over scheduling. WGNumber default in single tile in one time =
  // Max compute unit * 8 threads * SIMD32 per thread / max WG size * 512.
  auto WGNumber = gpuEuCount * 8 * 32 / maxWGSize * 512;

  // one or multi chunks for one bag.
  // all_wi_num <= maxWGSize: one wg is enough to finish all bags
  // bag_wi_num > (maxWGSize * WGNumber): all wg is not enough to finish one
  // bag. To avoid the inner-bag loop, all needed wg are launched
  // else: one wg is not enough to finish all bags, but all wg can finish at
  // least one bag
  auto local_range = maxWGSize;
  if (all_wi_num <= maxWGSize) {
    local_range = all_wi_num;
    WGNumber = 1;
  } else if (bag_wi_num > (maxWGSize * WGNumber)) {
    local_range = maxWGSize;
    // at least, one loop finish one bag
    WGNumber = (bag_wi_num + maxWGSize - 1) / maxWGSize;
  } else {
    for (auto factor = 0; (((maxWGSize - factor * 8) >= 8)); ++factor) {
      auto infactor = maxWGSize - factor * 8;
      if (all_wi_num % infactor == 0) {
        if ((all_wi_num / infactor) > WGNumber) {
          local_range = infactor;
        } else {
          WGNumber = all_wi_num / infactor;
          local_range = infactor;
        }
        break;
      }
    }
  }

  // for outer bag loop, how many bag finish in one loop
  auto bagsPerLoop = WGNumber * local_range / chunk_size / bag_chunk_num;

  // total work item size
  auto global_range = WGNumber * local_range;

  bool if_align_vector = ((weight_stride0 % 2 == 0) || (sizeof(scalar_t) != 2));

// launch vec kernel for embeddingbag, code pass according to vec size
#define VEC_EMBBAG_KERNEL(vec_size)                                           \
  {                                                                           \
    auto cgf = DPCPP_Q_CGF(cgh) {                                             \
      auto input = input_data;                                                \
      auto offset = offset_data;                                              \
      auto weight = weight_data;                                              \
      auto output = output_data;                                              \
      auto offset2bag = offset2bag_data;                                      \
      auto bag_size = bag_size_data;                                          \
      auto per_sample_weights =                                               \
          per_sample_weights_defined ? per_sample_weights_data : weight_data; \
      auto max_indices = mode == MODE_MAX ? max_indices_data : nullptr;       \
      using vec_t =                                                           \
          at::native::Memory::aligned_vector_loop<scalar_t, vec_size>;        \
      auto kfn = DPCPP_Q_KFN(sycl::nd_item<1> item) {                         \
        vec_chunk_kernel_embeddingbag<                                        \
            vec_size,                                                         \
            vec_t,                                                            \
            scalar_t,                                                         \
            accscalar_t,                                                      \
            index_t>(                                                         \
            mode,                                                             \
            input,                                                            \
            offset,                                                           \
            weight,                                                           \
            output,                                                           \
            offset2bag,                                                       \
            bag_size,                                                         \
            per_sample_weights_defined,                                       \
            per_sample_weights,                                               \
            per_sample_weights_stride,                                        \
            max_indices,                                                      \
            WGNumber,                                                         \
            numBags,                                                          \
            weight_total_elem,                                                \
            chunk_size,                                                       \
            bag_chunk_num,                                                    \
            bag_wi_num,                                                       \
            bagsPerLoop,                                                      \
            input_length,                                                     \
            weight_stride0,                                                   \
            weight_stride1,                                                   \
            include_last_offset,                                              \
            padding_idx,                                                      \
            if_align_vector,                                                  \
            item);                                                            \
      };                                                                      \
      cgh.parallel_for(                                                       \
          sycl::nd_range<1>(                                                  \
              sycl::range<1>(global_range), sycl::range<1>(local_range)),     \
          kfn);                                                               \
    };                                                                        \
    DPCPP_Q_SUBMIT(queue, cgf);                                               \
  }

  switch (vec_size) {
    case 16: {
      VEC_EMBBAG_KERNEL(16);
      break;
    }
    case 8: {
      VEC_EMBBAG_KERNEL(8);
      break;
    }
    case 4: {
      VEC_EMBBAG_KERNEL(4);
      break;
    }
    case 2: {
      VEC_EMBBAG_KERNEL(2);
      break;
    }
    case 1: {
      VEC_EMBBAG_KERNEL(1);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for EmbeddingBag. vec size ",
          vec_size);
  }
#undef VEC_EMBBAG_KERNEL
} // namespace AtenIpexTypeXPU

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_dpcpp_sum_avg(
    const Tensor& grad,
    const Tensor& indices_t,
    const Tensor& offset2bag_t,
    const Tensor& bag_size_t,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights_t,
    int64_t padding_idx) {
  auto indices = indices_t.contiguous();
  auto offset2bag = offset2bag_t.contiguous();
  auto bag_size = bag_size_t.contiguous();
  auto per_sample_weights = per_sample_weights_t.contiguous();

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  ptrdiff_t numel = indices.numel();

  if (numel == 0) {
    // return empty bags
    return at::zeros({num_weights, grad.size(1)}, grad.options());
  }

  int64_t stride = grad_weight.stride(0);

  auto sorted_indices = at::empty_like(indices);
  auto sorted_begin = sorted_indices.data_ptr<index_t>();
  auto orig_indices = at::empty_like(indices);
  auto orig_begin = orig_indices.data_ptr<index_t>();

  // directly
  {
    sorted_indices.copy_(indices);
    xpu::pstl::iota(orig_begin, orig_begin + numel, (index_t)0);
    xpu::pstl::sort<index_t, index_t>(
        indices.data_ptr<index_t>(), sorted_begin, orig_begin, numel, false);
  }

  Tensor count;
  if (scale_grad_by_freq) {
    count = at::empty_like(sorted_indices);
    index_t* count_begin = count.data_ptr<index_t>();
    // Take the maximum of each count per unique key:
    // sorted: 2 5 5 5 7 7 8 9 9
    //  count: 1 3 3 3 2 2 1 2 2
    //
    xpu::pstl::count_by_segment<index_t, index_t, index_t>(
        sorted_begin,
        sorted_begin + numel,
        count_begin,
        [](index_t a, index_t b) { return Numerics<index_t>::eq(a, b); });
  }

  return embedding_backward_deterministic_kernel<scalar_t, index_t>(
      grad,
      orig_indices,
      sorted_indices,
      count,
      num_weights,
      padding_idx,
      mode == MODE_MEAN,
      offset2bag,
      bag_size,
      per_sample_weights);
}

template <typename scalar_t, typename index_t>
void EmbeddingBag_accGradParametersKernel_max(
    index_t* max_indices,
    scalar_t* gradOutput,
    scalar_t* gradWeight,
    int64_t stride,
    int64_t numBags) {
  auto& queue = dpcppGetCurrentQueue();
  auto chunksPerBag = CeilDiv(stride, (int64_t)64);
  auto numChunks = numBags * chunksPerBag;
  auto kernel_range = 1024 * 64;

  auto cgf = DPCPP_Q_CGF(cgh) {
    auto max_indices_data = max_indices;
    auto gradOutput_data = gradOutput;
    auto gradWeight_data = gradWeight;

    auto kfn = DPCPP_Q_KFN(sycl::nd_item<2> item) {
      auto max_indices_ptr = max_indices_data;
      auto gradOutput_ptr = gradOutput_data;
      auto gradWeight_ptr = gradWeight_data;

      auto chunkOffset = item.get_group()[0] * item.get_local_range()[1] +
          item.get_local_id()[1];

      for (auto chunk = chunkOffset; chunk < numChunks;
           chunk += item.get_group_range()[0] * item.get_global_range()[1]) {
        auto featureDim = (chunk % chunksPerBag) * item.get_local_range(0) +
            item.get_local_id(0);
        if (featureDim < stride) {
          auto bag = chunk / chunksPerBag;

          auto word_idx = max_indices_ptr[bag * stride + featureDim];
          if (word_idx >= 0) {
            // If bag is empty, we have max_indices[idx] set to -1 in forward.
            atomicAdd(
                (dpcpp_global_ptr_pt<scalar_t>)&(
                    gradWeight_ptr[word_idx * stride + featureDim]),
                gradOutput_ptr[bag * stride + featureDim]);
          }
        }
      }
    };

    // kick off kernel
    cgh.parallel_for(
        sycl::nd_range<2>(
            sycl::range<2>(kernel_range, 4), sycl::range<2>(64, 4)),
        kfn);
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, typename index_t>
Tensor embedding_bag_backward_dpcpp_max(
    const Tensor& grad,
    const Tensor& max_indices_t,
    int64_t num_weights,
    int64_t padding_idx) {
  auto max_indices = max_indices_t.contiguous();
  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());
  int64_t stride = grad_weight.stride(0);
  int64_t numBags = grad.size(0);

  EmbeddingBag_accGradParametersKernel_max<scalar_t>(
      max_indices.data_ptr<index_t>(),
      grad.data_ptr<scalar_t>(),
      grad_weight.data_ptr<scalar_t>(),
      stride,
      numBags);

  return grad_weight;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_dpcpp(
    const Tensor& weight_t,
    const Tensor& indices_t,
    const Tensor& offsets_t,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights_t,
    bool include_last_offset,
    int64_t padding_idx) {
  auto weight = weight_t.contiguous();
  auto indices_original = indices_t.contiguous();
  auto offsets_original = offsets_t.contiguous();
  auto per_sample_weights = per_sample_weights_t.contiguous();

  Tensor indices, offsets;
  std::tie(indices, offsets) =
      promoteIndicesAndOffsets(indices_original, offsets_original);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag_dpcpp", indices_arg, {kLong, kInt});
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag_dpcpp", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag_dpcpp", indices_arg, offsets_arg);
  isOnSameDevice("embedding_bag_dpcpp", indices_arg, offsets_arg);
  auto weight_arg = TensorArg(weight, "weight", 1);
  isOnSameDevice("embedding_bag_dpcpp", weight_arg, indices_arg);
  isOnSameDevice("embedding_bag_dpcpp", weight_arg, offsets_arg);

  bool ignore_offsets = indices.sizes().size() == 2;
  int64_t numIndices = indices.numel();
  int64_t numBags = ignore_offsets ? indices.size(0) : offsets.size(0);

  // include last offset = True, means the last element of offsets will be set
  // equal to the length of input. Default it is False.
  if (include_last_offset) {
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at least 1");
    numBags -= 1;
  }
  int64_t weight_total_elem = weight.numel();

  auto bag_size = at::empty(numBags, indices.options());
  auto offset2bag = at::empty({indices.size(0)}, indices.options());
  auto output = at::empty({numBags, weight.size(1)}, weight.options());

  Tensor max_indices = at::empty({numBags, weight.size(1)}, indices.options());

#ifndef VEC_EMBBAG_KERNEL_OPT
#define EXTEND_EMBBAG_TEMPLATE(mode) \
  embedding_bag_##mode##_template(   \
      indices,                       \
      offsets,                       \
      weight,                        \
      per_sample_weights,            \
      output,                        \
      offset2bag,                    \
      bag_size,                      \
      max_indices,                   \
      numIndices,                    \
      numBags,                       \
      weight.stride(0),              \
      padding_idx,                   \
      ignore_offsets)

  switch (mode) {
    case MODE_SUM:
      EXTEND_EMBBAG_TEMPLATE(sum);
      break;
    case MODE_MEAN:
      EXTEND_EMBBAG_TEMPLATE(mean);
      break;
    case MODE_MAX:
      EXTEND_EMBBAG_TEMPLATE(max);
      break;
    default:
      TORCH_CHECK(0, "Invalid EmbeddingBag mode (max, sum, mean) ...");
  };
#else
  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_bag_dpcpp",
      [&] {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dpcpp", [&] {
              EmbeddingBag_updateOutputKernel<scalar_t, index_t>(
                  mode,
                  indices.data_ptr<index_t>(),
                  offsets.data_ptr<index_t>(),
                  weight.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>(),
                  offset2bag.data_ptr<index_t>(),
                  weight_total_elem,
                  numIndices,
                  numBags,
                  weight.stride(0),
                  weight.stride(1),
                  bag_size.data_ptr<index_t>(),
                  mode == MODE_MAX ? max_indices.data_ptr<index_t>() : NULL,
                  per_sample_weights.defined()
                      ? per_sample_weights.data_ptr<scalar_t>()
                      : NULL,
                  per_sample_weights.defined() ? per_sample_weights.stride(0)
                                               : 0,
                  include_last_offset,
                  padding_idx,
                  ignore_offsets);
            });
      });
#endif

  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
      output, offset2bag, bag_size, max_indices);
}

Tensor _embedding_bag_dense_backward_dpcpp(
    const Tensor& grad_t,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const Tensor& per_sample_weights,
    int64_t padding_idx) {
  Tensor grad = grad_t.contiguous();
  Tensor result;

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "embedding_bag_dense_backward_dpcpp",
      [&] {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(), "embedding_bag_dense_backward_dpcpp", [&] {
              switch (mode) {
                case MODE_SUM:
                case MODE_MEAN:
                  if (mode == MODE_MEAN) {
                    TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  }
                  result =
                      embedding_bag_backward_dpcpp_sum_avg<scalar_t, index_t>(
                          grad,
                          indices,
                          offset2bag,
                          bag_size,
                          num_weights,
                          scale_grad_by_freq,
                          mode,
                          per_sample_weights,
                          padding_idx);
                  return result;
                case MODE_MAX:
                  TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
                  result = embedding_bag_backward_dpcpp_max<scalar_t, index_t>(
                      grad, max_indices, num_weights, padding_idx);
                  return result;
                default:
                  TORCH_CHECK(
                      0,
                      "Unknown mode for embedding_bag_backward_dpcpp ",
                      mode);
              }
            });
      });
  return result;
}

template <typename scalar_t, typename index_t>
static void _embedding_bag_per_sample_weights_backward_kernel(
    const scalar_t* grad,
    int64_t grad_stride0,
    int64_t grad_stride1,
    const scalar_t* weight,
    int64_t weight_stride0,
    int64_t weight_stride1,
    const index_t* indices, // contiguous
    const index_t* offset2bag, // contiguous
    int64_t num_samples,
    int64_t embedding_features,
    scalar_t* output,
    index_t padding_idx) {
  using accscalar_t = acc_type<scalar_t>;

  auto& dpcpp_queue = dpcppGetCurrentQueue();
  int64_t max_group_size = 64;

  int64_t num_group = (num_samples + max_group_size - 1) / max_group_size;
  sycl::range<1> global_range{num_group * max_group_size};
  sycl::range<1> local_range{max_group_size};

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(global_range, local_range),
        [=](sycl::nd_item<1> item_id) {
          int idx = item_id.get_global_linear_id();
          auto sg = item_id.get_sub_group();
          int sgSize =
              sg.get_local_range()[0]; // number of work-items in this sub-group
          int sgId = idx / sgSize; // subgroup index
          int sglid =
              sg.get_local_id()[0]; // index of the work-item in this sub-group

          int num_sg =
              num_group * max_group_size / sgSize; // number of sub-groups
          for (int sample_idx = sgId; sample_idx < num_samples;
               sample_idx += num_sg) {
            accscalar_t result = 0.;
            const int bag_idx = (int)offset2bag[sample_idx];
            const int embedding_idx = (int)indices[sample_idx];
            if (embedding_idx != padding_idx) {
              for (int feature_idx = sglid; feature_idx < embedding_features;
                   feature_idx += sgSize) {
                result +=
                    grad[grad_stride0 * bag_idx + grad_stride1 * feature_idx] *
                    weight
                        [weight_stride0 * embedding_idx +
                         weight_stride1 * feature_idx];
              }
            }
            // subgroup reduce sum
            for (int offset = sgSize / 2; offset > 0; offset /= 2) {
              result += sycl::shift_group_left(sg, result, offset);
            };
            if (sglid == 0) {
              output[sample_idx] = result;
            }
          }
        });
  };
  DPCPP_Q_SUBMIT(dpcpp_queue, cgf);
}

Tensor _embedding_bag_per_sample_weights_backward_dpcpp(
    const Tensor& grad,
    const Tensor& weight, // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  TORCH_CHECK(
      mode == MODE_SUM,
      "embedding_bag_backward: per_sample_weights only supported for mode='sum'");

  AT_ASSERT(grad.dim() == 2);
  auto embedding_features = grad.size(1);

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  AT_ASSERT(indices.dim() == 1);
  auto num_samples = indices.size(0);

  AT_ASSERT(weight.dim() == 2);
  AT_ASSERT(weight.size(1) == embedding_features);

  auto output = at::empty({num_samples}, grad.options());

  // Early return when there is no samples in the batch. This saves unnecesary
  // kernel launch
  if (num_samples == 0) {
    return output;
  }

  IPEX_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      grad.scalar_type(),
      "_embedding_bag_per_sample_weights_backward_dpcpp",
      [&]() {
        IPEX_DISPATCH_INDEX_TYPES(
            indices.scalar_type(),
            "_embedding_bag_per_sample_weights_backward_dpcpp",
            [&]() {
              _embedding_bag_per_sample_weights_backward_kernel<
                  scalar_t,
                  index_t>(
                  grad.data_ptr<scalar_t>(),
                  grad.stride(0),
                  grad.stride(1),
                  weight.data_ptr<scalar_t>(),
                  weight.stride(0),
                  weight.stride(1),
                  indices.data_ptr<index_t>(),
                  offset2bag.data_ptr<index_t>(),
                  num_samples,
                  embedding_features,
                  output.data_ptr<scalar_t>(),
                  padding_idx);
            });
      });
  return output;
}

} // namespace impl

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return impl::_embedding_bag_dpcpp(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

Tensor _embedding_bag_dense_backward(
    const Tensor& grad,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& bag_size,
    const Tensor& maximum_indices,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const c10::optional<at::Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;
  return impl::_embedding_bag_dense_backward_dpcpp(
      grad,
      indices,
      offset2bag,
      bag_size,
      maximum_indices,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights,
      padding_idx);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_forward_only(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const c10::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  return impl::_embedding_bag_dpcpp(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset,
      padding_idx);
}

Tensor _embedding_bag_per_sample_weights_backward(
    const Tensor& grad,
    const Tensor& weight, // NB: embedding table, not per_sample_weights
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    int64_t mode,
    int64_t padding_idx) {
  return impl::_embedding_bag_per_sample_weights_backward_dpcpp(
      grad, weight, indices_, offsets_, offset2bag, mode, padding_idx);
}

} // namespace AtenIpexTypeXPU
} // namespace at
