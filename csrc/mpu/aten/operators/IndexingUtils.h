#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/native/TensorIterator.h>

namespace at {
namespace AtenIpexTypeXPU {

[[noreturn]] static void invalid_mask(
    const Tensor& self,
    int64_t idx,
    const Tensor& mask,
    int64_t maskIdx) {
  TORCH_CHECK_INDEX(
      false,
      "The shape of the mask ",
      mask.sizes(),
      " at index ",
      maskIdx,
      " does not match the shape of the indexed tensor ",
      self.sizes(),
      " at index ",
      idx);
}

static std::vector<Tensor> expandTensors(
    const Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices) {
  std::vector<Tensor> result;
  for (c10::optional<Tensor> index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      Tensor index = std::move(*index_opt);
      if (index.scalar_type() == kByte || index.scalar_type() == kBool) {
        if (index.scalar_type() == kByte) {
          TORCH_WARN(
              "indexing with dtype torch.uint8 is now deprecated,"
              " please use a dtype torch.bool instead.");
        }
        // The sizes of the ByteTensor mask or bool tensor must match the sizes
        // of the corresponding dimensions in self
        for (int64_t j = 0; j < index.dim(); j++) {
          int64_t srcIdx = result.size() + j;
          if (index.size(j) != self.size(srcIdx)) {
            invalid_mask(self, srcIdx, index, j);
          }
        }
        // Replace with nonzeros
        auto nonzero = index.nonzero();
        for (int64_t j = 0; j < index.dim(); j++) {
          result.emplace_back(nonzero.select(1, j));
        }
      } else {
        result.emplace_back(std::move(index));
      }
    }
  }
  return result;
}

static void checkIndexTensorTypes(
    const c10::List<c10::optional<Tensor>>& indices) {
  for (c10::optional<Tensor> tensor_opt : indices) {
    if (tensor_opt.has_value() && tensor_opt->defined()) {
      auto scalarType = tensor_opt->scalar_type();
      if (scalarType != kLong && scalarType != kByte && scalarType != kBool) {
        TORCH_CHECK(
            "tensors used as indices must be long, byte or bool tensors");
      }
    }
  }
}

static bool hasContiguousSubspace(TensorList tl) {
  auto isDefined = [](const Tensor& tensor) { return tensor.defined(); };
  auto isNull = [](const Tensor& tensor) { return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

static std::tuple<Tensor, std::vector<Tensor>> transposeToFront(
    Tensor self,
    TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

inline std::tuple<Tensor, std::vector<Tensor>, std::vector<int64_t>>
transposeToFrontAndInvPerm(Tensor self, TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<int64_t> invPerm;
  std::vector<Tensor> transposedIndices;
  dims.reserve(self.dim());
  invPerm.resize(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    invPerm[dims[i]] = i;
  }
  return std::make_tuple(
      self.permute(dims), std::move(transposedIndices), std::move(invPerm));
}

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(tensors.size() >= 1, "all strides match");
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

static std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

static Tensor restride_src(
    const Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

static Tensor reshape_indexer(
    const Tensor& index,
    int64_t dims_before,
    int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices_list) {
    int64_t element_size_bytes = src.element_size();
    int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
    IntArrayRef replacement_shape;
    for (size_t dim = 0; dim < indices_list.size(); dim++) {
      if (!indices_list[dim].defined()) {
        if (dims_indexed == 0) {
          dims_before++;
        } else {
          dims_after++;
        }
        non_indexed_sizes.push_back(src.size(dim));
        non_indexed_strides.push_back(src.stride(dim) * element_size_bytes);
      } else {
        dims_indexed++;
        replacement_shape = indices_list[dim].sizes();
        indexed_sizes.push_back(src.size(dim));
        indexed_strides.push_back(src.stride(dim) * element_size_bytes);
      }
    }

    // Check if the indexed subspace contains a dim of size 0, but the
    // replacement shape does not. This implies that an index is out of bounds,
    // because there is no number that's a valid index for an empty tensor.
    // Normally, out of bounds is handled in the indexing kernel, but this case
    // fails earlier in restride_src with an unhelpful error message.
    if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
            indexed_sizes.end() &&
        std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
            replacement_shape.end()) {
      TORCH_CHECK_INDEX(
          false, "index is out of bounds for dimension with size 0");
    }

    this->dims_before = dims_before;
    this->dims_after = dims_after;
    this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

    for (auto& index : indices_list) {
      if (index.defined()) {
        indices.push_back(reshape_indexer(index, dims_before, dims_after));
      }
    }

    if (indices.size() >= 2) {
      if (!all_strides_match(indices)) {
        for (size_t i = 0; i < indices.size(); i++) {
          indices[i] = indices[i].contiguous();
        }
      }
    }
  }

  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  DimVector non_indexed_sizes;
  DimVector non_indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

static AdvancedIndex make_info(
    Tensor self,
    const c10::List<c10::optional<Tensor>>& orig) {
  checkIndexTensorTypes(orig);
  // LongTensors
  auto indices = expandTensors(self, orig);
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(
        false,
        "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ",
        shapes_as_str(indices));
  }
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }

  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined() && indices[i].device() != self.device()) {
      indices[i] = indices[i].to(self.device());
    }
  }
  return AdvancedIndex(self, indices);
}

static TensorIterator make_index_put_iterator(
    const AdvancedIndex& info,
    const Tensor& value) {
  TORCH_CHECK(
      is_expandable_to(value.sizes(), info.src.sizes()),
      "shape mismatch: value tensor of shape ",
      value.sizes(),
      " cannot be broadcast to indexing result of shape ",
      info.src.sizes());
  TORCH_CHECK(
      value.scalar_type() == info.src.scalar_type(),
      "Index put requires the source and destination dtypes match, "
      "got ",
      info.src.scalar_type(),
      " for the destination "
      "and ",
      value.scalar_type(),
      " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

static TensorIterator make_index_iterator(const AdvancedIndex& info) {
  TensorIteratorConfig config;
  config.check_all_same_dtype(false)
      .declare_static_dtype_and_device(
          info.src.scalar_type(), info.src.device())
      .add_owned_output(Tensor())
      .add_input(info.src);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

inline torch::List<c10::optional<Tensor>> toListOfOptionalTensors(
    ArrayRef<Tensor> list) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(list.size());
  for (const Tensor& a : list) {
    result.push_back(a);
  }
  return result;
}

inline torch::List<c10::optional<Tensor>> toListOfOptionalTensors(
    ArrayRef<IValue> list) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(list.size());
  for (const IValue& a : list) {
    result.push_back(a.toTensor());
  }
  return result;
}

} // namespace AtenIpexTypeXPU
} // namespace at
