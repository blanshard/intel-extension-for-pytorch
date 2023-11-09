#include <core/Memory.h>
#include <core/detail/IndexUtils.h>
#include <core/detail/TensorInfo.h>
#include <utils/DPCPP.h>
#include "comm/ATDispatch.h"
#include "comm/Numerics.h"
#include "comm/RegistrationDeclarations.h"

#include "Sort.h"

using namespace xpu::dpcpp::detail;
using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& input,
    int64_t dim,
    bool order,
    Tensor& sorted,
    Tensor& indices) {
  return sort_out_stable(input, false, dim, order, sorted, indices);
}

std::tuple<at::Tensor, at::Tensor> sort(
    const at::Tensor& self,
    int64_t dim,
    bool descending) {
  Tensor sorted, indices;
  return sort_out_stable(self, false, dim, descending, sorted, indices);
}

std::tuple<Tensor, Tensor> sort(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor sorted, indices;
  return sort_out_stable(self, stable, dim, descending, sorted, indices);
}

std::tuple<Tensor&, Tensor&> sort_out(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  return sort_out_stable(self, stable, dim, descending, values, indices);
}

Tensor argsort(const Tensor& self, bool stable, int64_t dim, bool descending) {
  Tensor sorted, indices;
  return std::get<1>(
      sort_out_stable(self, stable, dim, descending, sorted, indices));
}

} // namespace AtenIpexTypeXPU
} // namespace at
