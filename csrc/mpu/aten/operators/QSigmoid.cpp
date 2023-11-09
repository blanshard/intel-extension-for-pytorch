#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/core/CompileTimeFunctionPointer.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>
#include <torch/library.h>

#include <runtime/Utils.h>
#include <functional>
#include "Loops.h"
#include "comm/ATDispatch.h"
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;
using namespace at::native;

namespace at {
namespace AtenIpexTypeQuantizedXPU {

void qsigmoid_kernel(const Tensor& qx, Tensor& qy) {
  IPEX_DISPATCH_QINT_TYPES(qx.scalar_type(), "qsigmoid_xpu", [&]() {
    // output_scale = 1 / (2^8 - 1) * 2, the scale setted like minmax observer
    double output_scale = (0.00392157 * 2.f);
    // Algin with PyTorch symmetric quantization. The real dtype is a QInt8
    // See Note: [Opaque u8 tensor]
    int64_t output_zero_point = 128;

    // The range of sigmoid's output is (0,1) ,therefore we quantize our data
    // with zp = 1/(2^8-1)  at the end of this operator to maximize the
    // accuracy, which is diffenent from pytorch cpu implement(zp=1/2^8).

    auto data_type = at::kQUInt8;
    if (SCALAR_TYPE == at::kQInt32) {
      // output_scale = 1 / (2^32 - 1) * 2.f
      output_scale = (2.3283064370807974e-10) * 2.f;
      data_type = at::kQInt32;
    }
    auto x = at::dequantize(qx);
    auto y = at::sigmoid(x);
    qy = at::quantize_per_tensor(y, output_scale, output_zero_point, data_type);
  });
};

Tensor sigmoid(const Tensor& qx) {
  Tensor qy;
  qsigmoid_kernel(qx, qy);
  return qy;
}

Tensor& sigmoid_(Tensor& qx) {
  qsigmoid_kernel(qx, qx);
  return qx;
}

Tensor& sigmoid_out(const Tensor& qx, Tensor& qy) {
  qsigmoid_kernel(qx, qy);
  return qy;
}

} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
