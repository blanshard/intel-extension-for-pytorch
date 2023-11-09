#include <ATen/ATen.h>
#include <core/Allocator.h>
#include "comm/RegistrationDeclarations.h"

using namespace xpu::dpcpp;

namespace at {
namespace AtenIpexTypeXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  recordStreamInDevAlloc(
      self.storage().data_ptr(),
      DPCPPStream::unpack3(
          stream.id(), stream.device_index(), stream.device_type()));
}
} // namespace AtenIpexTypeXPU

namespace AtenIpexTypeQuantizedXPU {
void record_stream(Tensor& self, c10::Stream stream) {
  recordStreamInDevAlloc(
      self.storage().data_ptr(),
      DPCPPStream::unpack3(
          stream.id(), stream.device_index(), stream.device_type()));
}
} // namespace AtenIpexTypeQuantizedXPU
} // namespace at
