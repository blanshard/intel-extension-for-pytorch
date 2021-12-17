import torch
import torch.nn as nn
from torch.testing._internal.common_utils import TestCase

import ipex

cpu_device = torch.device("cpu")
dpcpp_device = torch.device("xpu")


class TestNNMethod(TestCase):
    def test_adaptive_max_pool3d(self, dtype=torch.float):
        x_cpu = torch.randn([1, 1, 4, 4, 4], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device)
        grad_cpu = torch.randn([1, 1, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        self.assertEqual(x_cpu, x_dpcpp.to(cpu_device))
        self.assertEqual(grad_cpu, grad_dpcpp.to(cpu_device))

        max_pool = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        y_cpu[0].backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        print("y_dpcpp", y_dpcpp[0].cpu())
        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.cpu())

        # self.assertEqual( y_cpu, y_dpcpp.to(cpu_device))
        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))

    def test_channels_last_simple_fwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 4, 6], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last_3d)
        max_pool = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])

        max_pool.to("xpu")
        y_dpcpp = max_pool(x_dpcpp)
        print("y_dpcpp", y_dpcpp[0].cpu())
        self.assertEqual(y_cpu[0], y_dpcpp[0].to(cpu_device))

    def test_channels_last_simple_bwd(self, dtype=torch.float):
        x_cpu = torch.randn([1, 4, 4, 4, 6], device=cpu_device, dtype=dtype)
        x_dpcpp = x_cpu.to(dpcpp_device).to(memory_format=torch.channels_last_3d)
        grad_cpu = torch.randn([1, 4, 2, 2, 2], device=cpu_device)
        grad_dpcpp = grad_cpu.to(dpcpp_device)

        max_pool = nn.AdaptiveMaxPool3d((2, 2, 2), return_indices=True)

        # cpu
        x_cpu.requires_grad_(True)
        y_cpu = max_pool(x_cpu)
        print("y_cpu", y_cpu[0])
        y_cpu[0].backward(grad_cpu)
        print("y_cpu backward", x_cpu.grad)

        max_pool.to("xpu")
        x_dpcpp.requires_grad_(True)
        y_dpcpp = max_pool(x_dpcpp)

        print("y_dpcpp", y_dpcpp[0].cpu())
        grad_dpcpp = grad_cpu.to("xpu")
        y_dpcpp[0].backward(grad_dpcpp)
        print("y_dpcpp backward", x_dpcpp.grad.cpu())

        self.assertEqual(x_cpu.grad, x_dpcpp.grad.to(cpu_device))
