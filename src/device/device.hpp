#ifndef _DEVICE_
#define _DEVICE_

#include <stdlib.h>
#include <string>
#include <CL/sycl.hpp>
#include "common.hpp"

using namespace cl::sycl;

// CUDA GPU selector
class CudaGpuSelector : public device_selector {
    public:
        int operator()(const device &Device) const override {
            const std::string DriverVersion = Device.get_info<info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

// Intel GPU
class IntelGpuSelector : public device_selector {
    public:
        int operator()(const device &Device) const override {
            const std::string vendor = Device.get_info<info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};

sycl::queue get_queue();
void copy_in(sycl::queue queue, C_REAL* in, C_REAL* out, int N, int M);

#endif