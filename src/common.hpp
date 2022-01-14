#ifndef _COMMON_
#define _COMMON_

#include <stdlib.h>
#include <math.h>
#include <string>
#include <CL/sycl.hpp>

using namespace cl::sycl;

#ifdef REAL_S
#define C_REAL float
#else
#define C_REAL double
#endif

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

#endif