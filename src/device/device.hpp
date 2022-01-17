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


class Device {
    public:
        C_REAL* dV{nullptr}, sW{nullptr}, sH{nullptr}, H_sum{nullptr}, 
            HHt{nullptr}, VHt{nullptr}, WH{nullptr}, delta_W{nullptr}, delta_H{nullptr};

        Device(int seed, int N, int M, int K, C_REAL* V);
        ~Device();
        void gemm(C_REAL* A, C_REAL* B, C_REAL* C, bool _Ta, bool _Tb, int M, int N, int K, int lda, int ldb, int ldc);
        void sub_matrices(C_REAL* A, C_REAL* B, C_REAL* C, int M, int N);
        C_REAL nrm2(int n, C_REAL* X);
    private:
        int _random_seed;
        sycl::queue _queue;

        sycl::queue _get_queue();
        void _init_random_matrix(C_REAL* Mat, int N, int M, int seed);
};

#endif