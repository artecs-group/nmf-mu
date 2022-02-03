#ifndef _DEVICE_NMF_
#define _DEVICE_NMF_

#include <stdlib.h>
#include <string>
#include <CL/sycl.hpp>
#include "../common.hpp"

using namespace cl::sycl;

// CUDA GPU selector
class CudaGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string DriverVersion = Device.get_info<sycl::info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

// Intel GPU
class IntelGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string vendor = Device.get_info<sycl::info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};


class Device {
    public:
        C_REAL *dV{nullptr}, *sW{nullptr}, *sH{nullptr}, *XXt{nullptr}, *VHt{nullptr}, 
            *WtV{nullptr}, *WH{nullptr}, *delta_W{nullptr}, *delta_H{nullptr}, *nrm2_result{nullptr};

        double tmat_mul{0}, tsub{0}, tdiv{0}, tnrm2{0}, tadd_scalar{0}, taxpy{0}, tdot_mul{0}, tadjust{0};

        Device(int seed, int N, int M, int K, C_REAL* V, C_REAL* W, C_REAL* H);
        ~Device();
        void sync();
        void mat_mul(C_REAL* A, C_REAL* B, C_REAL* C, bool _Ta, bool _Tb, int M, int N, int K, int lda, int ldb, int ldc);
        void sub_matrices(C_REAL* A, C_REAL* B, int M, int N);
        void div_matrices(C_REAL* A, C_REAL* B, C_REAL* C, int M, int N);
        void nrm2(int n, C_REAL* X, C_REAL* result);
        void add_scalar(C_REAL* in, C_REAL* out, float scalar, int M, int N);
        void axpy(C_REAL* x, C_REAL* y, float scalar, int n);
        void element_mul(int M, int N, C_REAL* A, C_REAL* B);
        void adjust_matrix(C_REAL* Mat, int M, int N);
    private:
        sycl::queue _queue;
        double _init{0};

        sycl::queue _get_queue();
        void _init_random_matrix(C_REAL* Mat, int N, int M, int _seed);
        void _get_nd_range_dimensions(int M, int N, int* work_items, int* group_size);
};

#endif