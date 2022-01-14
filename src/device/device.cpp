#include <iostream>
#include "device/device.hpp"

sycl::queue get_queue() {
#if defined(INTEL_IGPU_DEVICE)
	IntelGPUSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CUDASelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue q{selector};
	std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    return q;
}


void copy_in(sycl::queue queue, C_REAL* in, C_REAL* out, int N, int M) {
    queue.memcpy(out, in, sizeof(C_REAL) * N*M);
}