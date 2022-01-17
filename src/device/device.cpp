#include <iostream>
#include <random>
#include "device/device.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose non_trans = oneapi::mkl::transpose::nontrans;

/* Spacing of floating point numbers. */
constexpr C_REAL eps{2.2204e-16};

Device::Device(int seed, int N, int M, int K, C_REAL* V) {
	_random_seed = seed;
	_queue = _get_queue();

	dV         = malloc_device<C_REAL>(N * M, _queue);
    sW         = malloc_shared<C_REAL>(N * K, _queue);
    sH         = malloc_shared<C_REAL>(K * M, _queue);
 	delta_W    = malloc_shared<C_REAL>(N * K, _queue);
    delta_H    = malloc_shared<C_REAL>(K * M, _queue);
	HHt        = malloc_device<C_REAL>(M * M, _queue);
	VHt        = malloc_device<C_REAL>(N * M, _queue);
	WH         = malloc_device<C_REAL>(N * M, _queue);
	H_sum      = malloc_device<C_REAL>(K , _queue);

    // Those matrices are for initialize random numbers.
    // It is possible to use "oneapi/mkl/rng.hpp"
    // but breaks the compatibility with other devices such CUDA.
    _init_random_matrix(sW, N, K, _random_seed);
    _init_random_matrix(sH, K, M, _random_seed);
	
	_queue.memcpy(dV, V, sizeof(C_REAL) * N*M);
}


Device::~Device() {
	if(dV != nullptr) free(dV, _queue);
	if(sW != nullptr) free(sW, _queue);
	if(sH != nullptr) free(sH, _queue);
	if(H_sum != nullptr) free(H_sum, _queue);
	if(HHt != nullptr) free(HHt, _queue);
	if(VHt != nullptr) free(VHt, _queue);
	if(WH != nullptr) free(WH, _queue);
	if(delta_W != nullptr) free(delta_W, _queue);
	if(delta_H != nullptr) free(delta_H, _queue);
}


sycl::queue Device::_get_queue() {
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


void Device::_init_random_matrix(C_REAL* Mat, int N, int M, int seed) {
    if(seed == -1)
        srand((unsigned)time(NULL));
    else
        srand(seed);
    
    for (size_t i = 0; i < N*M; i++)
        Mat[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);
}


/**
 * @brief Computes A * B. The dimensions of the matrices have to be like: 
 * 	op(A)[M, K] * op(B)[K, N] -> C[M, N].
 *  Where: 
 * 		* op(A) = A or A' and op(B) = B or B'
 * 
 * @param queue 
 * @param out 
 * @param A 
 * @param B 
 * @param _Ta 
 * @param _Tb 
 * @param N 
 * @param M 
 * @param K 
 * @param lda Elements between successive rows of A
 * @param ldb Elements between successive rows of B
 * @param ldc Elements between successive rows of C
 */
void Device::gemm(C_REAL* A, C_REAL* B, C_REAL* C, bool _Ta, bool _Tb, 
	int M, int N, int K, int lda, int ldb, int ldc)
{
	oneapi::mkl::transpose Ta = _Ta ? trans : non_trans;
	oneapi::mkl::transpose Tb = _Tb ? trans : non_trans;

	oneapi::mkl::blas::gemm(_queue, Ta, Tb, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
}


/**
 * @brief Computes C[M, N] = A[M, N] - B[M, N]
 * 
 * @param queue 
 * @param A 
 * @param B 
 * @param M 
 * @param N 
 */
void Device::sub_matrices(C_REAL* A, C_REAL* B, C_REAL* C, int M, int N) {
    int max_work_group_size = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int group_size = max_work_group_size < N ? max_work_group_size : N;

    // adjust work-groups number 
    int remainder = (N == group_size) ? 0 : group_size - (N % group_size);
    int work_items = M * (N + remainder);

    _queue.submit([&](handler& cgh) {
        cgh.parallel_for<class A_sub_B>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            int i = item.get_global_id(0);
			
			if(i < M*N)
            	C[i] = A[i] - B[i];
        });
    });	
}


C_REAL Device::nrm2(int n, C_REAL* X) {
	C_REAL result{0.0};
	nrm2(_queue, n, X, 1, &result);
	return result;
}

