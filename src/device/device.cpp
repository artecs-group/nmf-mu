#include <iostream>
#include <random>
#include <limits>
#include <oneapi/mkl/rng.hpp>
#include "oneapi/mkl.hpp"
#include "./device.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose non_trans = oneapi::mkl::transpose::nontrans;

/* Spacing of floating point numbers. */
static constexpr C_REAL EPS = std::numeric_limits<C_REAL>::epsilon();

Device::Device(int seed, int N, int M, int K, C_REAL* V, C_REAL* W, C_REAL* H) {
	_queue = _get_queue();

	dV         = malloc_device<C_REAL>(N * M, _queue);
    sW         = malloc_shared<C_REAL>(N * K, _queue);
    sH         = malloc_shared<C_REAL>(K * M, _queue);
 	delta_W    = malloc_device<C_REAL>(N * K, _queue);
    delta_H    = malloc_device<C_REAL>(K * M, _queue);
	XXt        = malloc_device<C_REAL>(K * K, _queue);
	VHt        = malloc_device<C_REAL>(N * K, _queue);
	WtV		   = malloc_device<C_REAL>(K * M, _queue);
	WH         = malloc_device<C_REAL>(N * M, _queue);

    if(W != nullptr && H != nullptr) {
        _queue.memcpy(sW, W, sizeof(C_REAL) * N*K);
        _queue.memcpy(sH, H, sizeof(C_REAL) * K*M);
    }
    else {
        _init_random_matrix(sW, N, K, seed);
        _init_random_matrix(sH, K, M, seed);
    }
	
	_queue.memcpy(dV, V, sizeof(C_REAL) * N*M);
    sync();
}


Device::~Device() {
	if(dV != nullptr) free(dV, _queue);
	if(sW != nullptr) free(sW, _queue);
	if(sH != nullptr) free(sH, _queue);
	if(XXt != nullptr) free(XXt, _queue);
	if(VHt != nullptr) free(VHt, _queue);
	if(WtV != nullptr) free(WtV, _queue);
	if(WH != nullptr) free(WH, _queue);
	if(delta_W != nullptr) free(delta_W, _queue);
	if(delta_H != nullptr) free(delta_H, _queue);
}


sycl::queue Device::_get_queue() {
#if defined(INTEL_GPU_DEVICE)
	IntelGPUSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CUDASelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
	std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    return queue;
}


void Device::sync() {
    _queue.wait();
}


void Device::_init_random_matrix(C_REAL* Mat, int N, int M, int _seed) {
    std::uint32_t seed = _seed < 0 ? (unsigned)time(NULL) : _seed;

    oneapi::mkl::rng::mcg31m1 engine(_queue, seed);
    oneapi::mkl::rng::uniform<C_REAL, oneapi::mkl::rng::uniform_method::standard> rng_distribution(0.0, 10000.0);
    oneapi::mkl::rng::generate(rng_distribution, engine, N*M, Mat);

    sync();
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
 * @param N rows of A and C
 * @param M columns of B and C
 * @param K columns of A and rows of B
 * @param lda Elements between successive rows of A
 * @param ldb Elements between successive rows of B
 * @param ldc Elements between successive rows of C
 */
void Device::mat_mul(C_REAL* A, C_REAL* B, C_REAL* C, bool _Ta, bool _Tb, 
	int M, int N, int K, int lda, int ldb, int ldc)
{
	oneapi::mkl::transpose Ta = _Ta ? trans : non_trans;
	oneapi::mkl::transpose Tb = _Tb ? trans : non_trans;

	oneapi::mkl::blas::row_major::gemm(_queue, Ta, Tb, M, N, K, 1, A, lda, B, ldb, 0, C, ldc);
    sync();
}


/**
 * @brief Computes B[M, N] = A[M, N] - B[M, N]
 * 
 * @param queue 
 * @param A 
 * @param B 
 * @param M 
 * @param N 
 */
void Device::sub_matrices(C_REAL* A, C_REAL* B, int M, int N) {
    // int group_size{0};
    // int work_items{0};
	// _get_nd_range_dimensions(M, N, &work_items, &group_size);

    // _queue.submit([&](handler& cgh) {
    //     cgh.parallel_for<class A_sub_B>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
    //         int i = item.get_global_id(0);
			
	//         if(i < M*N) {
    //             B[i] = -B[i];
    //         	B[i] = A[i] + B[i];
    //         }
    //     });
    // });
    oneapi::mkl::vm::sub(_queue, M*N, A, B, B);
    sync();
}


/**
 * @brief Computes C[M, N] = A[M, N] / B[M, N]
 * 
 * @param A 
 * @param B 
 * @param C 
 * @param M 
 * @param N 
 */
void Device::div_matrices(C_REAL* A, C_REAL* B, C_REAL* C, int M, int N) {
    // int group_size{0};
    // int work_items{0};
	// _get_nd_range_dimensions(M, N, &work_items, &group_size);

    // _queue.submit([&](handler& cgh) {
    //     cgh.parallel_for<class A_div_B>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
    //         int i = item.get_global_id(0);
			
	// 	    if(i < M*N)
    //         	C[i] = A[i] / B[i];
    //     });
    // });
    oneapi::mkl::vm::div(_queue, M*N, A, B, B);
    sync();
}


void Device::nrm2(int n, C_REAL* X, float* result) {
	oneapi::mkl::blas::nrm2(_queue, n, X, 1, result);
    sync();
}


void Device::add_scalar(C_REAL* in, C_REAL* out, float scalar, int M, int N) {
    int group_size{0};
    int work_items{0};
	_get_nd_range_dimensions(M, N, &work_items, &group_size);

    _queue.submit([&](handler& cgh) {
        cgh.parallel_for<class add_scalar>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            int i = item.get_global_id(0);
			
			if(i < M*N)
            	out[i] = in[i] + scalar;
        });
    });
    sync();
}


/**
 * @brief Calculates the number of work items and group size in base of the input dimensions.
 * 
 * @param M 
 * @param N 
 * @param work_items 
 * @param group_size 
 */
void Device::_get_nd_range_dimensions(int M, int N, int* work_items, int* group_size) {
	int max_work_group_size = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int remainder{0};
    
    if(max_work_group_size >= M*N)
        *group_size = M*N;
    else {
        *group_size = max_work_group_size < N ? max_work_group_size : N;
        // adjust work-groups number 
        remainder = (N == *group_size) ? 0 : *group_size - (N % *group_size);
    }
    *work_items = M * (N + remainder);
}


void Device::axpy(C_REAL* x, C_REAL* y, float scalar, int n) {
	oneapi::mkl::blas::axpy(_queue, n, scalar, x, 1, y, 1);
    sync();
}


void Device::adjust_matrix(C_REAL* Mat, int M, int N) {
    _queue.submit([&](handler& cgh) {
        cgh.parallel_for<class adjust_matrix>(range<2>(M, N), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(Mat[i*N + j] == 0)
                Mat[i*N + j] = EPS;
        });
    });
    sync();
}


void Device::element_mul(int M, int N, C_REAL* A, C_REAL* B) {
    // int group_size{0};
    // int work_items{0};
	// _get_nd_range_dimensions(M, N, &work_items, &group_size);

    // _queue.submit([&](handler& cgh) {
    //     cgh.parallel_for<class element_mul>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
    //         int i = item.get_global_id(0);
			
	// 		if(i < M*N)
    //         	out[i] = x[i] * y[i];
    //     });
    // });
    oneapi::mkl::vm::mul(_queue, M*N, A, B, B);
    sync();
}