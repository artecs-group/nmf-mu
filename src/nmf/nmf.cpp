#include <random>
#include "nmf/nmf.hpp"
#include "device/device.hpp"

void init_random_matrix(C_REAL* Mat, int N, int M, int seed) {
    if(seed == -1)
        srand((unsigned)time(NULL));
    else
        srand(seed);
    
    for (size_t i = 0; i < N*M; i++)
        Mat[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);
}


NMF::~NMF() {
    if(_W != NULL)
        delete[] _W;

    if(_H != NULL)
        delete[] _H; 
}


void NMF::fit_transform(const C_REAL* V) {
    if(V == NULL)
        throw "V argument is uninitialized";

    // queue where to run the kernels
    sycl::queue queue = get_queue();
    C_REAL* dV = malloc_device<C_REAL>(_N * _M, queue);
    C_REAL* sW = malloc_shared<C_REAL>(_N * _K, queue);
    C_REAL* sH = malloc_shared<C_REAL>(_M * _K, queue);
    
    // Those matrices are for initialize random numbers.
    // It is possible to use "oneapi/mkl/rng.hpp"
    // but breaks the compatibility with other devices such CUDA.
    init_random_matrix(sW, _N, _K, _random_seed);
    init_random_matrix(sH, _M, _K, _random_seed);
    copy_in(queue, V, dV, _N, _M);

    _fit_transform(V, W, H);

    _error = _beta_divergence();
    save_results(sW, sH);
    free(sW, queue);
    free(sH, queue);
}


void NMF::save_results(C_REAL* W, C_REAL* H) {
    C_REAL* _W = new C_REAL[_N*_K];
    C_REAL* _H = new C_REAL[_M*_K];

    std::copy(W, W + (_N*_K), _W);
    std::copy(H, H + (_N*_K), _H); 
}


void NMF::_fit_transform(C_REAL* V, C_REAL* W, C_REAL* H) {
    
    
}
