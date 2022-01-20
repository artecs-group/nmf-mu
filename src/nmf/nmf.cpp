#include <algorithm>
#include<time.h>
#include "nmf/nmf.hpp"


double gettime() {
	double final_time;
	struct timeval tv1;
	
	gettimeofday(&tv1, (struct timezone*)0);
	final_time = (tv1.tv_usec + (tv1.tv_sec)*1000000ULL);

	return final_time;
}


NMF::~NMF() {
    if(_W != nullptr)
        delete[] _W;

    if(_H != nullptr)
        delete[] _H; 
}


void NMF::fit_transform(const C_REAL* V, bool verbose) {
    if(V == NULL)
        throw "V argument is uninitialized";

    _check_non_negative(V);

    if(_beta_loss <= 0 && std::min_element(V, V+(_N*_M)) == 0)
        throw "When beta_loss <= 0 and X contains zeros, 
                the solver may diverge. Please add small values 
                to X, or use a positive beta_loss.";

    // Load device where to run kernels
    Device device(_random_seed, _N, _M, _K, V);

    double t_init = gettime();

    _fit_transform(device, verbose);
    _error = _beta_divergence(device, _beta_loss);

    std::cout << "Total time = " << (gettime() - t_init) << " (us)" << std::endl;
    std::cout << "Final error = " << _error << std::endl;

    _save_results(device);
}


void NMF::_save_results(Device device) {
    C_REAL* _W = new C_REAL[_N*_K];
    C_REAL* _H = new C_REAL[_M*_K];

    std::copy(device.W(), device.W() + (_N*_K), _W);
    std::copy(device.H(), device.H() + (_N*_K), _H); 
}


void NMF::_fit_transform(Device device, bool verbose) {
    _scale_regularization(&_l1_reg_W, &_l1_reg_H, &_l2_reg_W, &_l2_reg_H);
    _fit_multiplicative_update(device, _beta_loss, _max_iterations, _tolerance, 
        _l1_reg_W, _l1_reg_H, _l2_reg_W, _l2_reg_H, verbose);
}


/**
 * @brief Checks if there is non-negative numbers
 * 
 * @param V 
 */
void NMF::_check_non_negative(C_REAL* V) {
    for (size_t i{0}; i < _N*_M; i++) {
        if(V[i] < 0)
            throw "Not allowed negative values in matrix V.";
    }
}


void NMF::_scale_regularization(float* l1_reg_W, float* l1_reg_H, float* l2_reg_W, float* l2_reg_H) {
    int n_samples{_N}, n_features{_M};

    *l1_reg_W = n_features * (_alpha_W * _l1_ratio);
    *l1_reg_H = n_samples * (_alpha_H * _l1_ratio);
    *l2_reg_W = n_features * (_alpha_W * (1.0f - _l1_ratio));
    *l2_reg_H = n_samples * (_alpha_H * (1.0f - _l1_ratio));
}


void NMF::_fit_multiplicative_update(Device device, float beta_loss, int max_iterations,
    double tolerance, float l1_reg_W, float l1_reg_H, float l2_reg_W, float l2_reg_H, 
    bool verbose) 
{
    // used for the convergence criterion
    // Returns a float representing the divergence between X and WH, which is calculated as "||X - WH||_{loss}^2"
    double error_at_init = _beta_divergence(device, beta_loss);
    double previous_error{error_at_init};
    int n_iter{0};

    for (; n_iter < _max_iterations; n_iter++) {
        //(V*H') / (W*H*H')
        C_REAL* delta_W = _multiplicative_update_w(device, beta_loss, l1_reg_W, l2_reg_W);
        // W = W .* delta_W
        device.dot(device.W, delta_W, device.W, _N*_K);

        //(W'*V) / (W'*W*H)
        C_REAL* delta_H = _multiplicative_update_h(device, beta_loss, l1_reg_H, l2_reg_H);
        // H = H .* delta_H
        device.dot(device.H, delta_H, device.H, _K*_M);

        // test convergence criterion every 10 iterations
        if (tolerance > 0 && (n_iter % 10) == 0) {
            _error = _beta_divergence(device, beta_loss);

            if(verbose)
                std::cout << "Epoch " << n_iter << ", error: " << _error << std::endl;
            
            if ((previous_error - _error) / error_at_init < tolerance)
                break;

            previous_error = _error
        }
    }

    if (verbose && (tolerance == 0 || n_iter % 10 != 0))
        std::cout << "Epoch " << n_iter << ", error: " << _error << std::endl;

    _iterations = n_iter;
}


/**
 * @brief Calculates delta_W as (V*H') / (W*H*H')
 * 
 * @param device 
 * @param beta_loss unused in current version
 * @param l1_reg_W 
 * @param l2_reg_W 
 * @param gamma 
 * @return C_REAL* 
 */
C_REAL* _multiplicative_update_w(Device device, float beta_loss, float l1_reg_W, float l2_reg_W)
{
    // (numerator) VHt[N, K] = V[N, M] * H'[M, K]
    device.gemm(device.V, device.H, device.VHt, false, true, _N, _K, _M, _M, _K, _K);

    // (denominator) XXt[K, K] = H[K, M] * Ht[M, K]
    device.gemm(device.H, device.H, device.XXt, false, true, _K, _K, _M, _M, _K, _K);
    // (denominator) delta_W[N, K] = W[N, K] * XXt[K, K]
    device.gemm(device.W, device.XXt, device.delta_W, false, false, _N, _K, _K, _K, _K, _K);

    //Add L1 and L2 regularization
    if (l1_reg_W > 0)
        // denominator = denominator + l1_reg_W
        device.add_scalar(device.delta_W, device.delta_W, l1_reg_W, _N, _K);
    if (l2_reg_W > 0)
        //denominator = l2_reg_W * W + denominator
        device.axpy(device.W, device.delta_W, l2_reg_W, _N*_K);

    device.adjust_matrix(device.delta_W, _N, _K);

    // delta_W[N, K] = numerator[N, K] / denominator[N, K]
    device.div_matrices(device.VHt, device.delta_W, delta.delta_W, _N, _K);

    return device.delta_W;
}


/**
 * @brief Calculates delta_H as (W'*V) / (W'*W*H)
 * 
 * @param device 
 * @param beta_loss unused in current version
 * @param l1_reg_H 
 * @param l2_reg_H 
 * @return C_REAL* 
 */
C_REAL* _multiplicative_update_h(Device device, float beta_loss, float l1_reg_H, float l2_reg_H) {
    // (numerator) WtV[K, M] = W'[K, N] * V[N, M]
    device.gemm(device.W, device.V, device.WtV, true, false, _K, _M, _N, _N, _M, _M);

    // (denominator) XXt[K, K] = W'[K, N] * W[N, K]
    device.gemm(device.W, device.W, device.XXt, true, false, _K, _K, _N, _N, _K, _K);
    // (denominator) delta_H[K, M] = XXt[K, K] * H[K, M]
    device.gemm(device.XXt, device.H, device.delta_H, false, false, _K, _M, _K, _K, _M, _M);

    //Add L1 and L2 regularization
    if (l1_reg_H > 0)
        // denominator = denominator + l1_reg_H
        device.add_scalar(device.delta_H, device.delta_H, l1_reg_H, _K, _M);
    if (l2_reg_H > 0)
        //denominator = l2_reg_H * H + denominator
        device.axpy(device.H, device.delta_H, l2_reg_H, _K*_M);

    device.adjust_matrix(device.delta_H, _K, _M);

    // delta_H[K, M] = numerator[K, M] / denominator[K, M]
    device.div_matrices(device.WtV, device.delta_H, delta.delta_H, _K, _M);

    return device.delta_H;
}


/**
 * @brief Compute the beta-divergence of X and dot(W, H), "||X - WH||_{loss}^2".
 * 
 * @param V 
 * @param W 
 * @param H 
 * @param square_root 
 * @param queue 
 * @return float 
 */
float NMF::_beta_divergence(Device device) {
    // Frobenius norm
    if(_beta_loss == 2.0) {
        // WH = W * H
        device.gemm(device.W, device.H, device.WH, false, false, _N, _M, _K, _K, _M, _M);
        // WH = V - WH
        device.sub_matrices(device.V, device.WH, device.WH, _N, _M);
        C_REAL result = device.nrm2(_N*_M, device.WH);

        return result;
    }
    // TODO: add other beta divergences
    return 0;
}