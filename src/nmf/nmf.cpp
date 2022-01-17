#include <algorithm>
#include "nmf/nmf.hpp"

NMF::~NMF() {
    if(_W != nullptr)
        delete[] _W;

    if(_H != nullptr)
        delete[] _H; 
}


void NMF::fit_transform(const C_REAL* V) {
    if(V == NULL)
        throw "V argument is uninitialized";

    _check_non_negative(V);

    if(_beta_loss <= 0 && std::min_element(V, V+(_N*_M)) == 0)
        throw "When beta_loss <= 0 and X contains zeros, 
                the solver may diverge. Please add small values 
                to X, or use a positive beta_loss.";

    // Load device where to run kernels
    Device device(_random_seed, _N, _M, _K, V);

    _fit_transform(device);

    _error = _beta_divergence(device, _beta_loss);
    _save_results(device);
}


void NMF::_save_results(Device device) {
    C_REAL* _W = new C_REAL[_N*_K];
    C_REAL* _H = new C_REAL[_M*_K];

    std::copy(device.W(), device.W() + (_N*_K), _W);
    std::copy(device.H(), device.H() + (_N*_K), _H); 
}


void NMF::_fit_transform(Device device) {
    _scale_regularization(&_l1_reg_W, &_l1_reg_H, &_l2_reg_W, &_l2_reg_H);
    _fit_multiplicative_update(device, _beta_loss, _max_iterations, _tolerance, 
        _l1_reg_W, _l1_reg_H, _l2_reg_W, _l2_reg_H);
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
    double tolerance, float l1_reg_W, float l1_reg_H, float l2_reg_W, float l2_reg_H) 
{
    // gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    float gamma{0};
    if(beta_loss < 1)
        gamma = 1.0 / (2.0 - beta_loss);
    else if(beta_loss > 2)
        gamma = 1.0 / (beta_loss - 1.0);
    else
        gamma = 1.0;

    // used for the convergence criterion
    // Returns a float representing the divergence between X and WH, which is calculated as "||X - WH||_{loss}^2"
    double error_at_init = _beta_divergence(device, beta_loss);
    double previous_error{error_at_init};

    for (size_t i{0}; i < _max_iterations; i++) {
        //(X*H') / (W*H*H')
        _multiplicative_update_w(device, beta_loss, l1_reg_W, l2_reg_W, gamma);
    }
    
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
        gemm(device.W, device.H, device.WH, false, false, _N, _M, _K, _K, _M, _M);
        // WH = V - WH
        sub_matrices(device.V, device.WH, device.WH, _N, _M);
        C_REAL result = nrm2(_N*_M, device.WH);

        return result;
    }
    // TODO: add other beta divergences
    return 0;
}