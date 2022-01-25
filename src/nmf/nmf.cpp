#include <algorithm>
#include "./nmf.hpp"


NMF::NMF(int N, int M, int K, std::optional<double> tolerance, std::optional<int> max_iterations,
    std::optional<int> random_seed, std::optional<float> alpha_W, std::optional<float> alpha_H, 
    std::optional<float> l1_ratio, bool verbose)
{
    _N = N; _M = M; _K = K;
    _tolerance = tolerance.has_value() ? tolerance.value() : 1e-4;
    _max_iterations = max_iterations.has_value() ? max_iterations.value() : 200;
    _random_seed = random_seed.has_value() ? random_seed.value() : -1;
    _alpha_W = alpha_W.has_value() ? alpha_W.value() : 0.0;
    _alpha_H = alpha_H.has_value() ? alpha_H.value() : 0.0;
    _l1_ratio = l1_ratio.has_value() ? l1_ratio.value() : 0.0;
    _verbose = verbose;
}


NMF::~NMF() {
    if(_W != nullptr)
        delete[] _W;

    if(_H != nullptr)
        delete[] _H; 
}


void NMF::fit_transform(C_REAL* V, C_REAL* W, C_REAL* H) {
    if(V == NULL)
        throw "V argument is uninitialized";

    _check_non_negative(V);

    if(_beta_loss <= 0 && std::min_element(V, V+(_N*_M)) == 0)
        throw "When beta_loss <= 0 and X contains zeros, the solver may diverge. Please add small values to X, or use a positive beta_loss.";

    // Load device where to run kernels
    Device device(_random_seed, _N, _M, _K, V, W, H);

    _fit_transform(&device);
    _error = _beta_divergence(&device);
    _save_results(&device);
}


void NMF::_save_results(Device* device) {
    _W = new C_REAL[_N*_K];
    _H = new C_REAL[_K*_M];

    std::copy(device->sW, device->sW + (_N*_K), _W);
    std::copy(device->sH, device->sH + (_K*_M), _H);
}


void NMF::_fit_transform(Device* device) {
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


void NMF::_fit_multiplicative_update(Device* device, float beta_loss, int max_iterations,
    double tolerance, float l1_reg_W, float l1_reg_H, float l2_reg_W, float l2_reg_H) 
{
    // used for the convergence criterion
    // Returns a float representing the divergence between X and WH, which is calculated as "||X - WH||_{loss}^2"
    double error_at_init = _beta_divergence(device);
    double previous_error{error_at_init};
    int n_iter{0};

    for (; n_iter < _max_iterations; n_iter++) {
        //(V*H') / (W*H*H')
        C_REAL* delta_W = _multiplicative_update_w(device, beta_loss, l1_reg_W, l2_reg_W);
        // W = W .* delta_W
        device->dot(device->sW, delta_W, device->sW, _N*_K);

        //(W'*V) / (W'*W*H)
        C_REAL* delta_H = _multiplicative_update_h(device, beta_loss, l1_reg_H, l2_reg_H);
        // H = H .* delta_H
        device->dot(device->sH, delta_H, device->sH, _K*_M);

        // test convergence criterion every 10 iterations
        if (tolerance > 0 && (n_iter % 10) == 0) {
            _error = _beta_divergence(device);

            if(_verbose)
                std::cout << "Epoch " << n_iter << ", error: " << _error << std::endl;
            
            if ((previous_error - _error) / error_at_init < tolerance)
                break;

            previous_error = _error;
        }
    }

    if (_verbose && (tolerance == 0 || n_iter % 10 != 0))
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
C_REAL* NMF::_multiplicative_update_w(Device* device, float beta_loss, float l1_reg_W, float l2_reg_W)
{
    // (numerator) VHt[N, K] = V[N, M] * H'[M, K]
    device->mat_mul(device->dV, device->sH, device->VHt, false, true, _N, _K, _M, _M, _M, _K);
    // (denominator) XXt[K, K] = H[K, M] * H'[M, K]
    device->mat_mul(device->sH, device->sH, device->XXt, false, true, _K, _K, _M, _M, _M, _K);
    // (denominator) delta_W[N, K] = W[N, K] * XXt[K, K]
    device->mat_mul(device->sW, device->XXt, device->delta_W, false, false, _N, _K, _K, _K, _K, _K);

    //Add L1 and L2 regularization
    if (l1_reg_W > 0)
        // denominator = denominator + l1_reg_W
        device->add_scalar(device->delta_W, device->delta_W, l1_reg_W, _N, _K);
    if (l2_reg_W > 0)
        //denominator = l2_reg_W * W + denominator
        device->axpy(device->sW, device->delta_W, l2_reg_W, _N*_K);

    device->adjust_matrix(device->delta_W, _N, _K);

    // delta_W[N, K] = numerator[N, K] / denominator[N, K]
    device->div_matrices(device->VHt, device->delta_W, device->delta_W, _N, _K);

    device->sync();

    return device->delta_W;
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
C_REAL* NMF::_multiplicative_update_h(Device* device, float beta_loss, float l1_reg_H, float l2_reg_H) {
    // (numerator) WtV[K, M] = W'[K, N] * V[N, M]
    device->mat_mul(device->sW, device->dV, device->WtV, true, false, _K, _M, _N, _K, _M, _M);
    // (denominator) XXt[K, K] = W'[K, N] * W[N, K]
    device->mat_mul(device->sW, device->sW, device->XXt, true, false, _K, _K, _N, _K, _K, _K);
    // (denominator) delta_H[K, M] = XXt[K, K] * H[K, M]
    device->mat_mul(device->XXt, device->sH, device->delta_H, false, false, _K, _M, _K, _K, _M, _M);

    //Add L1 and L2 regularization
    if (l1_reg_H > 0)
        // denominator = denominator + l1_reg_H
        device->add_scalar(device->delta_H, device->delta_H, l1_reg_H, _K, _M);
    if (l2_reg_H > 0)
        //denominator = l2_reg_H * H + denominator
        device->axpy(device->sH, device->delta_H, l2_reg_H, _K*_M);

    device->adjust_matrix(device->delta_H, _K, _M);

    // delta_H[K, M] = numerator[K, M] / denominator[K, M]
    device->div_matrices(device->WtV, device->delta_H, device->delta_H, _K, _M);

    device->sync();

    return device->delta_H;
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
float NMF::_beta_divergence(Device* device) {
    // Frobenius norm
    if(_beta_loss == 2.0) {
        // WH[N, M] = W[N, K] * H[K, M]
        device->mat_mul(device->sW, device->sH, device->WH, false, false, _N, _M, _K, _K, _M, _M);
        // WH = V - WH
        device->sub_matrices(device->dV, device->WH, device->WH, _N, _M);
        C_REAL result = device->nrm2(_N*_M, device->WH);
        
        device->sync();
        return result;
    }
    // TODO: add other beta divergences
    return 0;
}