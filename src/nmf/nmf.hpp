/*******************************************************************************
 * Copyright 2022-2023 
 * Author: Youssef El Faqir El Rhazoui
 * MIT License
*******************************************************************************/

#ifndef _NMF_
#define _NMF_

#include <iostream>
#include "common.hpp"

void init_random_matrix(C_REAL* Mat, int N, int M, int seed);

class NMF {
/*
 * Non-Negative Matrix Factorization (NMF).
 * Find two non-negative matrices (W, H) whose product approximates the non-
 * negative matrix V. This factorization can be used for example for
 * dimensionality reduction, source separation or topic extraction.
 *
 * The objective function is:
 *  .. math::
 *      0.5 * ||V - WH||_{loss}^2
 *      + alpha_W * l1_{ratio} * n_features * ||vec(W)||_1
 *      + alpha_H * l1_{ratio} * n_samples * ||vec(H)||_1
 *      + 0.5 * alpha_W * (1 - l1_{ratio}) * n_features * ||W||_{Fro}^2
 *      + 0.5 * alpha_H * (1 - l1_{ratio}) * n_samples * ||H||_{Fro}^2
 *
 * Where:
 * :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)
 *
 * :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)
 *
 * The generic norm :math:`||V - WH||_{loss}` may represent
 * the Frobenius norm or another supported beta-divergence loss.
 *
 * The choice between options is controlled by the `beta_loss` parameter.
 *
 * The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
 * `H` to keep their impact balanced with respect to one another and to the data fit
 * term as independent as possible of the size `n_samples` of the training set.
 *
 * The objective function is minimized with an alternating minimization of W
 * and H.
*/
    public:
        NMF(int N, int M, int K, double tolerance=1e-4, int max_iterations=200,
            int random_seed=-1, float alpha_W=0.0, float alpha_H=0.0, 
            float l1_ratio=0.0) : 
                _N(N), _M(M), _K(K), _tolerance(tolerance), _max_iterations(max_iterations),
                _random_seed(random_seed), _alpha_W(alpha_W), _alpha_H(alpha_H), 
                _l1_ratio(l1_ratio) {}
        
        ~NMF();
        int get_iterations() { return _iterations; }
        double get_error() { return _error; }
        C_REAL* get_W() { return _W; }
        C_REAL* get_H() { return _H; }

        void fit_transform(const C_REAL* V);
    
    private:
        int _N, _M, _K;
        float _beta_loss{2.0};
        double _tolerance;
        int _max_iterations;
        int _random_seed; // by -1 we mean initialize seed with time
        float _alpha_W;
        float _alpha_H;
        float _l1_ratio;
        float _l1_reg_W, _l1_reg_H, _l2_reg_W, _l2_reg_H;
        int _iterations{0};
        double _error{0.0};
        C_REAL* _W{NULL};
        C_REAL* _H{NULL};

        void _fit_transform(C_REAL* V, C_REAL* W, C_REAL* H);
        void _scale_regularization(const C_REAL* V, float* l1_reg_W, float* l1_reg_H, float* l2_reg_W, float* l2_reg_H);
        void _fit_multiplicative_update(const C_REAL* V, C_REAL* W, C_REAL* H, float beta_loss, 
            int max_iterations, double tolerance, float l1_reg_W, float l1_reg_H,
            float l2_reg_W, float l2_reg_H);
        float _beta_divergence(const C_REAL* V, C_REAL* W, C_REAL* H, bool square_root);
        C_REAL* _multiplicative_update_w(const C_REAL* V, C_REAL* W, C_REAL* H, float beta_loss, float l1_reg_W, 
            float l2_reg_W, float gamma, C_REAL* H_sum, C_REAL* HHt, C_REAL* VHt);
        C_REAL* _multiplicative_update_h(const C_REAL* V, C_REAL* W, C_REAL* H, float beta_loss, float l1_reg_H, 
            float l2_reg_H, float gamma);
        void save_results(C_REAL* W, C_REAL* H);
};

#endif