#include <iostream>
#include <string>
#include <sys/time.h>
#include "./common.hpp"
#include "./nmf/nmf.hpp"


double gettime() {
	double final_time;
	struct timeval tv1;
	
	gettimeofday(&tv1, (struct timezone*)0);
	final_time = (tv1.tv_usec + (tv1.tv_sec)*1000000ULL);

	return final_time;
}


void parse_arguments(int argc, char **argv, int* N, int* M, int* K, char* file_name, std::optional<double>* tolerance,
    std::optional<int>* max_it, std::optional<int>* seed, std::optional<float>* alpha_W,
    std::optional<float>* alpha_H, std::optional<float>* l1_ratio) 
{
    if (argc < 5 || argc > 17) {
        std::cerr 
            << "./exec dataInput.bin N M K [--tolerance 1e-4] [--max_iterations 200] [--alpha_W 0] [--alpha_H 0] [--l1_ratio 0] [--seed -1]" 
            << std::endl;
		//return 1;
	}

    strcpy(file_name, argv[1]);
	*N = atoi(argv[2]);
    *M = atoi(argv[3]);
    *K = atoi(argv[4]);

    //parse optional arguments
    for(int i{5} ; i < argc; i++) {
        if(std::string(argv[i]) == "--tolerance") {
            //std::cout << "Tol = " << std::string(argv[++i]) << std::endl;
            if(i + 1 < argc)  tolerance->emplace(std::stod(std::string(argv[++i]), nullptr));
            else              std::cerr << "--tolerance option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--max_iterations") {
            if(i + 1 < argc)  max_it->emplace(std::stoi(std::string(argv[++i]), nullptr));
            else              std::cerr << "--max_iterations option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--alpha_W") {
            if(i + 1 < argc)  alpha_W->emplace(std::stof(std::string(argv[++i]), nullptr));
            else              std::cerr << "--alpha_W option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--alpha_H") {
            if(i + 1 < argc)  alpha_H->emplace(std::stof(std::string(argv[++i]), nullptr));
            else              std::cerr << "--alpha_H option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--l1_ratio") {
            if(i + 1 < argc)  l1_ratio->emplace(std::stof(std::string(argv[++i]), nullptr));
            else              std::cerr << "--l1_ratio option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--seed") {
            if(i + 1 < argc)  seed->emplace(std::stoi(std::string(argv[++i]), nullptr));
            else              std::cerr << "--seed option requires one argument." << std::endl;
        }
        else
            std::cerr << "Argument \"" << std::string(argv[i]) << "\" not valid." << std::endl;
    }
}


C_REAL* get_V(int N, int M, char* file_name) {
	C_REAL *V = new C_REAL[N*M];

	FILE *fIn = fopen(file_name, "r");
	fread(V, sizeof(C_REAL), N*M, fIn);
	fclose(fIn);

	return V;
}


C_REAL* mul(int N, int M, int K, C_REAL* W, C_REAL* H) {
    C_REAL* V = new C_REAL[N*M];

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            for (size_t k = 0; k < K; k++)
                V[i*M + j] += W[i*K + k] * H[k*M + j];
    

    return V;
}


void print_matrix(C_REAL *m, int I, int J) {	
	std::cout << "--------------------- matrix --------------------" << std::endl;
	std::cout << "             ";
	for (int j = 0; j < J; j++) {
		if (j < 10)
			std::cout << j << "      ";
		else if (j < 100)
			std::cout << j << "     ";
		else 
			std::cout << j << "    ";
	}
	std::cout << std::endl;

	for (int i = 0; i < I; i++) {
		if (i<10)
			std::cout << "Line   " << i << ": ";
		else if (i<100)
			std::cout << "Line  " << i << ": ";
		else
			std::cout << "Line " << i << ": ";

		for (int j = 0; j < J; j++)
			std::cout << m[i*J + j] << " ";
		std::cout << std::endl;
	}
}


int main(int argc, char **argv) {
    int N, M, K;
    char file_name[255];
    std::optional<double> tolerance = std::nullopt;
    std::optional<int> max_it = std::nullopt;
    std::optional<int> seed = std::nullopt;
    std::optional<float> alpha_W = std::nullopt;
    std::optional<float> alpha_H = std::nullopt;
    std::optional<float> l1_ratio = std::nullopt;

    parse_arguments(argc, argv, &N, &M, &K, file_name, &tolerance, &max_it, &seed, &alpha_W, &alpha_H, &l1_ratio);

    C_REAL* V = get_V(N, M, file_name);
    
    NMF nmf = NMF(N, M, K, tolerance, max_it, seed, alpha_W, alpha_H, l1_ratio);

    double t_init = gettime();

    nmf.fit_transform(V, false);

    std::cout << "Total time = " << (gettime() - t_init) << " (us)" << std::endl;
    std::cout << "Final error = " << nmf.get_error() << std::endl;

    C_REAL* Vnew = mul(N, M, K, nmf.get_W(), nmf.get_H());
    std::cout << "Original V = " << std::endl;
    print_matrix(V, N, M);
    std::cout << "Approximation V = " << std::endl;
    print_matrix(Vnew, N, M);
    delete[] Vnew;

    delete[] V;
    return 0;
}