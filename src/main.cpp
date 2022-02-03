#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include "./common.hpp"
#include "./nmf/nmf.hpp"

template <typename T>
void print_matrix(T* m, int I, int J) {	
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
    std::cout << std::endl;
}


double gettime() {
	double final_time;
	struct timeval tv1;
	
	gettimeofday(&tv1, (struct timezone*)0);
	final_time = (tv1.tv_usec + (tv1.tv_sec)*1000000ULL);

	return final_time;
}


void parse_arguments(int argc, char **argv, int* N, int* M, int* K, char* file_name, std::optional<double>* tolerance,
    std::optional<int>* max_it, std::optional<int>* seed, std::optional<float>* alpha_W,
    std::optional<float>* alpha_H, std::optional<float>* l1_ratio, bool* verbose) 
{
    if (argc < 5 || argc > 17) {
        std::cerr 
            << "./exec dataInput.bin N M K [--tolerance 1e-4] [--max_iterations 200] [--alpha_W 0] [--alpha_H 0] [--l1_ratio 0] [--seed -1]" 
            << std::endl;
	}

    strcpy(file_name, argv[1]);
	*N = atoi(argv[2]);
    *M = atoi(argv[3]);
    *K = atoi(argv[4]);

    //parse optional arguments
    for(int i{5} ; i < argc; i++) {
        if(std::string(argv[i]) == "--tolerance")               tolerance->emplace(std::stod(std::string(argv[++i]), nullptr));
        else if(std::string(argv[i]) == "--max_iterations")     max_it->emplace(std::stoi(std::string(argv[++i]), nullptr));
        else if(std::string(argv[i]) == "--alpha_W")            alpha_W->emplace(std::stof(std::string(argv[++i]), nullptr));
        else if(std::string(argv[i]) == "--alpha_H")            alpha_H->emplace(std::stof(std::string(argv[++i]), nullptr));
        else if(std::string(argv[i]) == "--l1_ratio")           l1_ratio->emplace(std::stof(std::string(argv[++i]), nullptr));
        else if(std::string(argv[i]) == "--seed")               seed->emplace(std::stoi(std::string(argv[++i]), nullptr));
        else if(std::string(argv[i]) == "--verbose")            *verbose = true;
        else
            std::cerr << "Argument \"" << std::string(argv[i]) << "\" not valid." << std::endl;
    }
}


C_REAL* get_matrix(int N, int M, char* file_name) {
	C_REAL *Mat = new C_REAL[N*M];

	FILE *fIn = fopen(file_name, "rb");
	fread(Mat, sizeof(C_REAL), N*M, fIn);
	fclose(fIn);

	return Mat;
}


C_REAL* read_image(int N, int M, char* file_name) {
    C_REAL *out = new C_REAL[N*M];
    unsigned char pixel;
    int num_of_rows = 0, num_of_cols = 0, max_val;

    std::ifstream infile(file_name, std::ios::binary);
    std::stringstream ss;
    std::string inputLine = "";

    getline(infile, inputLine); // read the first line : P5
    if(inputLine.compare("P5") != 0) std::cerr << "Version error" << std::endl;
    ss << infile.rdbuf();   //read the third line : width and height
    ss >> num_of_cols >> num_of_rows;
    ss >> max_val;

    for(int i{0}; i < N*M; i++) {
        ss >> pixel;
        out[i] = (C_REAL)pixel;
    }

	infile.close();
	return out;
}


void write_image(int N, int M, char* file_name, C_REAL* Mat) {
    std::ofstream out_file(file_name, std::ios::binary);
    std::stringstream ss;
	unsigned char* pixels = new unsigned char[N*M];

    for(int i{0}; i < N*M; i++) {
        if(Mat[i] > 255)
            pixels[i] = 255;
        else
            pixels[i] = (unsigned char) Mat[i];
    }

    int max = *std::max_element(pixels, pixels+(N*M));

    ss << "P5" << std::endl
       << N << " " << M << std::endl
       << max << std::endl;

    out_file << ss.rdbuf();

	for(int i{0}; i < N*M; i++)
		out_file << pixels[i];

	out_file.close();
	delete[] pixels;
}


C_REAL* mul(int N, int M, int K, C_REAL* W, C_REAL* H) {
    C_REAL* V = new C_REAL[N*M] {0};

    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++)
            for (size_t k = 0; k < K; k++)
                V[i*M + j] += W[i*K + k] * H[k*M + j];
    
    return V;
}


int main(int argc, char **argv) {
    int N, M, K;
    bool verbose = false;
    char file_name[255];
    std::optional<double> tolerance = std::nullopt;
    std::optional<int> max_it = std::nullopt;
    std::optional<int> seed = std::nullopt;
    std::optional<float> alpha_W = std::nullopt;
    std::optional<float> alpha_H = std::nullopt;
    std::optional<float> l1_ratio = std::nullopt;

    parse_arguments(argc, argv, &N, &M, &K, file_name, &tolerance, &max_it, &seed, &alpha_W, &alpha_H, &l1_ratio, &verbose);

    C_REAL* V = read_image(N, M, file_name);
    // C_REAL* V = get_matrix(N, M, (char*)"V.bin");
    
    NMF nmf = NMF(N, M, K, tolerance, max_it, seed, alpha_W, alpha_H, l1_ratio, verbose);

    double t_init = gettime();

    nmf.fit_transform(V);

    std::cout << std::endl 
        << "Total time = " << (gettime() - t_init)/1000 << " (ms)" << std::endl 
        << "Final error = " << nmf.get_error() << std::endl
        << "Iterations = " << nmf.get_iterations() << std::endl;

    C_REAL* Vnew = mul(N, M, K, nmf.get_W(), nmf.get_H());
    write_image(N, M, (char*)"lena_grey_approx.pgm", Vnew);

    delete[] V;
    delete[] Vnew;
    return 0;
}