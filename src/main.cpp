#include <iostream>
#include "nmf/nmf.hpp"

void parse_arguments(int argc, char **argv, char* file_name) {
    if (argc < 5 || argc > 11) {
        std::cout << "./exec dataInput.bin N M K [--tolerance=1e-4] [--max_iterations=200] [--alpha_W=0] [--alpha_H=0] [--l1_ratio=0] [--seed=]" 
            << std::endl;
		return 1;
	}

    strcpy(file_name, argv[1]);
	int N = atoi(argv[2]);
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);
}

int main(int argc, char **argv) {
    char file_name[255];
    parse_arguments(argc, argv, file_name);


    NMF nmf = NMF();
}