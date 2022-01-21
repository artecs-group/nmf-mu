#include <iostream>
#include <vector>
#include <string>
#include "./nmf/nmf.hpp"

void parse_arguments(int argc, char **argv, char* file_name, std::optional<double> tolerance,
    std::optional<int> max_it, std::optional<int> seed, std::optional<float> alpha_W,
    std::optional<float> alpha_H, std::optional<float> l1_ratio) 
{
    if (argc < 5 || argc > 11) {
        std::cerr 
            << "./exec dataInput.bin N M K [--tolerance 1e-4] [--max_iterations 200] [--alpha_W 0] [--alpha_H 0] [--l1_ratio 0] [--seed -1]" 
            << std::endl;
		//return 1;
	}

    strcpy(file_name, argv[1]);
	int N = atoi(argv[2]);
    int M = atoi(argv[3]);
    int K = atoi(argv[4]);

    //parse optional arguments
    std::vector <std::string> sources;
    std::string destination;

    for(int i{5} ; i < argc; i++) {
        if(std::string(argv[i]) == "--tolerance") {
            if(i + 1 < argc)  tolerance.emplace(std::stod(argv[i++]));
            else              std::cerr << "--tolerance option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--max_iterations") {
            if(i + 1 < argc)  max_it.emplace(std::atoi(argv[i++]));
            else              std::cerr << "--max_iterations option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--alpha_W") {
            if(i + 1 < argc)  alpha_W.emplace(std::stof(argv[i++]));
            else              std::cerr << "--alpha_W option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--alpha_H") {
            if(i + 1 < argc)  alpha_H.emplace(std::stof(argv[i++]));
            else              std::cerr << "--alpha_H option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--l1_ratio") {
            if(i + 1 < argc)  l1_ratio.emplace(std::stof(argv[i++]));
            else              std::cerr << "--l1_ratio option requires one argument." << std::endl;
        }
        else if(std::string(argv[i]) == "--seed") {
            if(i + 1 < argc)  seed.emplace(std::atoi(argv[i++]));
            else              std::cerr << "--seed option requires one argument." << std::endl;
        }
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

    parse_arguments(argc, argv, file_name);


    //NMF nmf = NMF();
}