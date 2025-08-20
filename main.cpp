#include <iostream>
#include "includes/RootFinding.hpp"
// TODO: 
/**
 * CREATE A CUDA VERSION OF THIS FILE
 * 
 */



double f(double x) {
    return x * x - 2;
}

double f_prime(double x) {
    return 2 * x;
}

int main() {
    RootFinding root_finding(0.0001, 100, RootFindingMethod::NewtonRaphson);
    double root = root_finding.find_root(f, 0, 1, f_prime, 1.0, 1.0);
    std::cout << "Root: " << root << std::endl;
    return 0;
}