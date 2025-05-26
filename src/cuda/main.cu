#include <iostream>
#include "../../includes/RootFinding.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>

double f(double x) {
    return x * x - 2;
}

double f_prime(double x) {
    return 2 * x;
}

int main() {
    RootFinding root_finding(0.00001, 100, RootFindingMethod::NewtonRaphson);
    double newton_root = root_finding.find_root_newton(f, f_prime, 1.0);
    std::cout << "Newton Root (CUDA): " << newton_root << std::endl;

    root_finding = RootFinding(0.00001, 100, RootFindingMethod::Bisection);
    double bisection_root = root_finding.find_root_bracket(f, 0, 1);
    std::cout << "Bisection Root (CUDA): " << bisection_root << std::endl;

    root_finding = RootFinding(0.00001, 100, RootFindingMethod::FalsePosition);
    double false_position_root = root_finding.find_root_bracket(f, 0, 1);
    std::cout << "False Position Root (CUDA): " << false_position_root << std::endl;

    return 0;
}
#endif 