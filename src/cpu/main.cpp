#include <iostream>
#include "../../includes/RootFinding.hpp"

double f(double x) {
    return x * x - 2;
}

double f_prime(double x) {
    return 2 * x;
}

int main() {
    RootFinding root_finding(0.00001, 100, RootFindingMethod::NewtonRaphson);
    double newton_root = root_finding.find_root_newton(f, f_prime, 1.0);
    std::cout << "Newton Root: " << newton_root << std::endl;

    root_finding = RootFinding(0.00001, 100, RootFindingMethod::Bisection);
    double bisection_root = root_finding.find_root_bracket(f, 0, 1);
    std::cout << "Bisection Root: " << bisection_root << std::endl;

    root_finding = RootFinding(0.00001, 100, RootFindingMethod::FalsePosition);
    double false_position_root = root_finding.find_root_bracket(f, 0, 1);
    std::cout << "False Position Root: " << false_position_root << std::endl;

    return 0;
} 