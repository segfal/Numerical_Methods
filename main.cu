#include <iostream>
#include "includes/RootFinding.hpp"

int main() {
    RootFinding root_finding(0.0001, 100, RootFindingMethod::NewtonRaphson);
    double root = root_finding.find_root(nullptr, 0, 1, nullptr, 1.0, 1.0);
    std::cout << "Root: " << root << std::endl;
    return 0;
} 