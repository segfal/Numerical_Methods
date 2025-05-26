#ifndef ROOT_FINDING_HPP
#define ROOT_FINDING_HPP

#include <iostream>
#include <cmath>
#include <functional>

enum class RootFindingMethod {
    Bisection,
    FalsePosition,
    NewtonRaphson,
    Secant,
    Steffensen,
};

class RootFinding {
    private:
        double tolerance;
        int max_iterations;
        RootFindingMethod method;

    public:
        RootFinding(double tolerance, int max_iterations, RootFindingMethod method);
        
        // Overloaded find_root functions for different methods
        double find_root_bracket(std::function<double(double)> f, double a, double b);  // For Bisection and False Position
        double find_root_newton(std::function<double(double)> f, std::function<double(double)> f_prime, double x0);  // For Newton-Raphson
        double find_root_secant(std::function<double(double)> f, double x0, double x1);  // For Secant
        double find_root_steffensen(std::function<double(double)> f, double x0);  // For Steffensen
};

#endif 