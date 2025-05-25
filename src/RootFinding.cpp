#include "../includes/RootFinding.hpp"

RootFinding::RootFinding(double tolerance, int max_iterations, RootFindingMethod method) 
    : tolerance(tolerance), max_iterations(max_iterations), method(method) {}

double RootFinding::find_root(std::function<double(double)> f, double a, double b, 
                            std::function<double(double)> f_prime, double x0, double x1) {
    switch (method) {
        case RootFindingMethod::Bisection: {
            double c;
            for (int i = 0; i < max_iterations; i++) {
                c = (a + b) / 2;
                if (fabs(f(c)) < tolerance) return c;
                if (f(c) * f(a) < 0) b = c;
                else a = c;
            }
            return c;
        }
        case RootFindingMethod::NewtonRaphson: {
            double x = x0;
            for (int i = 0; i < max_iterations; i++) {
                double h = f(x) / f_prime(x);
                x = x - h;
                if (fabs(h) < tolerance) return x;
            }
            return x;
        }
        // Add other methods as needed
        default:
            throw std::invalid_argument("Method not implemented");
    }
} 