#include "../../includes/RootFinding.hpp"

RootFinding::RootFinding(double tolerance, int max_iterations, RootFindingMethod method) 
    : tolerance(tolerance), max_iterations(max_iterations), method(method) {}

double RootFinding::find_root_bracket(std::function<double(double)> f, double a, double b) {
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
        case RootFindingMethod::FalsePosition: {
            double c;
            for (int i = 0; i < max_iterations; i++) {
                c = b - f(b) * (b - a) / (f(b) - f(a));
                if (fabs(f(c)) < tolerance) return c;
                if (f(c) * f(a) < 0) b = c;
                else a = c;
            }
            return c;
        }
        default:
            throw std::invalid_argument("Invalid method for this signature");
    }
}

double RootFinding::find_root_newton(std::function<double(double)> f, std::function<double(double)> f_prime, double x0) {
    if (method != RootFindingMethod::NewtonRaphson) {
        throw std::invalid_argument("Invalid method for this signature");
    }
    
    double x = x0;
    for (int i = 0; i < max_iterations; i++) {
        double h = f(x) / f_prime(x);
        x = x - h;
        if (fabs(h) < tolerance) return x;
    }
    return x;
}

double RootFinding::find_root_secant(std::function<double(double)> f, double x0, double x1) {
    if (method != RootFindingMethod::Secant) {
        throw std::invalid_argument("Invalid method for this signature");
    }
    
    double x = x1;
    double x_prev = x0;
    for (int i = 0; i < max_iterations; i++) {
        double x_next = x - f(x) * (x - x_prev) / (f(x) - f(x_prev));
        if (fabs(x_next - x) < tolerance) return x_next;
        x_prev = x;
        x = x_next;
    }
    return x;
}

double RootFinding::find_root_steffensen(std::function<double(double)> f, double x0) {
    if (method != RootFindingMethod::Steffensen) {
        throw std::invalid_argument("Invalid method for this signature");
    }
    
    double x = x0;
    for (int i = 0; i < max_iterations; i++) {
        double fx = f(x);
        double gx = f(x + fx) / fx - 1;
        double x_next = x - fx / gx;
        if (fabs(x_next - x) < tolerance) return x_next;
        x = x_next;
    }
    return x;
} 