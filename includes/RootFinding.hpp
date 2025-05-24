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

/// Bisection Method
struct BisectionMethod {
    double find_root(std::function<double(double)> f, double a, double b) {
        double c = (a + b) / 2;
        return c;
    }
};

/// False Position Method
struct FalsePositionMethod {
    double find_root(std::function<double(double)> f, double a, double b) {
        double c = (a + b) / 2;
        return c;
    }
};

/// Newton Raphson Method
struct NewtonRaphsonMethod {
    double find_root(std::function<double(double)> f, std::function<double(double)> f_prime, double x0) {
        double c = x0 - f(x0) / f_prime(x0);
        return c;
    }
};

/// Secant Method
struct SecantMethod {
    double find_root(std::function<double(double)> f, double x0, double x1) {
        double c = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0));
        return c;
    }
};

/// Steffensen Method
struct SteffensenMethod {
    double find_root(std::function<double(double)> f, double x0) {
        double c = x0 - f(x0) / f(x0);
        return c;
    }
};







class RootFinding {
    private:
        double tolerance;
        int max_iterations;
        RootFindingMethod method;

    public:
        RootFinding(double tolerance, int max_iterations, RootFindingMethod method);
        double find_root(std::function<double(double)> f, double a, double b, std::function<double(double)> f_prime, double x0, double x1);
};


RootFinding::RootFinding(double tolerance, int max_iterations, RootFindingMethod method) : tolerance(tolerance), max_iterations(max_iterations), method(method) {}

double RootFinding::find_root(std::function<double(double)> f, double a, double b, std::function<double(double)> f_prime, double x0, double x1 ) {
    switch (method) {
        case RootFindingMethod::Bisection:
            return BisectionMethod().find_root(f, a, b);
        case RootFindingMethod::FalsePosition:
            return FalsePositionMethod().find_root(f, a, b);
        case RootFindingMethod::NewtonRaphson:
            return NewtonRaphsonMethod().find_root(f, f_prime, x0);
        case RootFindingMethod::Secant:
            return SecantMethod().find_root(f, x0, x1);
        case RootFindingMethod::Steffensen:
            return SteffensenMethod().find_root(f, x0);
        default:
            throw std::invalid_argument("Invalid method");
    }
}


#endif