#ifndef M_PI
#define M_PI 3.14159265358979323846
#define inv_sqrt2 1. / std::sqrt(2.0)
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>
#include <memory>
#include <omp.h>


namespace py = pybind11;

// Helper: Standard Normal CDF using the complementary error function.
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * inv_sqrt2);
}

class CubicSpline {
public:
    CubicSpline(py::array_t<double> x_array, py::array_t<double> y_array) {

        auto buf_x = x_array.request();
        auto buf_y = y_array.request();
        if (buf_x.ndim != 1 || buf_y.ndim != 1)
            throw std::runtime_error("x and y arrays must be one-dimensional");
        if (buf_x.shape[0] != buf_y.shape[0])
            throw std::runtime_error("x and y arrays must have the same length");
        n = buf_x.shape[0];
        if (n < 2)
            throw std::runtime_error("At least two data points are required for spline interpolation");

        // Copy nodes and function values.
        x.resize(n);
        a.resize(n);
        auto ptr_x = static_cast<double*>(buf_x.ptr);
        auto ptr_y = static_cast<double*>(buf_y.ptr);

        for (size_t i = 0; i < n; i++) {
            x[i] = ptr_x[i];
            a[i] = ptr_y[i];
        }

        x_front = x[0];
        x_back = x[n-1];
        a_front = a[0];
        a_back = a[n-1];

        // Because we assume evenly spaced x, compute and store spacing.
        spacing = x[1] - x[0];
        _inv_spacing = 1.0 / spacing;
        for (size_t i = 1; i < n - 1; i++) {
            double current_spacing = x[i+1] - x[i];
            if (std::abs(current_spacing - spacing) > 1e-2)
                throw std::runtime_error("x array is not evenly spaced");
        }
        // Compute interval widths h[i] = spacing (constant)
        std::vector<double> h(n - 1, spacing);

        // Estimate the endpoint derivatives using one-sided differences.
        double fprime0 = (a[1] - a[0]) / spacing;
        double fprimen = (a[n-1] - a[n-2]) / spacing;

        // Build the alpha vector.
        std::vector<double> alpha(n, 0.0);
        alpha[0] = 3.0 * ((a[1] - a[0]) / spacing - fprime0);
        alpha[n-1] = 3.0 * (fprimen - (a[n-1] - a[n-2]) / spacing);
        for (size_t i = 1; i < n - 1; i++) {
            alpha[i] = 3.0/spacing * (a[i+1] - a[i]) - 3.0/spacing * (a[i] - a[i-1]);
        }

        std::vector<double> l(n, 0.0), mu(n, 0.0), z(n, 0.0);
        l[0] = 2.0 * spacing;
        mu[0] = 0.5;
        z[0] = alpha[0] / l[0];
        for (size_t i = 1; i < n - 1; i++) {
            l[i] = 2.0 * (x[i+1] - x[i-1]) - spacing * mu[i-1];
            mu[i] = spacing / l[i];
            z[i] = (alpha[i] - spacing * z[i-1]) / l[i];
        }
        l[n-1] = spacing * (2 - mu[n-2]);
        z[n-1] = (alpha[n-1] - spacing * z[n-2]) / l[n-1];

        c.resize(n, 0.0);
        b.resize(n - 1, 0.0);
        d.resize(n - 1, 0.0);
        c[n-1] = z[n-1];
        // Back substitution.
        for (int j = n - 2; j >= 0; j--) {
            c[j] = z[j] - mu[j] * c[j+1];
            b[j] = (a[j+1] - a[j]) / spacing - spacing * (c[j+1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j+1] - c[j]) / (3.0 * spacing);
        }
    }

    double operator()(double x_val) const {
        if (x_val <= x_front)
            return a_front;
        if (x_val >= x_back)
            return a_back;

        // Since x is evenly spaced, compute the index directly.
        size_t low = static_cast<size_t>((x_val - x_front) * _inv_spacing);
        if (low >= n - 1)
            low = n - 2;
        double dx = x_val - x[low];
        return a[low] + (b[low] + (c[low] + d[low] * dx) * dx) * dx;
    }

private:
    std::vector<double> x;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> d;
    double x_front;
    double x_back;
    double a_front;
    double a_back;
    double spacing;
    double _inv_spacing;
    size_t n;
};



class NIG {
public:
    // Parameters: (alpha, beta, loc, scale)
    double a, b, loc, scale;
    size_t spline_points;

    NIG(double a_ = 1.5, double b_ = 0.5, double loc_ = 0.0, double scale_ = 1.0, size_t spline_points_ = 200)
        : a(a_), b(b_), loc(loc_), scale(scale_), spline_points(spline_points_), spline_initialized(false) {
            _exp_sqrt_a2_b2 = std::exp(std::sqrt(a*a-b*b));
            _inv_scale = 1./scale;
            int numProcs = omp_get_num_procs();
            int maxThreads = omp_get_max_threads();
            std::cout << "NIG is using: " << numProcs << " Processors and " << maxThreads << " Threads." << std::endl; 
        }

    // Compute the PDF for a 1-D NumPy array of x values.
    py::array_t<double> pdf(py::array_t<double> input_array) const {
        auto buf = input_array.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Input should be a 1-D NumPy array");
        size_t n = buf.shape[0];
        auto result = py::array_t<double>(n);
        auto r_in = input_array.unchecked<1>();
        auto r_out = result.mutable_unchecked<1>();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++){
            double x_val = r_in(i);
            r_out(i) = _pdf_single(x_val);
        }
        return result;
    }

    py::array_t<double> cdf(py::array_t<double> input_array) const {
        auto buf = input_array.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Input should be a 1-D NumPy array");
        size_t n = buf.shape[0];
        auto result = py::array_t<double>(n);
        auto r_in = input_array.unchecked<1>();
        auto r_out = result.mutable_unchecked<1>();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++){
            double x_val = r_in(i);
            r_out(i) = _cdf_single(x_val);
        }
        return result;
    }

    py::array_t<double> ppf(py::array_t<double> input_array) const {
        auto buf = input_array.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Input should be a 1-D NumPy array");
        size_t n = buf.shape[0];
        auto result = py::array_t<double>(n);
        auto r_in = input_array.unchecked<1>();
        auto r_out = result.mutable_unchecked<1>();

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++){
            double q = r_in(i);
            r_out(i) = _ppf_single(q);
        }
        return result;
    }

    py::array_t<double> nig_values_from_normal_values(py::array_t<double> input_array) const {
        auto buf = input_array.request();
        if (buf.ndim != 1)
            throw std::runtime_error("Input should be a 1-D NumPy array");
        size_t n = buf.shape[0];
        auto result = py::array_t<double>(n);
        auto r_in = input_array.unchecked<1>();
        auto r_out = result.mutable_unchecked<1>();

        // Lazy initialization of the cubic spline approximation.
        if (!spline_initialized) {
            build_ppf_spline();
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++){
            double x_val = r_in(i);
            r_out(i) = (*ppf_spline)(x_val);
        }
        return result;
    }

private:
    // Precomputed values for pdf
    double _exp_sqrt_a2_b2;
    double _inv_scale;

    // Predefined integrator
    using TanhSinh = boost::math::quadrature::tanh_sinh<double>;
    mutable TanhSinh integrator;


    double _pdf_single(double x) const {
        double y = (x - loc) * _inv_scale;
        double sqrt_one_plus_y2 = std::sqrt(1 + y*y);
        double right_factor = _exp_sqrt_a2_b2 * std::exp(b*y);
        double left_factor = a * boost::math::cyl_bessel_k(1, a * sqrt_one_plus_y2) / (M_PI * sqrt_one_plus_y2);
        double res = left_factor * right_factor;
        return res * _inv_scale;
    }

    double _cdf_single(double x) const {
        auto integrand = [this](double t) -> double {
            double val = _pdf_single(t);
            return std::isfinite(val) ? val : 0.;
        };
        double tol = 1e-12;
        double lower_limit = -60;
        double upper_limit = 60;
        double result = 0;
        if (x < loc) {
            result = integrator.integrate(integrand, lower_limit, x, tol);
        } else {
            result = 1 - integrator.integrate(integrand, x, upper_limit, tol);
        }
        return result;
    }

    double _ppf_single(double q) const {
        auto f = [this, q](double x) -> double {
            return _cdf_single(x) - q;
        };
        double L = loc - 100 * scale;
        double U = loc + 100 * scale;
        int iter = 0;
        const int max_iter = 100;
        while (f(L) > 0 && iter < max_iter) {
            L -= 100 * scale;
            ++iter;
        }
        if (iter == max_iter)
            throw std::runtime_error("Failed to find a suitable lower bound for PPF computation.");
        iter = 0;
        while (f(U) < 0 && iter < max_iter) {
            U += 100 * scale;
            ++iter;
        }
        if (iter == max_iter)
            throw std::runtime_error("Failed to find a suitable upper bound for PPF computation.");
        auto tol = boost::math::tools::eps_tolerance<double>(30);
        auto r = boost::math::tools::bisect(f, L, U, tol);
        double root = (r.first + r.second) / 2.0;
        return root;
    }

    void build_ppf_spline() const {
        double start = -5;
        double end = 5;
        std::vector<double> q_vals(spline_points);
        std::vector<double> ppf_vals(spline_points);
        double step = (end - start) / static_cast<double>(spline_points - 1);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < spline_points; i++){
            double x_val = start + i * step;
            double u = norm_cdf(x_val);
            ppf_vals[i] = _ppf_single(u);
            q_vals[i] = x_val;
        }

        py::array_t<double> q_array(q_vals.size(), q_vals.data());
        py::array_t<double> ppf_array(ppf_vals.size(), ppf_vals.data());

        ppf_spline = std::make_unique<CubicSpline>(q_array, ppf_array);
        spline_initialized = true;
    }

    mutable std::unique_ptr<CubicSpline> ppf_spline;
    mutable bool spline_initialized;
};

PYBIND11_MODULE(nig, m) {
    m.doc() = "Module that implements the NIG distribution with two PPF functions: "
              "Additionally, nig_values_from_normal_values(x) computes nig.ppf(norm.cdf(x)) to map normal variables to NIG.";
    py::class_<NIG>(m, "NIG")
        .def(py::init<double, double, double, double, size_t>(),
             py::arg("a") = 1.5,
             py::arg("b") = 0.5,
             py::arg("loc") = 0.0,
             py::arg("scale") = 1.0,
             py::arg("spline_points") = 200)
        .def("pdf", &NIG::pdf,
             "Compute the NIG pdf for each element in the provided 1-D NumPy array")
        .def("cdf", &NIG::cdf,
             "Compute the NIG cdf for each element in the provided 1-D NumPy array")
        .def("ppf", &NIG::ppf,
             "Compute the NIG ppf (inverse cdf) for each element in the provided 1-D NumPy array using a cubic spline approximation")
        .def("nig_values_from_normal_values", &NIG::nig_values_from_normal_values,
             "Given an array of values from a normal variable, map them to NIG quantiles via "
             "y = nig.ppf(norm.cdf(x)).");
}
