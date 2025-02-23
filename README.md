# NIG Distribution with Cubic Spline Approximation

This repository implements the Normal Inverse Gaussian (NIG) distribution in C++ using pybind11, Boost and OpenMP. It includes a specialized function for mapping standard normal values to NIG quantiles using a cubic spline approximation.

A small script to evaluate the time spent for the different computations is provided. The function `nig_values_from_normal_values` processes 1 billion values in approximately 1.2 seconds.

Make sure to change the path to boost and OpenMP in your `setup.py`!!

## Files

- **nig.cpp**  
  Implements the NIG distribution in C++ with:
  - A `CubicSpline` class that assumes evenly spaced nodes for fast evaluation.
  - A `NIG` class with methods for:
    - `pdf`: Compute the probability density function.
    - `cdf`: Compute the cumulative distribution function.
    - `ppf`: Compute the inverse CDF using a cubic spline approximation.
    - `nig_values_from_normal_values`: Maps standard normal values to NIG quantiles by computing `nig.ppf(norm.cdf(x))`.

- **setup.py**  
  The build script for compiling the C++ extension using pybind11. It also configures Boost include paths.

- **run_timing.py**  
  A benchmarking script that:
  - Measures the performance of the pdf, cdf, and ppf routines compared to SciPy’s `norminvgauss`.
  - Benchmarks the copula mapping function (`nig_values_from_normal_values`) against the equivalent operation `ppf(norm.cdf(x))`.
  - Repeats measurements multiple times (at least 10 per test) and reports both average timings and speedup ratios.

- **tests/test_nig.py**  
  A set of pytest-based tests verifying that:
  - The C++ implementation’s pdf, cdf, and ppf match those from SciPy’s `norminvgauss` within acceptable tolerances.
  - The copula mapping function (`nig_values_from_normal_values`) produces equivalent results to `ppf(norm.cdf(x))` and is monotonic.

## Timing

All benchmarks were run on a MacOS machine with 8 cores and 8 threads.

### Overview

- **PDF Function:**  
  Achieves roughly a **4x speedup** compared to SciPy's implementation (varying with NIG parameters).

- **CDF and PPF Functions:**  
  Enjoy a dramatic **30x–50x speedup**, again depending on parameter settings.

- **Cubic Spline PPF Evaluation:**  
  After a one-time initialization, evaluating the PPF for 1 billion values takes approximately **1.2 seconds**. This is about **400,000x faster** than our C++ NIG PPF implementation and around **1.6 million times faster** than using SciPy's `norminvgauss`.

### Detailed Benchmark Summary

**Environment:**  
- **Processors/Threads:** 8

| Function                            | Benchmark (values)                                          | SciPy Average Time | C++ NIG Average Time | Speedup          |
|-------------------------------------|-------------------------------------------------------------|--------------------|----------------------|------------------|
| **PDF**                           | 10,000                                                     | 0.000656 sec       | 0.000173 sec         | **3.79x**        |
| **CDF**                           | 1,000                                                      | 0.542046 sec       | 0.024670 sec         | **21.97x**       |
| **PPF**                           | 100                                                        | 0.959331 sec       | 0.033139 sec         | **28.95x**       |
| **nig_values_from_normal_values** | Cubic spline on 10M values, standard on 1K values            | —                  | —                    | **39.57x** (comparing ppf(norm.cdf(x)) to nig_values_from_normal_values_map) |

*Note: The speedup factors are derived by comparing the average execution times between SciPy and the C++ NIG implementations.*
