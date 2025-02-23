#!/usr/bin/env python3
import time

import numpy as np
import scipy.stats as st
from scipy.stats import norm, norminvgauss

import nig


def average_time(func, *args, repeats=2):
    """Call func(*args) repeatedly and return the average time and last result."""
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times), result


def main():
    # Parameters for the NIG distribution.
    a = 3
    b = 1.5
    loc = 0.0
    scale = 1.2
    spline_points = 500

    # Create our NIG distribution instance.
    dist = nig.NIG(a, b, loc, scale, spline_points)

    # Create a corresponding frozen SciPy norminvgauss distribution.
    sp_dist = st.norminvgauss(a, b, loc=loc, scale=scale)

    repeats = 20

    # ----------------------------
    # Benchmark PDF computation
    # ----------------------------
    x_pdf = np.linspace(-5, 5, 5_000_000)

    sp_pdf_time, _ = average_time(sp_dist.pdf, x_pdf, repeats=repeats)
    cpp_pdf_time, _ = average_time(dist.pdf, x_pdf, repeats=repeats)
    speedup_pdf = sp_pdf_time / cpp_pdf_time if cpp_pdf_time > 0 else float("inf")

    print("PDF Benchmark:")
    print(f"  SciPy norminvgauss pdf average time: {sp_pdf_time:.6f} sec")
    print(f"  C++ NIG pdf average time:            {cpp_pdf_time:.6f} sec")
    print(f"  Speedup (SciPy / C++):               {speedup_pdf:.2f}x\n")

    # ----------------------------
    # Benchmark CDF computation
    # ----------------------------
    x_cdf = np.linspace(-5, 5, 1_00)

    sp_cdf_time, _ = average_time(sp_dist.cdf, x_cdf, repeats=repeats)
    cpp_cdf_time, _ = average_time(dist.cdf, x_cdf, repeats=repeats)
    speedup_cdf = sp_cdf_time / cpp_cdf_time if cpp_cdf_time > 0 else float("inf")

    print("CDF Benchmark:")
    print(f"  SciPy norminvgauss cdf average time: {sp_cdf_time:.6f} sec")
    print(f"  C++ NIG cdf average time:            {cpp_cdf_time:.6f} sec")
    print(f"  Speedup (SciPy / C++):               {speedup_cdf:.2f}x\n")

    # ----------------------------
    # Benchmark PPF computation
    # ----------------------------
    q = np.linspace(0.0001, 0.9999, 10)

    sp_ppf_time, _ = average_time(sp_dist.ppf, q, repeats=repeats)
    cpp_ppf_time, _ = average_time(dist.ppf, q, repeats=repeats)
    speedup_ppf = sp_ppf_time / cpp_ppf_time if cpp_ppf_time > 0 else float("inf")

    print("PPF Benchmark:")
    print(f"  SciPy norminvgauss ppf average time:          {sp_ppf_time:.6f} sec")
    print(f"  C++ NIG ppf average time:                     {cpp_ppf_time:.6f} sec")
    print(f"  Speedup (SciPy / C++):                        {speedup_ppf:.2f}x\n")

    # ----------------------------
    # Benchmark nig_values_from_normal_values Mapping
    # ----------------------------
    spline_eval_points = 10_000_000
    manual_eval_points = 1_000
    x_normal = np.random.uniform(-5, 5, manual_eval_points)
    x_huge = np.random.uniform(-5, 5, spline_eval_points)
    # Warm up the nig_values_from_normal_values (to initialize the spline, etc.).
    dist.nig_values_from_normal_values(np.array([0]))

    fast_repeats = repeats * 5

    nig_values_from_normal_values_time, _ = average_time(
        dist.nig_values_from_normal_values, x_huge, repeats=fast_repeats
    )
    ppf_from_norm_time, _ = average_time(
        lambda x: dist.ppf(norm.cdf(x)), x_normal, repeats=fast_repeats
    )
    speedup_nig_values_from_normal_values = (
        ppf_from_norm_time / nig_values_from_normal_values_time
        if nig_values_from_normal_values_time > 0
        else float("inf")
    )

    print("nig_values_from_normal_values Map Benchmark:")
    print(
        f"  C++ NIG nig_values_from_normal_values average time:                 {nig_values_from_normal_values_time:.6f} sec"
    )
    print(
        f"  C++ NIG ppf(norm.cdf(x)) average time:                              {ppf_from_norm_time:.6f} sec"
    )
    print(
        f"  Speedup (ppf(norm.cdf(x)) / nig_values_from_normal_values_map):     {speedup_nig_values_from_normal_values * spline_eval_points / manual_eval_points:.2f}x\n"
    )


if __name__ == "__main__":
    main()
