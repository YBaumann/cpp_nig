import time

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm, norminvgauss

import nig

PDF_VALUES = 10_000_000
CDF_VALUES = 1_000
PPF_VALUES = 100
UNIFORM_BOUNDS = 20

# Define several sets of parameters to test.
NIG_PARAMS = [
    (12, 4, 1.0, 10),  # original example
    (10, 5, 0.0, 5),  # original example
    (8, 2, -1.0, 1),  # original example
    (15, 6, 2.0, 8),  # original example
    (5, 1, -2.0, 2),  # original example
    (1, 0.5, 0.0, 1),  # small parameters, standard scale
    (20, 10, 5.0, 15),  # larger parameters and wider spread
    (7, 3, -5.0, 0.5),  # negative location and small scale
    (9, 1, 0.0, 0.2),  # extremely small scale for high precision
    (10, 0, 0.0, 1),  # symmetric case (B=0)
]

SPLINE_POINTS = 10_000


# This fixture creates both the C++ and SciPy versions for each parameter set.
@pytest.fixture(params=NIG_PARAMS)
def nig_pair(request):
    A, B, LOC, SCALE = request.param
    cpp_nig = nig.NIG(A, B, LOC, SCALE, SPLINE_POINTS)
    scipy_nig = norminvgauss(A, B, LOC, SCALE)
    return (A, B, LOC, SCALE, cpp_nig, scipy_nig)


def test_pdf(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, SCIPY_NIG = nig_pair
    xx_values = np.linspace(-UNIFORM_BOUNDS, UNIFORM_BOUNDS, PDF_VALUES)
    scipy_pdf_values = SCIPY_NIG.pdf(xx_values)
    cpp_pdf_values = CPP_NIG.pdf(xx_values)
    assert np.allclose(
        cpp_pdf_values, scipy_pdf_values, atol=1e-12
    ), f"PDF values do not match for parameters {(A, B, LOC, SCALE)}!"


def test_cdf_value_comparison(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, SCIPY_NIG = nig_pair
    xx_values = np.linspace(-UNIFORM_BOUNDS, UNIFORM_BOUNDS, CDF_VALUES)
    cpp_arr = CPP_NIG.cdf(xx_values)
    scipy_arr = SCIPY_NIG.cdf(xx_values)
    indices = np.where(scipy_arr > 1)[0]  # SCIPY CDF fails at some points
    mask = np.ones(scipy_arr.shape[0], bool)
    if indices.size > 0:
        mask[indices[0] :] = False
    assert np.allclose(
        cpp_arr[mask], scipy_arr[mask], atol=1e-5
    ), f"CDF value comparison failed for parameters {(A, B, LOC, SCALE)}!"


def test_cdf_value_right_tail(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, SCIPY_NIG = nig_pair
    # Testing only the right tail: from UNIFORM_BOUNDS//5 to UNIFORM_BOUNDS.
    xx_values = np.linspace(UNIFORM_BOUNDS // 10, UNIFORM_BOUNDS, CDF_VALUES)
    cpp_arr = CPP_NIG.cdf(xx_values)
    scipy_arr = SCIPY_NIG.cdf(xx_values)
    indices = np.where(scipy_arr > 1)[0]
    mask = np.ones(scipy_arr.shape[0], bool)
    if indices.size > 0:
        mask[indices[0] :] = False
    assert np.allclose(
        cpp_arr[mask], scipy_arr[mask], atol=1e-8
    ), f"Right tail CDF value comparison failed for parameters {(A, B, LOC, SCALE)}!"


def test_cdf_monotonic_increasing(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, _ = nig_pair
    xx_values = np.linspace(-UNIFORM_BOUNDS, UNIFORM_BOUNDS, CDF_VALUES)
    cpp_arr = CPP_NIG.cdf(xx_values)
    # Check that the CDF is monotonic increasing (allowing for tiny numerical differences).
    diffs = np.diff(cpp_arr)
    assert np.all(
        diffs >= -1e-12
    ), f"CDF is not monotonic increasing for parameters {(A, B, LOC, SCALE)}!"


def test_ppf_cdf_bijection(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, _ = nig_pair
    xx_values = np.linspace(1e-6, 1 - 1e-6, PPF_VALUES)
    ppf_values = CPP_NIG.ppf(xx_values)
    cdf_values = CPP_NIG.cdf(ppf_values)
    assert np.allclose(
        xx_values, cdf_values, atol=1e-8
    ), f"PPF-CDF bijection failed for parameters {(A, B, LOC, SCALE)}!"


def test_ppf_value_comparison(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, SCIPY_NIG = nig_pair
    xx_values = np.linspace(1e-6, 1 - 1e-6, PPF_VALUES)
    cpp_ppf = CPP_NIG.ppf(xx_values)
    try:
        scipy_ppf = SCIPY_NIG.ppf(xx_values)
    except Exception as e:
        print(f"SCIPY ppf failed for nig: {nig_pair}.")
        return
    assert np.allclose(
        cpp_ppf, scipy_ppf, atol=1e-8
    ), f"PPF value comparison failed for parameters {(A, B, LOC, SCALE)}!"


def test_ppf_monotonic_increasing(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, _ = nig_pair
    xx_values = np.linspace(0, 1, 10 * PPF_VALUES)
    ppf_values = CPP_NIG.ppf(xx_values)
    diffs = np.diff(ppf_values)
    assert np.all(
        diffs >= -1e-12
    ), f"PPF is not monotonic increasing for parameters {(A, B, LOC, SCALE)}!"


def test_nig_values_from_normal_values_vs_ppf(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, SCIPY_NIG = nig_pair
    x_values = np.linspace(-5, 5, 10_000)
    expected = CPP_NIG.ppf(norm.cdf(x_values))
    result = CPP_NIG.nig_values_from_normal_values(x_values)
    assert np.allclose(
        result, expected, atol=1e-7
    ), f"nig_values_from_normal_values vs ppf(norm.cdf(x)) mismatch for parameters {(A, B, LOC, SCALE)}!"


def test_nig_values_from_normal_values_monotonic_increasing(nig_pair):
    A, B, LOC, SCALE, CPP_NIG, SCIPY_NIG = nig_pair
    x_values = np.linspace(-5, 5, 1_000_000)
    result = CPP_NIG.nig_values_from_normal_values(x_values)
    diffs = np.diff(result)
    assert np.all(
        diffs >= 1e-8
    ), f"nig_values_from_normal_values result is not monotonic increasing for parameters {(A, B, LOC, SCALE)}!"
