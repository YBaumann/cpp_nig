import os
import sys

import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# I am on MacOs, not sure if you need something different here:
boost_include_dir = "/opt/homebrew/opt/boost/include"

ext_modules = [
    Extension(
        "nig",
        ["nig.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            boost_include_dir,  # If BOOST_INCLUDE is not set, this will be ignored.
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="nig",
    version="0.1.0",
    author="Your Name",
    author_email="baumann@swissquant.com",
    description="NIG distribution with PPF approximations using cubic spline and pybind11",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.6.0", "numpy"],
    install_requires=["pybind11>=2.6.0", "numpy"],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
