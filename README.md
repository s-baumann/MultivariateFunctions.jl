# MultivariateFunctions.jl

| Build | Coverage | Documentation |
|-------|----------|---------------|
| [![Build status](https://github.com/s-baumann/MultivariateFunctions.jl/workflows/CI/badge.svg)](https://github.com/s-baumann/MultivariateFunctions.jl/actions) | [![codecov](https://codecov.io/gh/s-baumann/MultivariateFunctions.jl/branch/main/graph/badge.svg?token=sElLVJgRel)](https://codecov.io/gh/s-baumann/MultivariateFunctions.jl) | [![docs-latest-img](https://img.shields.io/badge/docs-latest-blue.svg)](https://s-baumann.github.io/MultivariateFunctions.jl/dev/index.html) |

This implements single algebra and evaluation on Multivariate functions.
There are a few ways in which it can be used.
* This can be used for approximation functions. It can currently implement OLS functions, Chebyshev polynomials, the Schumaker shape preserving spline and basic interpolation schemes. It can also do Recursive Partitioning and create Multivariate Adaptive Regression (MARS) Splines. It could be extended to implement other approximation schemes.
* All basic algebra and calculus on a MultivariateFunction can be done analytically.
* Newton's method is implemented so that roots and optima can be found using analytical Jacobians and Hessians.
