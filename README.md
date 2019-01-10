# MultivariateFunctions.jl

| Build | Coverage |
|-------|----------|
| [![Build Status](https://travis-ci.com/s-baumann/MultivariateFunctions.jl.svg?branch=master)](https://travis-ci.org/s-baumann/MultivariateFunctions.jl) | [![Coverage Status](https://coveralls.io/repos/github/s-baumann/MultivariateFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/s-baumann/MultivariateFunctions.jl?branch=master)

This implements single algebra and evaluation on Multivariate functions.
There are a few ways in which it can be used.
* This can be used for approximation functions. It can currently implement OLS functions, chebyshev polynomials, the schumaker shape preserving spline and basic interpolation schemes. It can also create Multivariate Adaptive Regression (MARS) Splines. It could be extended to implement other approximation schemes.
* As in the [StochasticIntegrals.jl](https://github.com/s-baumann/StochasticIntegrals.jl) package this package can be used to define functions that will be the integrands in stochastic integrals. This has the benefit that the means, variances & covariances implied by these stochastic integrals can be found analytically.
* All basic algebra and calculus on a MultivariateFunction can be done analytically.
* Newton's method is implemented so that roots and optima can be found using analytical Jacobians and Hessians.
