# MultivariateFunctions.jl

Documentation
[![docs-latest-img](https://img.shields.io/badge/docs-latest-blue.svg)](https://s-baumann.github.io/MultivariateFunctions.jl/dev/index.html) |

This implements single algebra and evaluation on Multivariate functions.
There are a few ways in which it can be used.
* This can be used for approximation functions. It can currently implement OLS functions, Chebyshev polynomials, the Schumaker shape preserving spline and basic interpolation schemes. It can also do Recursive Partitioning and create Multivariate Adaptive Regression (MARS) Splines. It could be extended to implement other approximation schemes.
* All basic algebra and calculus on a MultivariateFunction can be done analytically.
* Newton's method is implemented so that roots and optima can be found using analytical Jacobians and Hessians.

This was written for a pre 1.0 version of Julia and no longer runs given changes in the language.
