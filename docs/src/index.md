# MultivariateFunctions

This implements single algebra and evaluation on Multivariate functions.
There are a few ways in which it can be used.
* This can be used for approximation functions. It can currently implement OLS functions, chebyshev polynomials, the schumaker shape preserving spline and basic interpolation schemes. It can also create Multivariate Adaptive Regression (MARS) Splines. It could be extended to implement other approximation schemes.
* As in the [StochasticIntegrals.jl](https://github.com/s-baumann/StochasticIntegrals.jl) package this package can be used to define functions that will be the integrands in stochastic integrals. This has the benefit that the means, variances & covariances implied by these stochastic integrals can be found analytically.
* All basic algebra and calculus on a MultivariateFunction can be done analytically.
* The Newton's method is implemented so that roots and optima can be found using analytical Jacobians and Hessians.

```@contents
pages = ["index.md",
         "1_structs_and_limitations.md",
         "2_Interpolation_and_splines.md",
         "3_examples_algebra.md",
         "4_examples_interpolation.md",
         "5_examples_approximation.md",
         "99_refs.md"]
Depth = 2
```
