# Supported Approximation Methods

In addition the following approximation schemes are available, each of which can be used in any number of dimensions (subject to having enough computational power)
* OLS regression - Performs an OLS regression of the data and generates a Sum\_Of\_Functions containing the resultant approximation. This should work well in many dimensions.
* Chebyshev polynomials - Creates a Sum\_Of\_Functions that uses Chebyshev polynomials to approximate a certain function. Unlike the other approximation schemes this does not take in an arbitrary collection of datapoints but rather takes in a function that it evaluates at certain points in a grid to make an approximation function. This might be useful if the original function is expensive (so you want a cheaper one). You might also want to numerically integrate a function by getting a Chebyshev approximation function that can be analytically integrated. See Judd (1998) for details on how this is done.
* Mars regression spline - Creates a Sum\_Of\_Piecewise\_Functions representing a MARS regression spline. See Friedman (1991) for an explanation of the spline.
* Monotonic MARS spline - A variant of MARS that guarantees monotonicity in each dimension. Uses only forward hinges `max(0, x - t)` (for increasing) or backward hinges `max(0, t - x)` (for decreasing), with non-negative coefficients enforced via NNLS. Each dimension can be independently specified as increasing or decreasing. An optional `min_gradient` parameter adds a guaranteed minimum slope in every dimension, eliminating flat regions to give strict monotonicity.

# Iterative Fitting

The `MultivariateFitter` and `MultivariateAdjustedFitter` structs support iterative fitting of multivariate functions. Each call to `fit!` fits new data and blends with the previous model using an exponentially decaying weight. This is useful for applications where data arrives sequentially (e.g. daily signal-to-return mappings).

Supported methods: `:mars`, `:recursive_partitioning`, `:monotonic_mars`, `:ols`, `:saturated_ols`.

Key features:
* **Blending**: `weight = min(1/(times_through+1), weight_on_new)` controls how much the new fit contributes.
* **Simplification**: Periodic trimming via `trim_mars_spline` on synthetic data prevents unbounded complexity growth. The `simplify_to` parameter (target basis count after trimming) can be larger than `MaxM` (basis count per daily fit), allowing accumulated models to be richer than any single fit.
* **MultivariateAdjustedFitter**: Fits a shared shape function `f(x)` with per-group affine adjustments `y_g ≈ a_g + b_g * f(x)`. Useful when multiple groups share the same signal-to-response shape but at different scales.
