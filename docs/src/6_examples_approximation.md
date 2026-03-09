
# Examples - Approximation

## OLS approximation

If we have lots of data that we want to summarise with OLS
```
# Generating example data
using Random
using Distributions
using DataStructures
Random.seed!(1)
obs = 1000
X = rand(obs)
y = X .+ rand(Normal(),obs) .+ 7
# And now making an approximation function
approxFunction, reg = create_ols_approximation(y, X, 2)
```

## Numerical Integration with Chebyshev polynomials

And if we want to approximate the sin function in the [2.3, 5.6] bound with 7 polynomial terms and 20 approximation nodes:
```
chebyshevs = create_chebyshev_approximation(sin, 20, 7, OrderedDict{Symbol,Tuple{Float64,Float64}}(:default => (2.3, 5.6)))
```
We can integrate the above term in the normal way to achieve Gauss-Chebyshev quadrature:
```
integral(chebyshevs, 2.3, 5.6)
```

## Multivariate: MARS Spline for approximation

First we will generate some example data.
```
using MultivariateFunctions
using Random
using DataFrames
using Distributions
using DataStructures

Random.seed!(1992)
nObs = 1000
dd = DataFrame()
dd[!, :x] = rand( Normal(),nObs) + 0.1 .* rand( Normal(),nObs)
dd[!, :z] = rand( Normal(),nObs) + 0.1 .* rand( Normal(),nObs)
dd[!, :w] = (0.5 .* rand( Normal(),nObs)) .+ 0.7.*(dd[!, :z] .- dd[!, :x]) + 0.1 .* rand( Normal(),nObs)
dd[!, :y] = (dd[!, :x] .*dd[!, :w] ) .* (dd[!, :z] .- dd[!, :w]) .+ dd[!, :x] + rand( Normal(),nObs)
dd[7,:y] = 1.0
y = :y
x_variables = Set{Symbol}([:w, :x, :z])
```
It is important to note here that we have a set of symbols for x\_variables. This is the set of columns in the
dataframe that we will use to predict y - the dependent variable.

We can then create an approximation with recursive partitioning:
```
number_of_divisions = 7
rp_4, rp_reg_4 = create_recursive_partitioning(dd, y, x_variables, number_of_divisions; rel_tol = 1e-3)
```
We can also create a MARS approximation spline:
```
rp_1, rp_reg_1 = create_mars_spline(dd, y, x_variables, number_of_divisions; rel_tol = 1e-3)
```
Note that the rel\_tol here is the tolerance in the optimisation step for hinges (or divisions in the recursive partitioning case). In most applied cases it generally doesn't matter much if there is a hinge at 1.0006 or at 1.0007 so in most settings this can be set higher than you would generally set the tolerance for a numerical optimiser. For this reason the default value is 1e-02.

## Monotonic MARS Spline

If you need a MARS approximation that is guaranteed to be monotonic in each dimension, use `create_monotonic_mars_spline`. First generate some example data:
```
using MultivariateFunctions
using Random
using DataFrames
using Distributions
using DataStructures

Random.seed!(1)
nObs = 500
dd = DataFrame()
dd[!, :x] = rand(Normal(), nObs)
dd[!, :z] = rand(Normal(), nObs)
dd[!, :y] = 3.0 .* max.(0.0, dd[!, :x] .- 0.5) .+ 2.0 .* max.(0.0, dd[!, :z] .+ 1.0) .+ 1.0
```
By default all dimensions are monotone increasing:
```
result = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 5; rel_tol = 1e-3)
monotone_model = result.model
```
You can specify directions per dimension. For example, increasing in `:x` and decreasing in `:z`:
```
result = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 5;
    directions = Dict{Symbol,Symbol}(:x => :increasing, :z => :decreasing))
```
The returned named tuple contains the `model` (a `Sum_Of_Piecewise_Functions`), the `coefficients` vector, and the `rss` (residual sum of squares).

Monotonicity is guaranteed by construction: each basis function is a product of non-negative, monotone hinge functions, and all non-intercept coefficients are constrained to be non-negative via NNLS fitting.

By default the result is monotone (non-decreasing) which allows flat regions where the hinge functions are zero. To ensure strict monotonicity (no flat regions), set `min_gradient` to a positive value. This adds a linear term with that slope in every dimension before fitting, so the partial derivative in each dimension is always at least `min_gradient`:
```
result = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 5;
    min_gradient = 0.01)
```

## Weighted Approximation

All approximation methods accept an optional `weights` keyword. Weights must be a `Vector{Float64}` with one entry per observation:

```
using MultivariateFunctions
using DataFrames, Random, Distributions

Random.seed!(1)
nObs = 200
dd = DataFrame()
dd[!, :x] = rand(Normal(), nObs)
dd[!, :z] = rand(Normal(), nObs)
dd[!, :y] = 2.0 .* max.(0.0, dd[!, :x] .- 0.3) .+ 1.5 .* max.(0.0, dd[!, :z] .+ 0.5) .+ 1.0

# Random positive weights
w = rand(nObs) .+ 0.1

# Weighted OLS
model_w, reg_w = create_saturated_ols_approximation(dd, :y, [:x, :z], 2; weights = w)

# Weighted MARS
result_w = create_mars_spline(dd, :y, Set([:x, :z]), 5; weights = w)

# Weighted monotonic MARS
result_mw = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 5; weights = w)
```

Near-zero weights effectively exclude observations, which is useful for downweighting outliers:
```
# Add outliers and downweight them
dd_out = vcat(dd, DataFrame(x = [0.0], z = [0.0], y = [1000.0]))
w_out = vcat(ones(nObs), [1e-10])
model_clean, _ = create_saturated_ols_approximation(dd_out, :y, [:x, :z], 2; weights = w_out)
```

## Iterative Fitting with MultivariateFitter

The `MultivariateFitter` allows iterative fitting where each call to `fit!` blends new data with the accumulated model. This is useful for daily signal-to-return mappings.

```
using MultivariateFunctions
using DataFrames
using Random
using Distributions

Random.seed!(1)

# Create a fitter: 4 basis functions per fit, trim to 8 every 5 fits
fitter = MultivariateFitter(:mars, Set([:x, :z]);
    MaxM = 4, simplify_to = 8, simplification_frequency = 5,
    weight_on_new = 0.5)

# Simulate daily fitting
for day in 1:10
    dd = DataFrame()
    dd[!, :x] = rand(Normal(), 200)
    dd[!, :z] = rand(Normal(), 200)
    dd[!, :y] = 2.0 .* max.(0.0, dd[!, :x] .- 0.3) .+ dd[!, :z] .+ rand(Normal(), 200)
    fit!(fitter, dd, :y)
end

# Evaluate the accumulated model
dd_test = DataFrame(x = [1.0, 2.0], z = [0.5, -0.5])
predictions = evaluate(fitter, dd_test)
```

Both fitter types support callable syntax as a shorthand for `evaluate`:
```
# These are equivalent:
predictions = evaluate(fitter, dd_test)
predictions = fitter(dd_test)

# Single-point evaluation:
val = fitter(Dict(:x => 1.0, :z => 0.5))
```

Weights can be passed to `fit!` for weighted fitting:
```
w = rand(200) .+ 0.1
fit!(fitter, dd, :y; weights = w)
```

## Monotonic MARS with Iterative Fitting

The `MultivariateFitter` supports `:monotonic_mars` for iterative fitting with monotonicity guarantees:
```
fitter = MultivariateFitter(:monotonic_mars, Set([:x, :z]);
    MaxM = 4, weight_on_new = 0.5,
    directions = Dict(:x => :increasing, :z => :increasing),
    min_gradient = 0.01)

for day in 1:10
    dd = DataFrame()
    dd[!, :x] = rand(Normal(), 200)
    dd[!, :z] = rand(Normal(), 200)
    dd[!, :y] = 3.0 .* max.(0.0, dd[!, :x]) .+ 2.0 .* dd[!, :z] .+ 1.0 .+ rand(Normal(), 200)
    fit!(fitter, dd, :y)
end
```

## Iterative Fitting with Group Adjustments

The `MultivariateAdjustedFitter` fits a shared shape `f(x)` with per-group affine coefficients `y_g ≈ a_g + b_g * f(x)`:

```
fitter = MultivariateAdjustedFitter(:mars, Set([:x, :z]);
    MaxM = 4, weight_on_new = 0.5,
    coefficient_bounds = ((-2.0, 2.0), (0.1, 3.0)))

dd = DataFrame()
dd[!, :x] = rand(Normal(), 300)
dd[!, :z] = rand(Normal(), 300)
dd[!, :y] = rand(Normal(), 300)
groups = vcat(fill(:A, 150), fill(:B, 150))

fit!(fitter, dd, :y, groups)

# Group-specific predictions (these are equivalent)
predictions_A = evaluate(fitter, dd[1:10, :], :A)
predictions_A = fitter(dd[1:10, :], :A)

# Single-point evaluation
val = fitter(Dict(:x => 1.0, :z => 0.5), :B)
```

The `coefficient_bounds` parameter clamps per-group `(a, b)` to specified ranges. Setting `fit_intercept = false` forces `a = 0`:
```
fitter = MultivariateAdjustedFitter(:mars, Set([:x, :z]);
    MaxM = 4, weight_on_new = 0.5,
    fit_intercept = false,
    coefficient_bounds = ((-5.0, 5.0), (0.5, 2.0)))
```

For unknown groups (not seen during fitting), predictions use default coefficients `(a=0.0, b=1.0)`:
```
# Group :C was never in the training data
predictions_C = fitter(dd[1:5, :], :C)  # uses a=0, b=1
```
