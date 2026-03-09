# --- Multivariate Fitters ---

const DEFAULT_MV_COEFFICIENTS = (0.0, 1.0)

"""
    MultivariateFitter

A mutable struct for iteratively fitting multivariate functions. Each call to `fit!`
fits new data and blends with the previous fit. Periodic simplification via
`trim_mars_spline` (on synthetic data from the blended model) reduces complexity.

### Fields
* `fun` - The current fitted `MultivariateFunction` (or `nothing` before first fit).
* `method` - Fitting method. One of `:mars`, `:recursive_partitioning`, `:monotonic_mars`, `:ols`, `:saturated_ols`.
* `x_variables` - `Set{Symbol}` of predictor dimension names.
* `times_through` - Number of times `fit!` has been called.
* `MaxM` - Number of basis functions for each daily fit.
* `simplify_to` - Target number of basis functions after trimming. Can be larger than `MaxM` to allow accumulated complexity. Set to `0` to disable simplification.
* `simplification_frequency` - Simplify every this many calls to `fit!`. `0` disables.
* `weight_on_new` - Maximum blending weight for new fit in `(0,1]`.
* `rel_tol` - Relative tolerance for MARS/RP optimisation.
* `ols_degree` - Degree for `:saturated_ols` method (ignored otherwise).
* `directions` - `Dict{Symbol,Symbol}` for monotonic MARS directions.
* `min_gradient` - Minimum gradient for monotonic MARS (strict monotonicity).
"""
mutable struct MultivariateFitter
    fun::Union{Nothing, MultivariateFunction}
    method::Symbol
    x_variables::Set{Symbol}
    times_through::Int
    MaxM::Int
    simplify_to::Int
    simplification_frequency::Int
    weight_on_new::Float64
    rel_tol::Float64
    ols_degree::Int
    directions::Dict{Symbol,Symbol}
    min_gradient::Float64
end

"""
    MultivariateFitter(method::Symbol, x_variables::Set{Symbol}; kwargs...)

Create a `MultivariateFitter` initialised with no fitted function.

### Keyword arguments
* `MaxM` - Basis functions per daily fit (default `5`).
* `simplify_to` - Target basis functions after trim (default `0`, disabled).
* `simplification_frequency` - Simplify every N fits (default `0`, disabled).
* `weight_on_new` - Max blending weight (default `1.0`).
* `rel_tol` - Optimisation tolerance (default `1e-2`).
* `ols_degree` - Polynomial degree for `:saturated_ols` (default `2`).
* `directions` - Per-dimension monotonicity directions (default all `:increasing`).
* `min_gradient` - Minimum gradient for monotonic MARS (default `0.0`).
"""
function MultivariateFitter(method::Symbol, x_variables::Set{Symbol};
                             MaxM::Int = 5,
                             simplify_to::Int = 0,
                             simplification_frequency::Int = 0,
                             weight_on_new::Float64 = 1.0,
                             rel_tol::Float64 = 1e-2,
                             ols_degree::Int = 2,
                             directions::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
                             min_gradient::Float64 = 0.0)
    if method ∉ (:mars, :recursive_partitioning, :monotonic_mars, :ols, :saturated_ols)
        error("Unknown method :$method. Use :mars, :recursive_partitioning, :monotonic_mars, :ols, or :saturated_ols.")
    end
    return MultivariateFitter(nothing, method, x_variables, 0, MaxM, simplify_to,
                               simplification_frequency, weight_on_new, rel_tol,
                               ols_degree, directions, min_gradient)
end

function evaluate(fitter::MultivariateFitter, dd::DataFrame)
    if fitter.fun === nothing
        return zeros(size(dd, 1))
    end
    return evaluate(fitter.fun, dd)
end

function evaluate(fitter::MultivariateFitter, coordinates::Dict{Symbol,Float64})
    if fitter.fun === nothing
        return 0.0
    end
    return evaluate(fitter.fun, coordinates)
end

(fitter::MultivariateFitter)(dd::DataFrame) = evaluate(fitter, dd)
(fitter::MultivariateFitter)(coordinates::Dict{Symbol,Float64}) = evaluate(fitter, coordinates)

# Internal: fit a new function to data using the specified method
function _fit_new_function(method::Symbol, dd::DataFrame, y::Symbol, x_variables::Set{Symbol},
                           MaxM::Int, rel_tol::Float64, ols_degree::Int,
                           directions::Dict{Symbol,Symbol}, min_gradient::Float64)
    if method == :mars
        result = create_mars_spline(dd, y, x_variables, MaxM; rel_tol = rel_tol)
        return result.model
    elseif method == :recursive_partitioning
        result = create_recursive_partitioning(dd, y, x_variables, MaxM; rel_tol = rel_tol)
        return result.model
    elseif method == :monotonic_mars
        result = create_monotonic_mars_spline(dd, y, x_variables, MaxM;
                    rel_tol = rel_tol, directions = directions, min_gradient = min_gradient)
        return result.model
    elseif method == :ols
        result = create_saturated_ols_approximation(dd, y, collect(x_variables), 1)
        return result[1]
    elseif method == :saturated_ols
        result = create_saturated_ols_approximation(dd, y, collect(x_variables), ols_degree)
        return result[1]
    else
        error("Unknown method :$method")
    end
end

# Internal: simplify a model by trimming on synthetic data
function _simplify_model(fun::MultivariateFunction, dd::DataFrame, simplify_to::Int,
                         method::Symbol, directions::Dict{Symbol,Symbol}, min_gradient::Float64)
    if simplify_to <= 0
        return fun
    end
    if method ∈ (:ols, :saturated_ols)
        # OLS blending is lossless (polynomial + polynomial = polynomial), no simplification needed
        return fun
    end
    if !(fun isa Sum_Of_Piecewise_Functions)
        return fun
    end

    n_funcs = length(fun.functions_)
    if n_funcs <= simplify_to
        return fun
    end

    # Generate synthetic data from the blended model
    synth_y = :_synth_y_
    dd_synth = copy(dd)
    dd_synth[!, synth_y] = evaluate(fun, dd)

    if method == :monotonic_mars
        # For monotonic MARS: use OLS-based backward deletion to select which basis
        # functions to keep, then re-estimate coefficients with NNLS to preserve monotonicity.
        return _trim_monotonic(fun, dd_synth, synth_y, simplify_to, directions, min_gradient)
    else
        # For mars and recursive_partitioning: use standard trim
        result = trim_mars_spline(dd_synth, synth_y, fun; final_number_of_functions = simplify_to)
        return result.model
    end
end

# Internal: trim a monotonic MARS model preserving monotonicity guarantees
function _trim_monotonic(fun::Sum_Of_Piecewise_Functions, dd::DataFrame, y::Symbol,
                         final_n::Int, directions::Dict{Symbol,Symbol}, min_gradient::Float64)
    array_of_funcs = fun.functions_
    # Also include global_funcs_ piecewise representation if needed
    if length(fun.global_funcs_.functions_) > 0
        # The global funcs include the linear floor terms from min_gradient.
        # We keep those separate and only trim the piecewise basis functions.
    end

    functions_to_delete = length(array_of_funcs) - final_n
    if functions_to_delete <= 0
        return fun
    end

    y_vec = dd[!, y]

    for M in 1:functions_to_delete
        best_lof = Inf
        best_m = 2
        len = length(array_of_funcs)
        for m in 2:len  # never delete the intercept (index 1)
            reduced = array_of_funcs[1:end .!= m]
            X = hcat(evaluate.(reduced, Ref(dd))...)
            coefficients = fit_nnls(X, y_vec)
            new_lof = sum((X * coefficients .- y_vec) .^ 2)
            if new_lof < best_lof
                best_lof = new_lof
                best_m = m
            end
        end
        array_of_funcs = array_of_funcs[1:end .!= best_m]
    end

    # Final NNLS fit with remaining basis functions
    X = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    coefficients = fit_nnls(X, y_vec)
    updated_model = Sum_Of_Piecewise_Functions(array_of_funcs .* coefficients)

    # Re-add global funcs (linear floor terms)
    if length(fun.global_funcs_.functions_) > 0
        updated_model = updated_model + fun.global_funcs_
    end

    return updated_model
end

"""
    fit!(fitter::MultivariateFitter, dd::DataFrame, y::Symbol)

Fit the `MultivariateFitter` to new data. The new fit is blended with the previous
fit using `weight = min(1/(times_through+1), weight_on_new)`. If
`simplification_frequency > 0` and the iteration count is a multiple, the model is
simplified via backward deletion on synthetic data.

### Arguments
* `dd` - DataFrame containing predictor columns and the response.
* `y` - Symbol naming the response column.
"""
function fit!(fitter::MultivariateFitter, dd::DataFrame, y::Symbol)
    newfun = _fit_new_function(fitter.method, dd, y, fitter.x_variables, fitter.MaxM,
                                fitter.rel_tol, fitter.ols_degree, fitter.directions,
                                fitter.min_gradient)

    if fitter.times_through > 0 && fitter.fun !== nothing
        new_weight = min(1.0 / (fitter.times_through + 1), fitter.weight_on_new)
        newfun = (new_weight * newfun) + ((1.0 - new_weight) * fitter.fun)
    end

    if fitter.simplification_frequency > 0 && fitter.times_through > 0 &&
       (fitter.times_through % fitter.simplification_frequency == 0)
        newfun = _simplify_model(newfun, dd, fitter.simplify_to, fitter.method,
                                  fitter.directions, fitter.min_gradient)
    end

    fitter.fun = newfun
    fitter.times_through += 1
end

# ============================================================
# MultivariateAdjustedFitter
# ============================================================

"""
    MultivariateAdjustedFitter

A mutable struct for iteratively fitting a shared multivariate shape function with
per-group affine adjustments. For each group `g`, predictions are `a_g + b_g * f(x)`.

### Fields
* `fun` - The current shared `MultivariateFunction` (or `nothing` before first fit).
* `coefficients` - `Dict` mapping group keys to `(a, b)` tuples.
* `method` - Fitting method (same options as `MultivariateFitter`).
* `x_variables` - `Set{Symbol}` of predictor dimension names.
* `times_through` - Number of times `fit!` has been called.
* `MaxM` - Number of basis functions per daily fit.
* `simplify_to` - Target basis functions after trimming (`0` = disabled).
* `simplification_frequency` - Simplify every N fits (`0` = disabled).
* `weight_on_new` - Maximum blending weight for new fit.
* `rel_tol` - Optimisation tolerance.
* `ols_degree` - Degree for `:saturated_ols`.
* `directions` - Per-dimension monotonicity directions.
* `min_gradient` - Minimum gradient for monotonic MARS.
* `adjust_for_groups` - If `true`, undo group coefficients before fitting the shape.
* `fit_intercept` - If `true`, estimate both `a` and `b`; if `false`, force `a=0`.
* `coefficient_bounds` - `((a_min, a_max), (b_min, b_max))` clamping bounds.
"""
mutable struct MultivariateAdjustedFitter
    fun::Union{Nothing, MultivariateFunction}
    coefficients::Dict
    method::Symbol
    x_variables::Set{Symbol}
    times_through::Int
    MaxM::Int
    simplify_to::Int
    simplification_frequency::Int
    weight_on_new::Float64
    rel_tol::Float64
    ols_degree::Int
    directions::Dict{Symbol,Symbol}
    min_gradient::Float64
    adjust_for_groups::Bool
    fit_intercept::Bool
    coefficient_bounds::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
end

"""
    MultivariateAdjustedFitter(method::Symbol, x_variables::Set{Symbol}; kwargs...)

Create a `MultivariateAdjustedFitter` initialised with no fitted function.

### Keyword arguments
Same as `MultivariateFitter`, plus:
* `coefficients` - Initial group coefficients `Dict` (default empty).
* `adjust_for_groups` - Undo group coefficients before fitting (default `true`).
* `fit_intercept` - Estimate group intercept (default `true`).
* `coefficient_bounds` - `((a_min, a_max), (b_min, b_max))` (default `((-1.0, 1.0), (0.1, 2.5))`).
"""
function MultivariateAdjustedFitter(method::Symbol, x_variables::Set{Symbol};
                                     MaxM::Int = 5,
                                     simplify_to::Int = 0,
                                     simplification_frequency::Int = 0,
                                     weight_on_new::Float64 = 1.0,
                                     rel_tol::Float64 = 1e-2,
                                     ols_degree::Int = 2,
                                     directions::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
                                     min_gradient::Float64 = 0.0,
                                     coefficients::Dict = Dict(),
                                     adjust_for_groups::Bool = true,
                                     fit_intercept::Bool = true,
                                     coefficient_bounds::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}} = ((-1.0, 1.0), (0.1, 2.5)))
    if method ∉ (:mars, :recursive_partitioning, :monotonic_mars, :ols, :saturated_ols)
        error("Unknown method :$method. Use :mars, :recursive_partitioning, :monotonic_mars, :ols, or :saturated_ols.")
    end
    return MultivariateAdjustedFitter(nothing, coefficients, method, x_variables, 0, MaxM,
                                       simplify_to, simplification_frequency, weight_on_new,
                                       rel_tol, ols_degree, directions, min_gradient,
                                       adjust_for_groups, fit_intercept, coefficient_bounds)
end

function evaluate(fitter::MultivariateAdjustedFitter, dd::DataFrame, group)
    coeffs = group in keys(fitter.coefficients) ? fitter.coefficients[group] : DEFAULT_MV_COEFFICIENTS
    if fitter.fun === nothing
        return fill(coeffs[1], size(dd, 1))
    end
    return coeffs[1] .+ coeffs[2] .* evaluate(fitter.fun, dd)
end

function evaluate(fitter::MultivariateAdjustedFitter, coordinates::Dict{Symbol,Float64}, group)
    coeffs = group in keys(fitter.coefficients) ? fitter.coefficients[group] : DEFAULT_MV_COEFFICIENTS
    if fitter.fun === nothing
        return coeffs[1]
    end
    return coeffs[1] + coeffs[2] * evaluate(fitter.fun, coordinates)
end

(fitter::MultivariateAdjustedFitter)(dd::DataFrame, group) = evaluate(fitter, dd, group)
(fitter::MultivariateAdjustedFitter)(coordinates::Dict{Symbol,Float64}, group) = evaluate(fitter, coordinates, group)

"""
    fit!(fitter::MultivariateAdjustedFitter, dd::DataFrame, y::Symbol, groups::Vector)

Fit the `MultivariateAdjustedFitter` to new data with group labels.

1. Onboard new groups with default coefficients `(0.0, 1.0)`.
2. If `adjust_for_groups`, undo group coefficients: `y_adj = (y - a_g) / b_g`.
3. Fit the shared shape function to adjusted data.
4. Blend with previous fit.
5. Re-estimate per-group `(a_g, b_g)` via weighted OLS, clamped to bounds.
6. Periodically simplify.
"""
function fit!(fitter::MultivariateAdjustedFitter, dd::DataFrame, y::Symbol, groups::Vector)
    # Onboard new groups
    new_groups = setdiff(unique(groups), keys(fitter.coefficients))
    for g in new_groups
        fitter.coefficients[g] = DEFAULT_MV_COEFFICIENTS
    end

    # Undo group coefficients to map y into shared function space
    y_vals = dd[!, y]
    if fitter.adjust_for_groups && fitter.fun !== nothing
        y_adjusted = similar(y_vals, Float64)
        for i in eachindex(y_vals)
            a, b = fitter.coefficients[groups[i]]
            y_adjusted[i] = (y_vals[i] - a) / b
        end
    else
        y_adjusted = Float64.(y_vals)
    end

    # Create temporary dataframe with adjusted y
    dd_fit = copy(dd)
    adj_y = :_adj_y_
    dd_fit[!, adj_y] = y_adjusted

    # Fit the shared shape
    newfun = _fit_new_function(fitter.method, dd_fit, adj_y, fitter.x_variables, fitter.MaxM,
                                fitter.rel_tol, fitter.ols_degree, fitter.directions,
                                fitter.min_gradient)

    # Blend with previous fit
    if fitter.times_through > 0 && fitter.fun !== nothing
        new_weight = min(1.0 / (fitter.times_through + 1), fitter.weight_on_new)
        newfun = (new_weight * newfun) + ((1.0 - new_weight) * fitter.fun)
    end

    # Simplify periodically
    if fitter.simplification_frequency > 0 && fitter.times_through > 0 &&
       (fitter.times_through % fitter.simplification_frequency == 0)
        newfun = _simplify_model(newfun, dd, fitter.simplify_to, fitter.method,
                                  fitter.directions, fitter.min_gradient)
    end

    fitter.fun = newfun

    # Re-estimate per-group coefficients
    (a_min, a_max) = fitter.coefficient_bounds[1]
    (b_min, b_max) = fitter.coefficient_bounds[2]
    blend_weight = fitter.times_through > 0 ? min(1.0 / (fitter.times_through + 1), fitter.weight_on_new) : 1.0

    for g in unique(groups)
        mask = groups .== g
        f_vals = evaluate(newfun, dd[mask, :])
        y_g = Float64.(y_vals[mask])
        n_g = length(y_g)
        if n_g < 2
            continue
        end

        if fitter.fit_intercept
            # OLS with intercept: y = a + b*f
            mean_f = sum(f_vals) / n_g
            mean_y = sum(y_g) / n_g
            var_f = sum((f_vals .- mean_f) .^ 2) / n_g
            if var_f < 1e-12
                new_a = mean_y
                new_b = 1.0
            else
                cov_fy = sum((f_vals .- mean_f) .* (y_g .- mean_y)) / n_g
                new_b = cov_fy / var_f
                new_a = mean_y - new_b * mean_f
            end
        else
            # No-intercept OLS: y = b*f
            sum_f2 = sum(f_vals .^ 2)
            if sum_f2 < 1e-12
                new_b = 1.0
            else
                new_b = sum(f_vals .* y_g) / sum_f2
            end
            new_a = 0.0
        end

        # Clamp to bounds
        new_a = clamp(new_a, a_min, a_max)
        new_b = clamp(new_b, b_min, b_max)

        # Blend with previous coefficients
        old_a, old_b = fitter.coefficients[g]
        fitter.coefficients[g] = (blend_weight * new_a + (1 - blend_weight) * old_a,
                                  blend_weight * new_b + (1 - blend_weight) * old_b)
    end

    fitter.times_through += 1
end
