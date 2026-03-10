# Remove element at index without allocating a boolean mask
_remove_at(arr, i) = vcat(arr[1:i-1], arr[i+1:end])

# Helper for weighted RSS computation (allocation-free)
function _weighted_rss(residuals::AbstractVector{Float64}, weights::Union{Nothing, Vector{Float64}})
    if weights === nothing
        return sum(r -> r^2, residuals)
    else
        return sum(i -> weights[i] * residuals[i]^2, eachindex(residuals))
    end
end

function add_split_with_step_function(array_of_funcs::Array, ind::Int, split_variable::Symbol, split_point::Float64, removeSplitFunction::Bool)
    basis_function1 = Piecewise_Function( vcat(Sum_Of_Functions([PE_Function(0.0)]), Sum_Of_Functions([PE_Function(1.0)])) , OrderedDict{Symbol,Array{Float64,1}}(split_variable .=> [[-Inf, split_point]]))
    basis_function2 = Piecewise_Function( vcat(Sum_Of_Functions([PE_Function(1.0)]), Sum_Of_Functions([PE_Function(0.0)])) , OrderedDict{Symbol,Array{Float64,1}}(split_variable .=> [[-Inf, split_point]]))
    other_functions = _remove_at(array_of_funcs, ind) # ind is the index of the function to split.
    split_function = array_of_funcs[ind]
    if removeSplitFunction
        return vcat(other_functions, basis_function1 * split_function, basis_function2 * split_function)
    else
        return vcat(array_of_funcs, basis_function1 * split_function, basis_function2 * split_function)
    end
end

function add_split_with_max_function(array_of_funcs::Array, ind::Int, split_variable::Symbol, split_point::Float64, removeSplitFunction::Bool)
    max_func = Sum_Of_Functions([PE_Function(1.0, UnitMap([split_variable => PE_Unit(0.0,split_point,1)]))])
    basis_function1 = Piecewise_Function( vcat(Sum_Of_Functions([PE_Function(0.0)]), max_func) , OrderedDict{Symbol,Array{Float64,1}}(split_variable .=> [[-Inf, split_point]]))
    basis_function2 = Piecewise_Function( vcat(-1 * max_func, Sum_Of_Functions([PE_Function(0.0)])) , OrderedDict{Symbol,Array{Float64,1}}(split_variable .=> [[-Inf, split_point]]))
    other_functions = _remove_at(array_of_funcs, ind) # ind is the index of the function to split.
    split_function = array_of_funcs[ind]
    if removeSplitFunction
        return vcat(other_functions, basis_function1 * split_function, basis_function2 * split_function)
    else
        return vcat(array_of_funcs, basis_function1 * split_function, basis_function2 * split_function)
    end
end

function optimise_split(dd::DataFrame, y::Symbol, array_of_funcs::Array, ind::Int, split_variable::Symbol, rel_tol::Float64, SplitFunction::Function, removeSplitFunction::Bool; weights::Union{Nothing, Vector{Float64}} = nothing)
    lower_limit = minimum(dd[!, split_variable]) + eps()
    upper_limit = maximum(dd[!, split_variable]) - eps()

    # Pre-compute design matrix columns for basis functions that don't change
    y_vec = dd[!, y]
    if removeSplitFunction
        unchanged_funcs = _remove_at(array_of_funcs, ind)
    else
        unchanged_funcs = array_of_funcs
    end
    if length(unchanged_funcs) > 0
        X_cached = hcat(evaluate.(unchanged_funcs, Ref(dd))...)
    else
        X_cached = Matrix{Float64}(undef, size(dd, 1), 0)
    end

    opt = optimize(lower_limit, upper_limit; rel_tol = rel_tol) do split_point
        # Only construct and evaluate the 2 new basis functions
        model = SplitFunction(array_of_funcs, ind, split_variable, split_point, removeSplitFunction)
        new_funcs = model[end-1:end]
        X_new = hcat(evaluate.(new_funcs, Ref(dd))...)
        X = hcat(X_cached, X_new)
        if weights !== nothing
            # Near-zero weights can make the weighted design matrix rank-deficient,
            # causing GLM's Cholesky factorization to fail. Check before calling GLM.
            sqrtW = sqrt.(weights)
            Xw = sqrtW .* X
            if !isposdef(Xw' * Xw)
                return Inf
            end
        end
        if weights === nothing
            reg = fit(LinearModel, X, y_vec; dropcollinear=true)
        else
            reg = fit(LinearModel, X, y_vec; dropcollinear=true, weights=FrequencyWeights(weights))
        end
        return _weighted_rss(reg.rr.mu .- reg.rr.y, weights)
    end
    return (opt.minimum, opt.minimizer)
end

"""
    create_recursive_partitioning(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int; rel_tol::Float64 = 1e-2, weights = nothing)

This creates a recusive partitioning approximation. This seperates the space in to a series of hypercubes each of which has a constant
value within the hypercube. Each step of the algorithm divides a hypercube along some dimension so that the different parts of the hypercube
can recieve a different value.
The relative tolerance is used in a one-dimensional optimisation step to determine what points at which split values to place
a hypercube in a particular dimension. The default is intentionally set high because it generally doesnt matter
that much. For small scale data however you might want to decrease it and increase it for large scale data. You might also want to
decrease it if spline creation time doesnt matter much. Note that a small rel_tol only affects creation time for the spline and
not the evaluation time.

Returns a named tuple `(model, regression)` where `model` is a `Sum_Of_Piecewise_Functions` and `regression` is the GLM `LinearModel` object.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted least squares fit is used at each step.
"""
function create_recursive_partitioning(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int; rel_tol::Float64 = 1e-2, weights::Union{Nothing, Vector{Float64}} = nothing)
    Arr = Array{Sum_Of_Functions,length(x_variables)}(undef, repeat([1], length(x_variables))...)
    Arr[repeat([1], length(x_variables))...] = Sum_Of_Functions([PE_Function(1.0)])
    pw_func = Piecewise_Function(Arr, OrderedDict{Symbol,Array{Float64,1}}(x_variables .=> repeat([[-Inf]],length(x_variables))) )
    array_of_funcs = Array{Piecewise_Function,1}([pw_func])
    for M in 2:MaxM
        best_lof    = Inf
        best_m      = 1
        best_dimen  = collect(x_variables)[1]
        best_split  = 0.0
        for m in 1:length(array_of_funcs)
            for dimen in x_variables
                lof, spt = optimise_split(dd, y, array_of_funcs, m, dimen, rel_tol, add_split_with_step_function, true; weights=weights)
                if lof < best_lof
                    best_lof = lof
                    best_m = m
                    best_dimen = dimen
                    best_split = spt
                end
            end
        end
        array_of_funcs = add_split_with_step_function(array_of_funcs, best_m, best_dimen, best_split, true)
    end
    updated_model, reg = create_ols_approximation(dd, y, array_of_funcs; weights=weights)
    return (model = updated_model, regression = reg)
end

"""
    create_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int; rel_tol::Float64 = 1e-2, weights = nothing)

This creates a mars spline given a dataframe, response variable and a set of x_variables from the dataframe.
The relative tolerance is used in a one-dimensional optimisation step to determine what points at which split values to place
a max(0,x-split) function in a particular dimension. The default is intentionally set high because precision is generally not the
not that important. For small scale data however you might want to decrease it and increase it for large scale data. You might also want to
decrease it if spline creation time doesnt matter much. Note that a small rel_tol only affects creation time for the spline and
not the evaluation time.

Returns a named tuple `(model, regression)` where `model` is a `Sum_Of_Piecewise_Functions` and `regression` is the GLM `LinearModel` object.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted least squares fit is used at each step.
"""
function create_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int; rel_tol::Float64 = 1e-2, weights::Union{Nothing, Vector{Float64}} = nothing)
    # This should be made more efficient using FAST MARS. https://statistics.stanford.edu/sites/default/files/LCS%20110.pdf
    Arr = Array{Sum_Of_Functions,length(x_variables)}(undef, repeat([1], length(x_variables))...)
    Arr[repeat([1], length(x_variables))...] = Sum_Of_Functions([PE_Function(1.0)])
    pw_func = Piecewise_Function(Arr, OrderedDict{Symbol,Array{Float64,1}}(x_variables .=> repeat([[-Inf]],length(x_variables))) )
    array_of_funcs = Vector{Piecewise_Function}([pw_func])
    for M in 2:MaxM
        best_lof    = Inf
        best_m      = 1
        best_dimen  = collect(x_variables)[1]
        best_split  = 0.0
        for m in 1:length(array_of_funcs)
            underlying = underlying_dimensions(array_of_funcs[m])
            for dimen in setdiff(x_variables, underlying)
                lof, spt = optimise_split(dd, y, array_of_funcs, m, dimen, rel_tol, add_split_with_max_function, false; weights=weights)
                if lof < best_lof
                    best_lof = lof
                    best_m = m
                    best_dimen = dimen
                    best_split = spt
                end
            end
        end
        array_of_funcs = add_split_with_max_function(array_of_funcs, best_m, best_dimen, best_split, false)
    end
    updated_model, reg = create_ols_approximation(dd, y, array_of_funcs; weights=weights)
    return (model = updated_model, regression = reg)
end

# Solve OLS on a column subset of a precomputed design matrix.
# Returns RSS only (no model construction) for fast candidate evaluation.
function _ols_rss_for_columns(X_full::Matrix{Float64}, y_vec::Vector{Float64},
                               cols::Vector{Int}, weights::Union{Nothing, Vector{Float64}})
    X = @view X_full[:, cols]
    if weights === nothing
        coefs = X \ y_vec
    else
        sqrtW = sqrt.(weights)
        Xw = sqrtW .* X
        yw = sqrtW .* y_vec
        coefs = Xw \ yw
    end
    return _weighted_rss(X * coefs .- y_vec, weights)
end

# Backward deletion core: precomputes the design matrix once, then repeatedly
# finds and removes the column (excluding column 1 / the intercept) whose
# deletion increases RSS the least. Returns the surviving column indices.
function _backward_delete(X_full::Matrix{Float64}, y_vec::Vector{Float64},
                           weights::Union{Nothing, Vector{Float64}},
                           keep_cols::Vector{Int},
                           should_stop::Function)
    while length(keep_cols) > 1
        best_rss = Inf
        best_idx = 2
        for m in 2:length(keep_cols)
            candidate = vcat(keep_cols[1:m-1], keep_cols[m+1:end])
            rss = _ols_rss_for_columns(X_full, y_vec, candidate, weights)
            if rss < best_rss
                best_rss = rss
                best_idx = m
            end
        end
        if should_stop(best_rss)
            break
        end
        keep_cols = vcat(keep_cols[1:best_idx-1], keep_cols[best_idx+1:end])
    end
    return keep_cols
end

function trim_mars_spline_final_number_of_functions(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions, final_number_of_functions::Int; weights::Union{Nothing, Vector{Float64}} = nothing)
    if final_number_of_functions < 2
        error("Cannot trim the number of functions to less than 2")
    end
    array_of_funcs = model.functions_
    n_funcs = length(array_of_funcs)
    functions_to_delete = n_funcs - final_number_of_functions
    if functions_to_delete <= 0
        updated_model, reg = create_ols_approximation(dd, y, array_of_funcs; weights=weights)
        return (model = updated_model, regression = reg)
    end
    X_full = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    y_vec = Vector{Float64}(dd[!, y])
    keep_cols = collect(1:n_funcs)
    deleted = Ref(0)
    keep_cols = _backward_delete(X_full, y_vec, weights, keep_cols,
                                  _ -> (deleted[] += 1; deleted[] > functions_to_delete))
    updated_model, reg = create_ols_approximation(dd, y, array_of_funcs[keep_cols]; weights=weights)
    return (model = updated_model, regression = reg)
end
function trim_mars_spline_maximum_increase_in_RSS(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions, maximum_increase_in_RSS::Float64; weights::Union{Nothing, Vector{Float64}} = nothing)
    array_of_funcs = model.functions_
    n_funcs = length(array_of_funcs)
    X_full = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    y_vec = Vector{Float64}(dd[!, y])
    previous_rss = Ref(_ols_rss_for_columns(X_full, y_vec, collect(1:n_funcs), weights))
    keep_cols = collect(1:n_funcs)
    keep_cols = _backward_delete(X_full, y_vec, weights, keep_cols,
                                  best_rss -> begin
                                      if best_rss - previous_rss[] >= maximum_increase_in_RSS
                                          return true
                                      end
                                      previous_rss[] = best_rss
                                      return false
                                  end)
    updated_model, reg = create_ols_approximation(dd, y, array_of_funcs[keep_cols]; weights=weights)
    return (model = updated_model, regression = reg)
end
function trim_mars_spline_maximum_RSS(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions, maximum_RSS::Float64; weights::Union{Nothing, Vector{Float64}} = nothing)
    array_of_funcs = model.functions_
    n_funcs = length(array_of_funcs)
    X_full = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    y_vec = Vector{Float64}(dd[!, y])
    keep_cols = collect(1:n_funcs)
    keep_cols = _backward_delete(X_full, y_vec, weights, keep_cols,
                                  best_rss -> best_rss >= maximum_RSS)
    updated_model, reg = create_ols_approximation(dd, y, array_of_funcs[keep_cols]; weights=weights)
    return (model = updated_model, regression = reg)
end
"""
    trim_mars_spline(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions;
                       maximum_RSS::Float64 = -1.0, maximum_increase_in_RSS::Float64 = -1.0,
                       final_number_of_functions::Int = -1, weights = nothing)

This trims a mars spline created in the create_mars_spline function. This algorithm goes through
each piecewise function in the mars spline and deletes the one that contributes least to the fit.
A termination criterion must be set. There are three possible termination criterions. The first is
the maximum_RSS that can be tolerated. If this is set then functions will be deleted until the deletion of an
additional function would push RSS over this amount. The second is maximum_increase_in_RSS which will delete
functions until a deletion increases RSS by more than this amount. The final is final_number_of_functions
which reduces the number of fucntions to this number.

Returns a named tuple `(model, regression)` where `model` is the trimmed `Sum_Of_Piecewise_Functions` and `regression` is the final GLM `LinearModel` object.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), weighted RSS is used for all comparisons.
"""
function trim_mars_spline(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions;
                   maximum_RSS::Float64 = -1.0, maximum_increase_in_RSS::Float64 = -1.0, final_number_of_functions::Int = -1,
                   weights::Union{Nothing, Vector{Float64}} = nothing)
    if ((maximum_RSS > 0.0) && (maximum_increase_in_RSS > 0.0)) ||
       ((maximum_RSS > 0.0) && (final_number_of_functions > 0)) ||
       ((maximum_increase_in_RSS > 0.0) && (final_number_of_functions > 0))
        error("You cannot specify more than one condition for trimming the mars spline.")
    elseif (maximum_RSS < 0.0) && (maximum_increase_in_RSS < 0.0) && (final_number_of_functions < 0)
        error("You must specify at least one condition for trimming. The final number of functions to trim to, the maximum increase in RSS or the maximum RSS.")
    elseif (maximum_RSS > 0.0)
        return trim_mars_spline_maximum_RSS(dd, y, model, maximum_RSS; weights=weights)
    elseif (maximum_increase_in_RSS > 0.0)
        return trim_mars_spline_maximum_increase_in_RSS(dd, y, model, maximum_increase_in_RSS; weights=weights)
    elseif (final_number_of_functions > 0)
        return trim_mars_spline_final_number_of_functions(dd, y, model, final_number_of_functions; weights=weights)
    else
        error("This should be unreachable code. Please let the developer know if you get this.")
    end
end

# --- Monotonic MARS ---

function fit_nnls(X::Matrix{Float64}, y_vec::Vector{Float64}; weights::Union{Nothing, Vector{Float64}} = nothing)
    n, p = size(X)
    if p == 0
        return Float64[]
    end
    if p == 1
        if weights === nothing
            return X \ y_vec
        else
            sqrtW = sqrt.(weights)
            return (sqrtW .* X) \ (sqrtW .* y_vec)
        end
    end
    # Separate intercept (column 1, unconstrained) from remaining columns (>= 0).
    # Center data to analytically eliminate the intercept, then solve NNLS via
    # coordinate descent on the centered problem.
    X2 = X[:, 2:end]
    if weights === nothing
        y_mean = sum(y_vec) / n
        X2_means = vec(sum(X2, dims=1) ./ n)
    else
        W = sum(weights)
        y_mean = sum(weights .* y_vec) / W
        X2_means = vec(sum(weights .* X2, dims=1) ./ W)
    end
    y_c = y_vec .- y_mean
    X2_c = X2 .- X2_means'
    # Coordinate descent NNLS: min ||X2_c * β2 - y_c||² s.t. β2 >= 0
    # For weighted case, use sqrt(weights) scaling so normal equations become X'WX and X'Wy
    p2 = p - 1
    if weights === nothing
        AtA = X2_c' * X2_c
        Atb = X2_c' * y_c
    else
        sqrtW = sqrt.(weights)
        X2_cw = sqrtW .* X2_c
        y_cw = sqrtW .* y_c
        AtA = X2_cw' * X2_cw
        Atb = X2_cw' * y_cw
    end
    β2 = zeros(p2)
    for iter in 1:5000
        converged = true
        for j in 1:p2
            r_j = Atb[j]
            for k in 1:p2
                if k != j
                    r_j -= AtA[j, k] * β2[k]
                end
            end
            new_val = AtA[j, j] > 1e-15 ? max(r_j / AtA[j, j], 0.0) : 0.0
            if abs(new_val - β2[j]) > 1e-14 * (1 + abs(β2[j]))
                converged = false
            end
            β2[j] = new_val
        end
        if converged
            break
        end
    end
    # Recover intercept
    β1 = y_mean - sum(X2_means .* β2)
    return vcat([β1], β2)
end

function add_split_monotone(array_of_funcs::Array, ind::Int, split_variable::Symbol,
                            split_point::Float64, direction::Symbol)
    max_func = Sum_Of_Functions([PE_Function(1.0, UnitMap([split_variable => PE_Unit(0.0, split_point, 1)]))])
    if direction == :increasing
        # max(0, x - split_point): zero below split, linear above
        basis_function = Piecewise_Function(
            vcat(Sum_Of_Functions([PE_Function(0.0)]), max_func),
            OrderedDict{Symbol,Array{Float64,1}}(split_variable .=> [[-Inf, split_point]])
        )
    else
        # max(0, split_point - x): linear below split, zero above
        basis_function = Piecewise_Function(
            vcat(-1 * max_func, Sum_Of_Functions([PE_Function(0.0)])),
            OrderedDict{Symbol,Array{Float64,1}}(split_variable .=> [[-Inf, split_point]])
        )
    end
    split_function = array_of_funcs[ind]
    return vcat(array_of_funcs, basis_function * split_function)
end

function optimise_monotone_split(dd::DataFrame, y::Symbol, array_of_funcs::Array, ind::Int,
                                 split_variable::Symbol, direction::Symbol, rel_tol::Float64;
                                 weights::Union{Nothing, Vector{Float64}} = nothing)
    lower_limit = minimum(dd[!, split_variable]) + eps()
    upper_limit = maximum(dd[!, split_variable]) - eps()
    y_vec = dd[!, y]
    X_cached = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    opt = optimize(lower_limit, upper_limit; rel_tol = rel_tol) do split_point
        model = add_split_monotone(array_of_funcs, ind, split_variable, split_point, direction)
        new_func = model[end:end]
        X_new = hcat(evaluate.(new_func, Ref(dd))...)
        X = hcat(X_cached, X_new)
        coefficients = fit_nnls(X, y_vec; weights=weights)
        return _weighted_rss(X * coefficients .- y_vec, weights)
    end
    return (opt.minimum, opt.minimizer)
end

"""
    create_monotonic_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int;
                                 rel_tol::Float64 = 1e-2,
                                 directions::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
                                 min_gradient::Float64 = 0.0,
                                 weights::Union{Nothing, Vector{Float64}} = nothing)

Creates a MARS spline that is guaranteed to be monotonic in each specified dimension.
Each basis function uses only forward hinges `max(0, x - t)` (for increasing dimensions) or
backward hinges `max(0, t - x)` (for decreasing dimensions), and non-intercept coefficients are
constrained to be non-negative via NNLS fitting. This guarantees the resulting function is monotonic
in each dimension in the specified direction.

The `directions` argument maps each dimension symbol to either `:increasing` or `:decreasing`.
If not specified, all dimensions default to `:increasing`.

`MaxM` specifies the total number of basis functions (including the constant intercept).
Each forward step adds one basis function, so `MaxM - 1` splits are performed.

If `min_gradient` is set to a positive value, a linear term with that slope is added in every
dimension (with appropriate sign for the direction). This ensures the function is strictly
increasing (or decreasing) everywhere with at least the specified gradient, eliminating flat
regions. The MARS basis functions are fit to the residual after removing these linear terms.

Returns a named tuple `(model, coefficients, rss)` where `model` is a `Sum_Of_Piecewise_Functions`,
`coefficients` is the NNLS coefficient vector, and `rss` is the residual sum of squares.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted NNLS fit is used at each step.
"""
function create_monotonic_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int;
                                      rel_tol::Float64 = 1e-2,
                                      directions::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
                                      min_gradient::Float64 = 0.0,
                                      weights::Union{Nothing, Vector{Float64}} = nothing)
    if isempty(directions)
        directions = Dict{Symbol,Symbol}(d => :increasing for d in x_variables)
    end
    for (dim, dir) in directions
        if dir ∉ (:increasing, :decreasing)
            error("Direction for dimension $dim must be :increasing or :decreasing, got :$dir")
        end
    end
    for dim in x_variables
        if !haskey(directions, dim)
            error("Direction must be specified for dimension $dim")
        end
    end
    if min_gradient < 0.0
        error("min_gradient must be non-negative, got $min_gradient")
    end

    # Build linear floor terms and adjust y if min_gradient > 0
    linear_funcs = PE_Function[]
    if min_gradient > 0.0
        for dim in x_variables
            sign = directions[dim] == :increasing ? 1.0 : -1.0
            push!(linear_funcs, PE_Function(sign * min_gradient, UnitMap([dim => PE_Unit(0.0, 0.0, 1)])))
        end
    end
    linear_sum = length(linear_funcs) > 0 ? Sum_Of_Functions(linear_funcs) : nothing

    # Adjust y by subtracting linear floor so MARS fits the residual
    dd_fit = dd
    y_fit = y
    if linear_sum !== nothing
        dd_fit = copy(dd)
        dd_fit[!, y] = dd[!, y] .- evaluate(linear_sum, dd)
        y_fit = y
    end

    Arr = Array{Sum_Of_Functions,length(x_variables)}(undef, repeat([1], length(x_variables))...)
    Arr[repeat([1], length(x_variables))...] = Sum_Of_Functions([PE_Function(1.0)])
    pw_func = Piecewise_Function(Arr, OrderedDict{Symbol,Array{Float64,1}}(x_variables .=> repeat([[-Inf]], length(x_variables))))
    array_of_funcs = Vector{Piecewise_Function}([pw_func])

    for M in 2:MaxM
        best_lof = Inf
        best_m = 1
        best_dimen = collect(x_variables)[1]
        best_split = 0.0
        for m in 1:length(array_of_funcs)
            underlying = underlying_dimensions(array_of_funcs[m])
            for dimen in setdiff(x_variables, underlying)
                lof, spt = optimise_monotone_split(dd_fit, y_fit, array_of_funcs, m, dimen, directions[dimen], rel_tol; weights=weights)
                if lof < best_lof
                    best_lof = lof
                    best_m = m
                    best_dimen = dimen
                    best_split = spt
                end
            end
        end
        array_of_funcs = add_split_monotone(array_of_funcs, best_m, best_dimen, best_split, directions[best_dimen])
    end

    X = hcat(evaluate.(array_of_funcs, Ref(dd_fit))...)
    y_vec = dd_fit[!, y_fit]
    coefficients = fit_nnls(X, y_vec; weights=weights)
    updated_model = Sum_Of_Piecewise_Functions(array_of_funcs .* coefficients)

    # Add linear floor back to the model
    if linear_sum !== nothing
        updated_model = updated_model + linear_sum
    end

    rss = _weighted_rss(evaluate(updated_model, dd) .- dd[!, y], weights)
    return (model = updated_model, coefficients = coefficients, rss = rss)
end
