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

function optimise_split(dd::DataFrame, y::Symbol, array_of_funcs::Array, ind::Int, split_variable::Symbol, rel_tol::Float64, SplitFunction::Function, removeSplitFunction::Bool; weights::Union{Nothing, Vector{Float64}} = nothing, X_all::Union{Nothing,Matrix{Float64}} = nothing)
    # The search bracket is strictly inside the data range so a split at the boundary
    # cannot produce an all-zero basis column.
    lower_limit = nextfloat(minimum(dd[!, split_variable]))
    upper_limit = prevfloat(maximum(dd[!, split_variable]))

    # Pre-compute design matrix columns for basis functions that don't change.
    # Callers looping over candidate splits can pass the design matrix of all of
    # array_of_funcs as X_all so it is not rebuilt for every candidate.
    y_vec = dd[!, y]
    if X_all === nothing
        if length(array_of_funcs) > 0
            X_all = hcat(evaluate.(array_of_funcs, Ref(dd))...)
        else
            X_all = Matrix{Float64}(undef, size(dd, 1), 0)
        end
    end
    if removeSplitFunction
        X_cached = X_all[:, vcat(1:(ind-1), (ind+1):length(array_of_funcs))]
    else
        X_cached = X_all
    end

    # Precompute the normal equation blocks for the unchanged columns. Each candidate split
    # point then only needs the cross products involving the 2 new columns and a small
    # (k+2)x(k+2) solve, instead of a full OLS refit.
    k = size(X_cached, 2)
    sqrtW = weights === nothing ? nothing : sqrt.(weights)
    Xw = weights === nothing ? X_cached : sqrtW .* X_cached
    yw = weights === nothing ? y_vec : sqrtW .* y_vec
    G = Xw' * Xw
    c_vec = Xw' * yw

    opt = optimize(lower_limit, upper_limit; rel_tol = rel_tol) do split_point
        # Only construct and evaluate the 2 new basis functions
        model = SplitFunction(array_of_funcs, ind, split_variable, split_point, removeSplitFunction)
        new_funcs = model[end-1:end]
        X_new = hcat(evaluate.(new_funcs, Ref(dd))...)
        X_new_w = weights === nothing ? X_new : sqrtW .* X_new
        A = Matrix{Float64}(undef, k + 2, k + 2)
        A[1:k, 1:k] = G
        cross = Xw' * X_new_w
        A[1:k, (k+1):(k+2)] = cross
        A[(k+1):(k+2), 1:k] = cross'
        A[(k+1):(k+2), (k+1):(k+2)] = X_new_w' * X_new_w
        b = vcat(c_vec, X_new_w' * yw)
        # A can be singular, for instance when a candidate basis function is zero at every
        # data point, so check the factorization rather than letting the solve throw.
        A_factor = cholesky(Hermitian(A); check = false)
        if issuccess(A_factor)
            beta = A_factor \ b
        elseif weights !== nothing
            # Near-zero weights can make the weighted design matrix rank-deficient;
            # treat such a candidate as invalid.
            return Inf
        else
            beta = pinv(A) * b
        end
        # RSS from explicit residuals rather than the y'y - b'beta identity, which loses
        # precision through cancellation when the fit is near-exact.
        residuals_ = y_vec .- X_cached * beta[1:k] .- X_new * beta[(k+1):(k+2)]
        return _weighted_rss(residuals_, weights)
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
        X_all = hcat(evaluate.(array_of_funcs, Ref(dd))...)
        for m in 1:length(array_of_funcs)
            for dimen in x_variables
                lof, spt = optimise_split(dd, y, array_of_funcs, m, dimen, rel_tol, add_split_with_step_function, true; weights=weights, X_all=X_all)
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
    create_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int; rel_tol::Float64 = 1e-2, weights = nothing, fast_k::Int = 0)

This creates a mars spline given a dataframe, response variable and a set of x_variables from the dataframe.
The relative tolerance is used in a one-dimensional optimisation step to determine what points at which split values to place
a max(0,x-split) function in a particular dimension. The default is intentionally set high because precision is generally not the
not that important. For small scale data however you might want to decrease it and increase it for large scale data. You might also want to
decrease it if spline creation time doesnt matter much. Note that a small rel_tol only affects creation time for the spline and
not the evaluation time.

Returns a named tuple `(model, regression)` where `model` is a `Sum_Of_Piecewise_Functions` and `regression` is the GLM `LinearModel` object.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted least squares fit is used at each step.

If `fast_k` is positive then the Fast MARS idea of Friedman (1993,
https://statistics.stanford.edu/sites/default/files/LCS%20110.pdf) is used: in each forward step only
the `fast_k` most promising parent basis functions (ranked by the improvement found when each was last
searched, with never-searched parents always included) are searched for a new split. The default of 0
searches every parent every step (the exact algorithm). A positive `fast_k` can speed up fitting
considerably when MaxM is large, at the cost of possibly selecting slightly different splits.
"""
function create_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int; rel_tol::Float64 = 1e-2, weights::Union{Nothing, Vector{Float64}} = nothing, fast_k::Int = 0)
    Arr = Array{Sum_Of_Functions,length(x_variables)}(undef, repeat([1], length(x_variables))...)
    Arr[repeat([1], length(x_variables))...] = Sum_Of_Functions([PE_Function(1.0)])
    pw_func = Piecewise_Function(Arr, OrderedDict{Symbol,Array{Float64,1}}(x_variables .=> repeat([[-Inf]],length(x_variables))) )
    array_of_funcs = Vector{Piecewise_Function}([pw_func])
    parent_last_lof = Float64[-Inf] # -Inf marks a parent that has never been searched.
    for M in 2:MaxM
        best_lof    = Inf
        best_m      = 1
        best_dimen  = collect(x_variables)[1]
        best_split  = 0.0
        X_all = hcat(evaluate.(array_of_funcs, Ref(dd))...)
        searchable = [m for m in 1:length(array_of_funcs) if !isempty(setdiff(x_variables, underlying_dimensions(array_of_funcs[m])))]
        if (fast_k > 0) && (length(searchable) > fast_k)
            searchable = searchable[partialsortperm(parent_last_lof[searchable], 1:fast_k)]
        end
        for m in searchable
            underlying = underlying_dimensions(array_of_funcs[m])
            best_for_parent = Inf
            for dimen in setdiff(x_variables, underlying)
                lof, spt = optimise_split(dd, y, array_of_funcs, m, dimen, rel_tol, add_split_with_max_function, false; weights=weights, X_all=X_all)
                best_for_parent = min(best_for_parent, lof)
                if lof < best_lof
                    best_lof = lof
                    best_m = m
                    best_dimen = dimen
                    best_split = spt
                end
            end
            parent_last_lof[m] = best_for_parent
        end
        array_of_funcs = add_split_with_max_function(array_of_funcs, best_m, best_dimen, best_split, false)
        push!(parent_last_lof, -Inf)
        push!(parent_last_lof, -Inf)
    end
    updated_model, reg = create_ols_approximation(dd, y, array_of_funcs; weights=weights)
    return (model = updated_model, regression = reg)
end

# Compute OLS RSS for a subset of columns from a precomputed design matrix.
function _ols_rss_for_columns(X_full::Matrix{Float64}, y_vec::Vector{Float64}, cols::Vector{Int}, weights::Union{Nothing, Vector{Float64}})
    X = X_full[:, cols]
    if weights === nothing
        beta = X \ y_vec
    else
        sqrtW = sqrt.(weights)
        beta = (sqrtW .* X) \ (sqrtW .* y_vec)
    end
    return _weighted_rss(X * beta .- y_vec, weights)
end

# OLS backward deletion using the exact ΔRSS formula from partitioned regression.
# When column j is removed, the increase in weighted RSS is exactly:
#   ΔRSS_j = β_j² / [(X'WX)⁻¹]_{jj}
# This avoids refitting for each candidate — one solve per deletion round
# reduces cost from O(p) regressions to O(1) per round.
function _backward_delete(X_full::Matrix{Float64}, y_vec::Vector{Float64},
                           weights::Union{Nothing, Vector{Float64}},
                           keep_cols::Vector{Int},
                           should_stop::Function)
    while length(keep_cols) > 1
        X = X_full[:, keep_cols]
        if weights === nothing
            XtX = X' * X
            Xty = X' * y_vec
        else
            sqrtW = sqrt.(weights)
            Xw = sqrtW .* X
            yw = sqrtW .* y_vec
            XtX = Xw' * Xw
            Xty = Xw' * yw
        end
        # Factorize once and branch on success (checking with isposdef and then solving
        # with a different factorization can throw on PSD-singular matrices).
        XtX_factor = cholesky(Hermitian(XtX); check = false)
        C = issuccess(XtX_factor) ? inv(XtX_factor) : pinv(XtX)
        beta = C * Xty
        current_rss = _weighted_rss(X * beta .- y_vec, weights)

        # Find the non-intercept column whose removal increases RSS the least
        best_delta = Inf
        best_idx = 2
        for m in 2:length(keep_cols)
            delta = beta[m]^2 / C[m, m]
            # A zero coefficient with a zero pinv diagonal (an all-zero basis column,
            # e.g. a zero-coefficient piece of a monotonic MARS model) gives 0/0;
            # removing such a column is free.
            if isnan(delta)
                delta = 0.0
            end
            if delta < best_delta
                best_delta = delta
                best_idx = m
            end
        end

        if should_stop(current_rss + best_delta)
            break
        end
        keep_cols = vcat(keep_cols[1:best_idx-1], keep_cols[best_idx+1:end])
    end
    return keep_cols
end

# The trimming functions only delete piecewise basis functions; any global (non-piecewise)
# terms of the model are held fixed. Their contribution is subtracted from the target for
# all the RSS comparisons and refits, and they are added back onto the returned model.
function _target_minus_globals(dd::DataFrame, y::Symbol, globals::Sum_Of_Functions)
    y_vec = Vector{Float64}(dd[!, y])
    if length(globals.functions_) > 0
        y_vec = y_vec .- evaluate(globals, dd)
    end
    return y_vec
end

# Final OLS refit for a trimmed model. With nonempty globals the pieces are refit to the
# adjusted target and the globals are added back, so the returned regression object is the
# fit on y minus the globals' contribution.
function _trim_refit(dd::DataFrame, y::Symbol, funcs::Array, globals::Sum_Of_Functions; weights::Union{Nothing, Vector{Float64}} = nothing)
    if length(globals.functions_) == 0
        updated_model, reg = create_ols_approximation(dd, y, funcs; weights=weights)
        return (model = updated_model, regression = reg)
    end
    dd_adj = copy(dd)
    dd_adj[!, y] = dd[!, y] .- evaluate(globals, dd)
    updated_model, reg = create_ols_approximation(dd_adj, y, funcs; weights=weights)
    return (model = updated_model + globals, regression = reg)
end

function trim_mars_spline_final_number_of_functions(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions, final_number_of_functions::Int; weights::Union{Nothing, Vector{Float64}} = nothing)
    if final_number_of_functions < 2
        error("Cannot trim the number of functions to less than 2")
    end
    array_of_funcs = model.functions_
    n_funcs = length(array_of_funcs)
    functions_to_delete = n_funcs - final_number_of_functions
    if functions_to_delete <= 0
        return _trim_refit(dd, y, array_of_funcs, model.global_funcs_; weights=weights)
    end
    X_full = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    y_vec = _target_minus_globals(dd, y, model.global_funcs_)
    keep_cols = collect(1:n_funcs)
    deleted = Ref(0)
    keep_cols = _backward_delete(X_full, y_vec, weights, keep_cols,
                                  _ -> (deleted[] += 1; deleted[] > functions_to_delete))
    return _trim_refit(dd, y, array_of_funcs[keep_cols], model.global_funcs_; weights=weights)
end
function trim_mars_spline_maximum_increase_in_RSS(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions, maximum_increase_in_RSS::Float64; weights::Union{Nothing, Vector{Float64}} = nothing)
    array_of_funcs = model.functions_
    n_funcs = length(array_of_funcs)
    X_full = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    y_vec = _target_minus_globals(dd, y, model.global_funcs_)
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
    return _trim_refit(dd, y, array_of_funcs[keep_cols], model.global_funcs_; weights=weights)
end
function trim_mars_spline_maximum_RSS(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions, maximum_RSS::Float64; weights::Union{Nothing, Vector{Float64}} = nothing)
    array_of_funcs = model.functions_
    n_funcs = length(array_of_funcs)
    X_full = hcat(evaluate.(array_of_funcs, Ref(dd))...)
    y_vec = _target_minus_globals(dd, y, model.global_funcs_)
    keep_cols = collect(1:n_funcs)
    keep_cols = _backward_delete(X_full, y_vec, weights, keep_cols,
                                  best_rss -> best_rss >= maximum_RSS)
    return _trim_refit(dd, y, array_of_funcs[keep_cols], model.global_funcs_; weights=weights)
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

If the model carries global (non-piecewise) terms in `global_funcs_` (for instance the linear floor
terms of a monotonic MARS spline with `min_gradient > 0`) these are held fixed: the piecewise basis is
refit to `y` minus the globals' contribution and the globals are added back onto the trimmed model, so
RSS criteria refer to the full model's residuals. Note this OLS trim does not preserve a monotonic
model's monotonicity guarantee — use the fitter's simplification path for that.

Returns a named tuple `(model, regression)` where `model` is the trimmed `Sum_Of_Piecewise_Functions` and `regression` is the final GLM `LinearModel` object (the fit on the globals-adjusted target when globals are present).

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
    β2 = _nnls_coordinate_descent(AtA, Atb)
    # Recover intercept
    β1 = y_mean - sum(X2_means .* β2)
    return vcat([β1], β2)
end

# Coordinate descent for min ||Aβ - b||² s.t. β >= 0, expressed through the normal
# equation blocks AtA = A'A and Atb = A'b.
function _nnls_coordinate_descent(AtA::Matrix{Float64}, Atb::Vector{Float64})
    p2 = length(Atb)
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
    return β2
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
                                 weights::Union{Nothing, Vector{Float64}} = nothing,
                                 X_all::Union{Nothing,Matrix{Float64}} = nothing)
    lower_limit = nextfloat(minimum(dd[!, split_variable]))
    upper_limit = prevfloat(maximum(dd[!, split_variable]))
    y_vec = dd[!, y]
    X_cached = X_all === nothing ? hcat(evaluate.(array_of_funcs, Ref(dd))...) : X_all

    # Precompute the centered NNLS blocks for the unchanged columns (column 1 is the
    # constant basis whose coefficient is the unconstrained intercept, mirroring fit_nnls).
    # Each candidate split point then only needs the cross products of its single new
    # column, instead of rebuilding the full Gram matrix.
    Z = X_cached[:, 2:end]
    sqrtW = weights === nothing ? nothing : sqrt.(weights)
    W = weights === nothing ? Float64(length(y_vec)) : sum(weights)
    if weights === nothing
        mu = vec(sum(Z, dims = 1)) ./ W
        y_mean = sum(y_vec) / W
    else
        mu = vec(sum(weights .* Z, dims = 1)) ./ W
        y_mean = sum(weights .* y_vec) / W
    end
    Z_c = Z .- mu'
    y_c = y_vec .- y_mean
    Z_cw = weights === nothing ? Z_c : sqrtW .* Z_c
    y_cw = weights === nothing ? y_c : sqrtW .* y_c
    G_c = Z_cw' * Z_cw
    h_c = Z_cw' * y_cw
    p2 = size(Z, 2) + 1

    opt = optimize(lower_limit, upper_limit; rel_tol = rel_tol) do split_point
        model = add_split_monotone(array_of_funcs, ind, split_variable, split_point, direction)
        v = evaluate(model[end], dd)
        v_mean = weights === nothing ? sum(v) / W : sum(weights .* v) / W
        v_c = v .- v_mean
        v_cw = weights === nothing ? v_c : sqrtW .* v_c
        AtA = Matrix{Float64}(undef, p2, p2)
        AtA[1:(p2-1), 1:(p2-1)] = G_c
        gcross = Z_cw' * v_cw
        AtA[1:(p2-1), p2] = gcross
        AtA[p2, 1:(p2-1)] = gcross
        AtA[p2, p2] = sum(abs2, v_cw)
        Atb = vcat(h_c, sum(v_cw .* y_cw))
        beta2 = _nnls_coordinate_descent(AtA, Atb)
        beta1 = y_mean - sum(mu .* beta2[1:(p2-1)]) - v_mean * beta2[p2]
        residuals_ = y_vec .- beta1 .- Z * beta2[1:(p2-1)] .- v .* beta2[p2]
        return _weighted_rss(residuals_, weights)
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
Basis functions that receive a zero NNLS coefficient are dropped from the model (they contribute
nothing), so `model` can hold fewer than `MaxM` pieces; `coefficients` aligns with the retained
`model.functions_`.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted NNLS fit is used at each step.

If `fast_k` is positive then only the `fast_k` most promising parent basis functions are searched in
each forward step (see `create_mars_spline` for details). The default of 0 searches every parent.
"""
function create_monotonic_mars_spline(dd::DataFrame, y::Symbol, x_variables::Set{Symbol}, MaxM::Int;
                                      rel_tol::Float64 = 1e-2,
                                      directions::Dict{Symbol,Symbol} = Dict{Symbol,Symbol}(),
                                      min_gradient::Float64 = 0.0,
                                      weights::Union{Nothing, Vector{Float64}} = nothing,
                                      fast_k::Int = 0)
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
    parent_last_lof = Float64[-Inf] # -Inf marks a parent that has never been searched.

    for M in 2:MaxM
        best_lof = Inf
        best_m = 1
        best_dimen = collect(x_variables)[1]
        best_split = 0.0
        X_all = hcat(evaluate.(array_of_funcs, Ref(dd_fit))...)
        searchable = [m for m in 1:length(array_of_funcs) if !isempty(setdiff(x_variables, underlying_dimensions(array_of_funcs[m])))]
        if (fast_k > 0) && (length(searchable) > fast_k)
            searchable = searchable[partialsortperm(parent_last_lof[searchable], 1:fast_k)]
        end
        for m in searchable
            underlying = underlying_dimensions(array_of_funcs[m])
            best_for_parent = Inf
            for dimen in setdiff(x_variables, underlying)
                lof, spt = optimise_monotone_split(dd_fit, y_fit, array_of_funcs, m, dimen, directions[dimen], rel_tol; weights=weights, X_all=X_all)
                best_for_parent = min(best_for_parent, lof)
                if lof < best_lof
                    best_lof = lof
                    best_m = m
                    best_dimen = dimen
                    best_split = spt
                end
            end
            parent_last_lof[m] = best_for_parent
        end
        array_of_funcs = add_split_monotone(array_of_funcs, best_m, best_dimen, best_split, directions[best_dimen])
        push!(parent_last_lof, -Inf)
    end

    X = hcat(evaluate.(array_of_funcs, Ref(dd_fit))...)
    y_vec = dd_fit[!, y_fit]
    coefficients = fit_nnls(X, y_vec; weights=weights)
    # NNLS typically zeroes many of the basis functions. Zero pieces are dropped from the
    # model: they contribute nothing, slow evaluation down and make design matrices built
    # from the pieces singular downstream. The returned coefficients align with the
    # retained model.functions_.
    keep = findall(c -> c != 0.0, coefficients)
    coefficients = coefficients[keep]
    updated_model = Sum_Of_Piecewise_Functions(array_of_funcs[keep] .* coefficients)

    # Add linear floor back to the model
    if linear_sum !== nothing
        updated_model = updated_model + linear_sum
    end

    rss = _weighted_rss(evaluate(updated_model, dd) .- dd[!, y], weights)
    return (model = updated_model, coefficients = coefficients, rss = rss)
end
