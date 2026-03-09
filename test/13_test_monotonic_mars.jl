using MultivariateFunctions
using Random
using DataFrames
using Distributions
using DataStructures

Random.seed!(2024)
tol = 1e-6
nObs = 500

function check_monotone_increasing(model, vary_dim::Symbol, fixed_dims::Dict{Symbol,Float64};
                                    grid = collect(-3.0:0.05:3.0))
    for (dim, val) in fixed_dims
        for fval in [val - 2.0, val, val + 2.0]
            dd_test = DataFrame(Dict(vary_dim => grid, (d => fill(v, length(grid)) for (d, v) in fixed_dims)...))
            dd_test[!, dim] .= fval
            vals = evaluate(model, dd_test)
            if !all(diff(vals) .>= -tol)
                return false
            end
        end
    end
    dd_test = DataFrame(Dict(vary_dim => grid, (d => fill(v, length(grid)) for (d, v) in fixed_dims)...))
    vals = evaluate(model, dd_test)
    return all(diff(vals) .>= -tol)
end

function check_monotone_decreasing(model, vary_dim::Symbol, fixed_dims::Dict{Symbol,Float64};
                                    grid = collect(-3.0:0.05:3.0))
    for (dim, val) in fixed_dims
        for fval in [val - 2.0, val, val + 2.0]
            dd_test = DataFrame(Dict(vary_dim => grid, (d => fill(v, length(grid)) for (d, v) in fixed_dims)...))
            dd_test[!, dim] .= fval
            vals = evaluate(model, dd_test)
            if !all(diff(vals) .<= tol)
                return false
            end
        end
    end
    dd_test = DataFrame(Dict(vary_dim => grid, (d => fill(v, length(grid)) for (d, v) in fixed_dims)...))
    vals = evaluate(model, dd_test)
    return all(diff(vals) .<= tol)
end

function check_strictly_increasing(model, vary_dim::Symbol, fixed_dims::Dict{Symbol,Float64},
                                    min_grad::Float64; grid = collect(-3.0:0.05:3.0))
    step = grid[2] - grid[1]
    for (dim, val) in fixed_dims
        for fval in [val - 2.0, val, val + 2.0]
            dd_test = DataFrame(Dict(vary_dim => grid, (d => fill(v, length(grid)) for (d, v) in fixed_dims)...))
            dd_test[!, dim] .= fval
            vals = evaluate(model, dd_test)
            if !all(diff(vals) .>= min_grad * step - tol)
                return false
            end
        end
    end
    return true
end

function check_strictly_decreasing(model, vary_dim::Symbol, fixed_dims::Dict{Symbol,Float64},
                                    min_grad::Float64; grid = collect(-3.0:0.05:3.0))
    step = grid[2] - grid[1]
    for (dim, val) in fixed_dims
        for fval in [val - 2.0, val, val + 2.0]
            dd_test = DataFrame(Dict(vary_dim => grid, (d => fill(v, length(grid)) for (d, v) in fixed_dims)...))
            dd_test[!, dim] .= fval
            vals = evaluate(model, dd_test)
            if !all(diff(vals) .<= -min_grad * step + tol)
                return false
            end
        end
    end
    return true
end

# =========================================
# Test 1: Univariate monotone increasing
# =========================================
dd1 = DataFrame()
dd1[!, :x] = rand(nObs) .* 4.0 .- 2.0
dd1[!, :y] = 3.0 .* max.(0.0, dd1[!, :x] .- 0.5) .+ max.(0.0, dd1[!, :x] .+ 1.0) .+ 2.0

result1 = create_monotonic_mars_spline(dd1, :y, Set([:x]), 4; rel_tol = 1e-3)
test1_type = result1.model isa Sum_Of_Piecewise_Functions

fitted1 = evaluate(result1.model, dd1)
rss1 = sum((fitted1 .- dd1[!, :y]) .^ 2)
test1_rss = rss1 < 10.0

x_grid = collect(-3.0:0.01:3.0)
dd_grid = DataFrame(x = x_grid)
vals1 = evaluate(result1.model, dd_grid)
test1_mono = all(diff(vals1) .>= -tol)

# =========================================
# Test 2: Univariate monotone decreasing
# =========================================
dd2 = DataFrame()
dd2[!, :x] = rand(nObs) .* 4.0 .- 2.0
dd2[!, :y] = 3.0 .* max.(0.0, 0.5 .- dd2[!, :x]) .+ 2.0

result2 = create_monotonic_mars_spline(dd2, :y, Set([:x]), 3; rel_tol = 1e-3,
    directions = Dict{Symbol,Symbol}(:x => :decreasing))

vals2 = evaluate(result2.model, dd_grid)
test2_mono = all(diff(vals2) .<= tol)

# =========================================
# Test 3: Multivariate monotone increasing
# =========================================
dd3 = DataFrame()
dd3[!, :x] = rand(Normal(), nObs)
dd3[!, :z] = rand(Normal(), nObs)
dd3[!, :y] = 3.0 .* max.(0.0, dd3[!, :x] .- 0.5) .+ 2.0 .* max.(0.0, dd3[!, :z] .+ 1.0) .+ 1.0

result3 = create_monotonic_mars_spline(dd3, :y, Set([:x, :z]), 5; rel_tol = 1e-3)

test3_mono_x = check_monotone_increasing(result3.model, :x, Dict(:z => 0.0))
test3_mono_z = check_monotone_increasing(result3.model, :z, Dict(:x => 0.0))

# =========================================
# Test 4: Mixed directions (x increasing, z decreasing)
# =========================================
dd4 = DataFrame()
dd4[!, :x] = rand(Normal(), nObs)
dd4[!, :z] = rand(Normal(), nObs)
dd4[!, :y] = 2.0 .* max.(0.0, dd4[!, :x] .- 0.3) .+ 3.0 .* max.(0.0, 1.0 .- dd4[!, :z]) .+ 1.5

result4 = create_monotonic_mars_spline(dd4, :y, Set([:x, :z]), 5; rel_tol = 1e-3,
    directions = Dict{Symbol,Symbol}(:x => :increasing, :z => :decreasing))

test4_mono_x = check_monotone_increasing(result4.model, :x, Dict(:z => 0.0))
test4_mono_z = check_monotone_decreasing(result4.model, :z, Dict(:x => 0.0))

# =========================================
# Test 5: More basis functions should not increase RSS
# =========================================
result3a = create_monotonic_mars_spline(dd3, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
result3b = create_monotonic_mars_spline(dd3, :y, Set([:x, :z]), 5; rel_tol = 1e-3)
test5_rss = result3a.rss >= result3b.rss - tol

# =========================================
# Test 6: Default directions (should be all increasing)
# =========================================
result_default = create_monotonic_mars_spline(dd3, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
test6_type = result_default.model isa Sum_Of_Piecewise_Functions

# =========================================
# Test 7: min_gradient ensures strictly increasing (no flat regions)
# =========================================
min_grad = 0.01
result7 = create_monotonic_mars_spline(dd1, :y, Set([:x]), 4; rel_tol = 1e-3, min_gradient = min_grad)

vals7 = evaluate(result7.model, dd_grid)
diffs7 = diff(vals7)
step = x_grid[2] - x_grid[1]
test7_strict = all(diffs7 .>= min_grad * step - tol)

# =========================================
# Test 8: min_gradient with decreasing direction
# =========================================
result8 = create_monotonic_mars_spline(dd2, :y, Set([:x]), 3; rel_tol = 1e-3,
    directions = Dict{Symbol,Symbol}(:x => :decreasing), min_gradient = min_grad)

vals8 = evaluate(result8.model, dd_grid)
diffs8 = diff(vals8)
test8_strict = all(diffs8 .<= -min_grad * step + tol)

# =========================================
# Test 9: min_gradient with multivariate mixed directions (checked at multiple fixed values)
# =========================================
result9 = create_monotonic_mars_spline(dd4, :y, Set([:x, :z]), 5; rel_tol = 1e-3,
    directions = Dict{Symbol,Symbol}(:x => :increasing, :z => :decreasing),
    min_gradient = min_grad)

test9_x = check_strictly_increasing(result9.model, :x, Dict(:z => 0.0), min_grad)
test9_z = check_strictly_decreasing(result9.model, :z, Dict(:x => 0.0), min_grad)

# =========================================
# Test 10: Contrast - min_gradient=0 allows flat regions, min_gradient>0 does not
# =========================================
# Use data that is constant over part of the x range so the fit should be flat there
dd10 = DataFrame()
dd10[!, :x] = collect(range(-2.0, 2.0, length=nObs))
dd10[!, :y] = max.(0.0, dd10[!, :x] .- 0.5) .+ 1.0  # flat for x < 0.5

result10_flat = create_monotonic_mars_spline(dd10, :y, Set([:x]), 3; rel_tol = 1e-3)
vals10_flat = evaluate(result10_flat.model, dd_grid)
diffs10_flat = diff(vals10_flat)
has_flat = any(abs.(diffs10_flat) .< tol)  # should have some flat regions

result10_strict = create_monotonic_mars_spline(dd10, :y, Set([:x]), 3; rel_tol = 1e-3, min_gradient = 0.01)
vals10_strict = evaluate(result10_strict.model, dd_grid)
diffs10_strict = diff(vals10_strict)
no_flat = all(diffs10_strict .>= 0.01 * step - tol)  # no flat regions

test10 = has_flat && no_flat

# =========================================
# Test 11: Three dimensions
# =========================================
dd11 = DataFrame()
dd11[!, :x] = rand(Normal(), nObs)
dd11[!, :z] = rand(Normal(), nObs)
dd11[!, :w] = rand(Normal(), nObs)
dd11[!, :y] = 2.0 .* max.(0.0, dd11[!, :x] .- 0.3) .+ 1.5 .* max.(0.0, dd11[!, :z] .+ 0.5) .+ max.(0.0, dd11[!, :w]) .+ 1.0

result11 = create_monotonic_mars_spline(dd11, :y, Set([:x, :z, :w]), 6; rel_tol = 1e-3)

test11_x = check_monotone_increasing(result11.model, :x, Dict(:z => 0.0, :w => 0.0))
test11_z = check_monotone_increasing(result11.model, :z, Dict(:x => 0.0, :w => 0.0))
test11_w = check_monotone_increasing(result11.model, :w, Dict(:x => 0.0, :z => 0.0))

# =========================================
# Test 12: Error conditions
# =========================================
test12_invalid_dir = try
    create_monotonic_mars_spline(dd1, :y, Set([:x]), 3;
        directions = Dict{Symbol,Symbol}(:x => :flat))
    false
catch e
    occursin("increasing or :decreasing", e.msg)
end

test12_missing_dir = try
    create_monotonic_mars_spline(dd3, :y, Set([:x, :z]), 3;
        directions = Dict{Symbol,Symbol}(:x => :increasing))
    false
catch e
    occursin("must be specified", e.msg)
end

test12_neg_grad = try
    create_monotonic_mars_spline(dd1, :y, Set([:x]), 3; min_gradient = -0.1)
    false
catch e
    occursin("non-negative", e.msg)
end

# All tests
test1_type && test1_rss && test1_mono &&
test2_mono &&
test3_mono_x && test3_mono_z &&
test4_mono_x && test4_mono_z &&
test5_rss &&
test6_type &&
test7_strict &&
test8_strict &&
test9_x && test9_z &&
test10 &&
test11_x && test11_z && test11_w &&
test12_invalid_dir && test12_missing_dir && test12_neg_grad
