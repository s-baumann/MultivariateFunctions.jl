using MultivariateFunctions
using Random
using DataFrames
using Distributions
using DataStructures

Random.seed!(2024)
tol = 1e-4

# =========================================
# Helper: generate data with known structure
# =========================================
function make_data(nObs::Int, seed::Int)
    Random.seed!(seed)
    dd = DataFrame()
    dd[!, :x] = rand(Normal(), nObs)
    dd[!, :z] = rand(Normal(), nObs)
    dd[!, :y] = 2.0 .* max.(0.0, dd[!, :x] .- 0.3) .+ 1.5 .* max.(0.0, dd[!, :z] .+ 0.5) .+ 1.0
    return dd
end

# =========================================
# Test 1: MultivariateFitter with MARS
# =========================================
dd = make_data(300, 100)
fitter1 = MultivariateFitter(:mars, Set([:x, :z]); MaxM = 4, weight_on_new = 1.0)

MultivariateFunctions.fit!(fitter1, dd, :y)
fitted1 = evaluate(fitter1, dd)
rss1 = sum((fitted1 .- dd[!, :y]) .^ 2)
test1_fitted = rss1 < 50.0
test1_type = fitter1.fun !== nothing
test1_times = fitter1.times_through == 1

# =========================================
# Test 2: Blending reduces weight over time
# =========================================
dd2 = make_data(300, 101)
MultivariateFunctions.fit!(fitter1, dd2, :y)
test2_times = fitter1.times_through == 2

dd3 = make_data(300, 102)
MultivariateFunctions.fit!(fitter1, dd3, :y)
test2_three = fitter1.times_through == 3

# =========================================
# Test 3: MultivariateFitter with recursive partitioning
# =========================================
fitter3 = MultivariateFitter(:recursive_partitioning, Set([:x, :z]); MaxM = 4)
MultivariateFunctions.fit!(fitter3, dd, :y)
fitted3 = evaluate(fitter3, dd)
rss3 = sum((fitted3 .- dd[!, :y]) .^ 2)
test3 = rss3 < 500.0 && fitter3.times_through == 1

# =========================================
# Test 4: MultivariateFitter with monotonic MARS
# =========================================
fitter4 = MultivariateFitter(:monotonic_mars, Set([:x, :z]); MaxM = 4, rel_tol = 1e-3)
MultivariateFunctions.fit!(fitter4, dd, :y)
fitted4 = evaluate(fitter4, dd)
rss4 = sum((fitted4 .- dd[!, :y]) .^ 2)
test4 = rss4 < 100.0

# =========================================
# Test 5: MultivariateFitter with OLS (hyperplane)
# =========================================
fitter5 = MultivariateFitter(:ols, Set([:x, :z]))
MultivariateFunctions.fit!(fitter5, dd, :y)
fitted5 = evaluate(fitter5, dd)
test5 = fitter5.fun !== nothing && fitter5.times_through == 1

# =========================================
# Test 6: MultivariateFitter with saturated OLS
# =========================================
fitter6 = MultivariateFitter(:saturated_ols, Set([:x, :z]); ols_degree = 2)
MultivariateFunctions.fit!(fitter6, dd, :y)
fitted6 = evaluate(fitter6, dd)
test6 = fitter6.fun !== nothing

# =========================================
# Test 7: Simplification with trim_mars_spline
# =========================================
fitter7 = MultivariateFitter(:mars, Set([:x, :z]);
    MaxM = 5, simplify_to = 3, simplification_frequency = 2, weight_on_new = 1.0)

MultivariateFunctions.fit!(fitter7, dd, :y)   # times_through becomes 1, no simplification
MultivariateFunctions.fit!(fitter7, dd2, :y)  # times_through becomes 2, simplification triggers

test7 = fitter7.fun !== nothing && fitter7.times_through == 2

# =========================================
# Test 8: Evaluate with Dict
# =========================================
val8 = evaluate(fitter1, Dict(:x => 1.0, :z => 0.5))
test8 = isa(val8, Float64)

# =========================================
# Test 9: Evaluate before any fit returns zeros
# =========================================
fitter9 = MultivariateFitter(:mars, Set([:x, :z]))
vals9 = evaluate(fitter9, dd)
test9 = all(vals9 .== 0.0)

val9b = evaluate(fitter9, Dict(:x => 1.0, :z => 0.5))
test9b = val9b == 0.0

# =========================================
# Test 10: MultivariateAdjustedFitter basic
# =========================================
Random.seed!(2024)
nObs = 300
dd10 = DataFrame()
dd10[!, :x] = rand(Normal(), nObs)
dd10[!, :z] = rand(Normal(), nObs)
# True shape: f(x,z) = 2*max(0, x-0.3) + 1.5*max(0, z+0.5) + 1
true_shape = 2.0 .* max.(0.0, dd10[!, :x] .- 0.3) .+ 1.5 .* max.(0.0, dd10[!, :z] .+ 0.5) .+ 1.0
# Two groups with different affine adjustments
groups10 = vcat(fill(:A, 150), fill(:B, 150))
dd10[!, :y] = similar(true_shape)
for i in 1:nObs
    if groups10[i] == :A
        dd10[i, :y] = 0.5 + 1.2 * true_shape[i] + 0.1 * randn()
    else
        dd10[i, :y] = -0.3 + 0.8 * true_shape[i] + 0.1 * randn()
    end
end

fitter10 = MultivariateAdjustedFitter(:mars, Set([:x, :z]);
    MaxM = 4, weight_on_new = 1.0,
    coefficient_bounds = ((-2.0, 2.0), (0.1, 3.0)))

MultivariateFunctions.fit!(fitter10, dd10, :y, groups10)
test10_times = fitter10.times_through == 1
test10_groups = haskey(fitter10.coefficients, :A) && haskey(fitter10.coefficients, :B)
test10_fun = fitter10.fun !== nothing

# Evaluate for each group
vals10_A = evaluate(fitter10, dd10[1:10, :], :A)
vals10_B = evaluate(fitter10, dd10[1:10, :], :B)
test10_different = !all(vals10_A .≈ vals10_B)  # different groups should give different results

# =========================================
# Test 11: MultivariateAdjustedFitter multiple fits
# =========================================
dd11 = make_data(300, 200)
dd11[!, :y] = similar(dd11[!, :y])
groups11 = vcat(fill(:A, 150), fill(:B, 150))
true_shape11 = 2.0 .* max.(0.0, dd11[!, :x] .- 0.3) .+ 1.5 .* max.(0.0, dd11[!, :z] .+ 0.5) .+ 1.0
for i in 1:300
    if groups11[i] == :A
        dd11[i, :y] = 0.5 + 1.0 * true_shape11[i]
    else
        dd11[i, :y] = -0.3 + 0.8 * true_shape11[i]
    end
end

MultivariateFunctions.fit!(fitter10, dd11, :y, groups11)
test11 = fitter10.times_through == 2

# =========================================
# Test 12: MultivariateAdjustedFitter with no intercept
# =========================================
fitter12 = MultivariateAdjustedFitter(:mars, Set([:x, :z]);
    MaxM = 4, fit_intercept = false,
    coefficient_bounds = ((-2.0, 2.0), (0.1, 3.0)))

MultivariateFunctions.fit!(fitter12, dd10, :y, groups10)
# With fit_intercept=false, a should be forced to 0
a_A, b_A = fitter12.coefficients[:A]
a_B, b_B = fitter12.coefficients[:B]
test12 = a_A == 0.0 && a_B == 0.0

# =========================================
# Test 13: MultivariateAdjustedFitter evaluate with Dict
# =========================================
val13 = evaluate(fitter10, Dict(:x => 1.0, :z => 0.5), :A)
test13 = isa(val13, Float64)

# =========================================
# Test 14: MultivariateAdjustedFitter evaluate for unknown group uses defaults
# =========================================
val14 = evaluate(fitter10, Dict(:x => 1.0, :z => 0.5), :UNKNOWN)
# Default coefficients are (0.0, 1.0), so result = 0.0 + 1.0 * f(x,z) = f(x,z)
val14_shape = evaluate(fitter10.fun, Dict(:x => 1.0, :z => 0.5))
test14 = abs(val14 - val14_shape) < tol

# =========================================
# Test 15: Error on invalid method
# =========================================
test15_fitter = try
    MultivariateFitter(:invalid, Set([:x]))
    false
catch e
    occursin("Unknown method", e.msg)
end

test15_adj = try
    MultivariateAdjustedFitter(:invalid, Set([:x]))
    false
catch e
    occursin("Unknown method", e.msg)
end

# =========================================
# Test 16: Simplification with monotonic MARS trim
# =========================================
fitter16 = MultivariateFitter(:monotonic_mars, Set([:x, :z]);
    MaxM = 4, simplify_to = 3, simplification_frequency = 2,
    rel_tol = 1e-3, weight_on_new = 1.0)

MultivariateFunctions.fit!(fitter16, dd, :y)
MultivariateFunctions.fit!(fitter16, dd2, :y)  # triggers simplification
test16 = fitter16.fun !== nothing && fitter16.times_through == 2

# =========================================
# Test 17: Evaluate unfitted adjusted fitter returns fill
# =========================================
fitter17 = MultivariateAdjustedFitter(:mars, Set([:x, :z]))
vals17 = evaluate(fitter17, dd[1:5, :], :A)
test17 = all(vals17 .== 0.0)

# =========================================
# Test 18: Callable syntax for MultivariateFitter
# =========================================
val18_df = fitter1(dd[1:3, :])
val18_dict = fitter1(Dict(:x => 1.0, :z => 0.5))
test18 = length(val18_df) == 3 && isa(val18_dict, Float64)

# =========================================
# Test 19: Callable syntax for MultivariateAdjustedFitter
# =========================================
val19_df = fitter10(dd10[1:3, :], :A)
val19_dict = fitter10(Dict(:x => 1.0, :z => 0.5), :B)
test19 = length(val19_df) == 3 && isa(val19_dict, Float64)

# All tests
test1_fitted && test1_type && test1_times &&
test2_times && test2_three &&
test3 &&
test4 &&
test5 &&
test6 &&
test7 &&
test8 &&
test9 && test9b &&
test10_times && test10_groups && test10_fun && test10_different &&
test11 &&
test12 &&
test13 &&
test14 &&
test15_fitter && test15_adj &&
test16 &&
test17 &&
test18 &&
test19
