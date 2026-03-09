using MultivariateFunctions
using Random
using DataFrames
using Distributions
using DataStructures

Random.seed!(2024)
tol = 1e-6

nObs = 200
dd = DataFrame()
dd[!, :x] = rand(Normal(), nObs)
dd[!, :z] = rand(Normal(), nObs)
dd[!, :y] = 2.0 .* max.(0.0, dd[!, :x] .- 0.3) .+ 1.5 .* max.(0.0, dd[!, :z] .+ 0.5) .+ 1.0

# =========================================
# Test 1: Uniform weights = no weights (OLS)
# =========================================
uniform_w = ones(nObs)

dd1 = DataFrame(default = dd[!, :x], y = dd[!, :y])
model1_nw, reg1_nw = create_ols_approximation(dd[!, :y], dd[!, :x], 2)
model1_uw, reg1_uw = create_ols_approximation(dd[!, :y], dd[!, :x], 2; weights = uniform_w)

fitted1_nw = evaluate.(Ref(model1_nw), dd[!, :x])
fitted1_uw = evaluate.(Ref(model1_uw), dd[!, :x])
test1 = maximum(abs.(fitted1_nw .- fitted1_uw)) < tol

# =========================================
# Test 2: Uniform weights = no weights (saturated OLS)
# =========================================
model2_nw = create_saturated_ols_approximation(dd, :y, [:x, :z], 2)
model2_uw = create_saturated_ols_approximation(dd, :y, [:x, :z], 2; weights = uniform_w)

fitted2_nw = evaluate(model2_nw[1], dd)
fitted2_uw = evaluate(model2_uw[1], dd)
test2 = maximum(abs.(fitted2_nw .- fitted2_uw)) < tol

# =========================================
# Test 3: Uniform weights = no weights (MARS)
# =========================================
Random.seed!(42)
model3_nw = create_mars_spline(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
Random.seed!(42)
model3_uw = create_mars_spline(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3, weights = uniform_w)

fitted3_nw = evaluate(model3_nw.model, dd)
fitted3_uw = evaluate(model3_uw.model, dd)
test3 = maximum(abs.(fitted3_nw .- fitted3_uw)) < tol

# =========================================
# Test 4: Uniform weights = no weights (recursive partitioning)
# =========================================
Random.seed!(42)
model4_nw = create_recursive_partitioning(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
Random.seed!(42)
model4_uw = create_recursive_partitioning(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3, weights = uniform_w)

fitted4_nw = evaluate(model4_nw.model, dd)
fitted4_uw = evaluate(model4_uw.model, dd)
test4 = maximum(abs.(fitted4_nw .- fitted4_uw)) < tol

# =========================================
# Test 5: Uniform weights = no weights (monotonic MARS)
# =========================================
Random.seed!(42)
model5_nw = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
Random.seed!(42)
model5_uw = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3, weights = uniform_w)

fitted5_nw = evaluate(model5_nw.model, dd)
fitted5_uw = evaluate(model5_uw.model, dd)
test5 = maximum(abs.(fitted5_nw .- fitted5_uw)) < tol

# =========================================
# Test 6: Zero-weighted outliers are ignored (OLS)
# =========================================
dd6 = copy(dd)
# Add extreme outliers at the end
n_outliers = 20
dd6_out = DataFrame()
dd6_out[!, :x] = fill(0.0, n_outliers)
dd6_out[!, :z] = fill(0.0, n_outliers)
dd6_out[!, :y] = fill(1000.0, n_outliers)
dd6 = vcat(dd6, dd6_out)

w6 = vcat(ones(nObs), fill(1e-10, n_outliers))

model6_clean = create_saturated_ols_approximation(dd, :y, [:x, :z], 2)
model6_weighted = create_saturated_ols_approximation(dd6, :y, [:x, :z], 2; weights = w6)

# With zero-weighted outliers, the fit should be similar to the clean data fit
fitted6_clean = evaluate(model6_clean[1], dd)
fitted6_weighted = evaluate(model6_weighted[1], dd)
test6 = maximum(abs.(fitted6_clean .- fitted6_weighted)) < 1e-4

# =========================================
# Test 7: Zero-weighted outliers ignored (MARS)
# =========================================
Random.seed!(42)
model7_clean = create_mars_spline(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
Random.seed!(42)
model7_weighted = create_mars_spline(dd6, :y, Set([:x, :z]), 3; rel_tol = 1e-3, weights = w6)

fitted7_clean = evaluate(model7_clean.model, dd)
fitted7_weighted = evaluate(model7_weighted.model, dd)
test7 = maximum(abs.(fitted7_clean .- fitted7_weighted)) < 1e-4

# =========================================
# Test 8: Zero-weighted outliers ignored (monotonic MARS)
# =========================================
Random.seed!(42)
model8_clean = create_monotonic_mars_spline(dd, :y, Set([:x, :z]), 3; rel_tol = 1e-3)
Random.seed!(42)
model8_weighted = create_monotonic_mars_spline(dd6, :y, Set([:x, :z]), 3; rel_tol = 1e-3, weights = w6)

fitted8_clean = evaluate(model8_clean.model, dd)
fitted8_weighted = evaluate(model8_weighted.model, dd)
test8 = maximum(abs.(fitted8_clean .- fitted8_weighted)) < 1e-4

# =========================================
# Test 9: Uniform weights = no weights (trim_mars_spline)
# =========================================
model9_base = create_mars_spline(dd, :y, Set([:x, :z]), 5; rel_tol = 1e-3)

model9_nw = trim_mars_spline(dd, :y, model9_base.model; final_number_of_functions = 3)
model9_uw = trim_mars_spline(dd, :y, model9_base.model; final_number_of_functions = 3, weights = uniform_w)

fitted9_nw = evaluate(model9_nw.model, dd)
fitted9_uw = evaluate(model9_uw.model, dd)
test9 = maximum(abs.(fitted9_nw .- fitted9_uw)) < tol

# =========================================
# Test 10: Weighted fitter works
# =========================================
fitter10 = MultivariateFitter(:mars, Set([:x, :z]); MaxM = 3, weight_on_new = 1.0)
w10 = rand(nObs) .+ 0.1  # random positive weights
MultivariateFunctions.fit!(fitter10, dd, :y; weights = w10)
test10 = fitter10.fun !== nothing && fitter10.times_through == 1

# =========================================
# Test 11: Weighted adjusted fitter works
# =========================================
groups11 = vcat(fill(:A, div(nObs, 2)), fill(:B, nObs - div(nObs, 2)))
fitter11 = MultivariateAdjustedFitter(:mars, Set([:x, :z]);
    MaxM = 3, weight_on_new = 1.0,
    coefficient_bounds = ((-5.0, 5.0), (0.1, 5.0)))
w11 = rand(nObs) .+ 0.1
MultivariateFunctions.fit!(fitter11, dd, :y, groups11; weights = w11)
test11 = fitter11.fun !== nothing && haskey(fitter11.coefficients, :A)

# =========================================
# Test 12: Double weights ≈ duplicating rows (OLS)
# =========================================
# Duplicate the first half of the data
half = div(nObs, 2)
dd12_dup = vcat(dd, dd[1:half, :])
w12 = vcat(fill(2.0, half), ones(nObs - half))

model12_dup = create_saturated_ols_approximation(dd12_dup, :y, [:x, :z], 1)
model12_w = create_saturated_ols_approximation(dd, :y, [:x, :z], 1; weights = w12)

fitted12_dup = evaluate(model12_dup[1], dd)
fitted12_w = evaluate(model12_w[1], dd)
test12 = maximum(abs.(fitted12_dup .- fitted12_w)) < 1e-4

# All tests
test1 && test2 && test3 && test4 && test5 &&
test6 && test7 && test8 && test9 &&
test10 && test11 && test12
