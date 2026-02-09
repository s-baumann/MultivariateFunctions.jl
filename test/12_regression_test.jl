using MultivariateFunctions
using SchumakerSpline: Schumaker
using DataStructures: OrderedDict
using DataFrames
using Random

# Regression test: snapshot exact numerical outputs from key operations.
# Any unintended numerical change will be caught by these tight tolerances.
const regression_tol = 1e-12

function approx_eq(a::Float64, b::Float64; tol::Float64 = regression_tol)
    return abs(a - b) < tol + tol * abs(b)
end

all_passed = true

# ============================================================
# A. PE_Function evaluation & algebra
# ============================================================

f1 = PE_Function(3.0, 1.5, 2.0, 3)
f2 = PE_Function(2.0, 0.5, 1.0, 2)

all_passed &= approx_eq(evaluate(f1, 5.0), 7291.387635342266)
all_passed &= approx_eq(evaluate(f1 + f1, 5.0), 14582.775270684531)
all_passed &= approx_eq(evaluate(f1 * 2.0, 5.0), 14582.775270684531)
all_passed &= approx_eq(evaluate(f1 * f1, 5.0), 5.316433364882209e7)
all_passed &= approx_eq(evaluate(f1^2, 5.0), 5.316433364882209e7)
all_passed &= approx_eq(evaluate(f2, 5.0), 236.4497951657808)
all_passed &= approx_eq(evaluate(f1 + f2, 5.0), 7527.837430508046)
all_passed &= approx_eq(evaluate(f1 * f2, 5.0), 1.7240471128509862e6)

# ============================================================
# B. Derivatives and integrals (univariate)
# ============================================================

all_passed &= approx_eq(evaluate(derivative(f1), 5.0), 18228.469088355665)
all_passed &= approx_eq(evaluate(derivative(derivative(f1)), 5.0), 43140.71017577508)
all_passed &= approx_eq(integral(f2, 1.0, 4.0), 57.63378140676127)

# ============================================================
# C. Multivariate PE_Function
# ============================================================

unit_x = PE_Unit(0.5, 1.0, 2)
unit_y = PE_Unit(0.0, 0.0, 3)
unit_z = PE_Unit(-0.3, 2.0, 1)
f_multi = PE_Function(2.5, Dict{Symbol,PE_Unit}(:x => unit_x, :y => unit_y, :z => unit_z))
coords = Dict{Symbol,Float64}(:x => 3.0, :y => 1.5, :z => 4.0)

all_passed &= approx_eq(evaluate(f_multi, coords), 100.69816709078574)

# Partial derivative w.r.t. :x
df_multi_x = derivative(f_multi, Dict{Symbol,Int}(:x => 1))
all_passed &= approx_eq(evaluate(df_multi_x, coords), 151.0472506361786)

# Mixed partial derivative w.r.t. :x and :y
df_multi_xy = derivative(f_multi, Dict{Symbol,Int}(:x => 1, :y => 1))
all_passed &= approx_eq(evaluate(df_multi_xy, coords), 302.0945012723572)

# Integral over rectangular region
v_integral = integral(f_multi, Dict{Symbol,Tuple{Float64,Float64}}(:x => (1.0, 3.0), :y => (0.0, 1.5), :z => (2.0, 4.0)))
all_passed &= approx_eq(v_integral, 24.626122800158615)

# Hessian diagonal
hess = Hessian(f_multi, [:x, :y, :z])
hess_eval = evaluate(hess, coords)
all_passed &= approx_eq(hess_eval[1,1], 176.22179240887505)
all_passed &= approx_eq(hess_eval[2,2], 268.5284455754287)
all_passed &= approx_eq(hess_eval[3,3], -21.146615089065016)

# ============================================================
# D. Piecewise function
# ============================================================

piece1 = PE_Function(1.0, 0.0, 0.0, 1)  # f(x) = x
piece2 = PE_Function(3.0, 0.0, 2.0, 2)  # f(x) = 3*(x-2)^2
pw = Piecewise_Function([Sum_Of_Functions([piece1]), Sum_Of_Functions([piece2])], [-Inf, 2.0])

all_passed &= approx_eq(evaluate(pw, 1.0), 1.0)
all_passed &= approx_eq(evaluate(pw, 3.0), 3.0)
all_passed &= approx_eq(evaluate(pw, 5.0), 27.0)

dpw = derivative(pw)
all_passed &= approx_eq(evaluate(dpw, 1.0), 1.0)
all_passed &= approx_eq(evaluate(dpw, 3.0), 6.0)

all_passed &= approx_eq(integral(pw, 0.0, 5.0), 29.0)

# ============================================================
# E. Root finding
# ============================================================

# x^2 - 4 = 0, starting from x=3
f_root1 = PE_Function(1.0, 0.0, 0.0, 2) + PE_Function(-4.0)
root1 = uniroot(f_root1, Dict{Symbol,Float64}(:default => 3.0))
all_passed &= root1.convergence
all_passed &= approx_eq(root1.coordinates[:default], 2.0)

# x^3 - 8 = 0, starting from x=3
f_root2 = PE_Function(1.0, 0.0, 0.0, 3) + PE_Function(-8.0)
root2 = uniroot(f_root2, Dict{Symbol,Float64}(:default => 3.0))
all_passed &= root2.convergence
all_passed &= approx_eq(root2.coordinates[:default], 2.0000000000000226)

# ============================================================
# F. Interpolation
# ============================================================

# Linear interpolation
x_lin = [1.0, 3.0, 5.0, 7.0]
y_lin = [2.0, 6.0, 4.0, 8.0]
lin_interp = create_linear_interpolation(x_lin, y_lin)

all_passed &= approx_eq(evaluate(lin_interp, 1.0), 2.0)
all_passed &= approx_eq(evaluate(lin_interp, 2.0), 4.0)
all_passed &= approx_eq(evaluate(lin_interp, 4.0), 5.0)
all_passed &= approx_eq(evaluate(lin_interp, 6.0), 6.0)
all_passed &= approx_eq(integral(lin_interp, 1.0, 7.0), 30.0)

# Constant interpolation to right
x_const = [1.0, 3.0, 5.0]
y_const = [10.0, 20.0, 30.0]
const_interp_r = create_constant_interpolation_to_right(x_const, y_const)

all_passed &= approx_eq(evaluate(const_interp_r, 0.0), 10.0)
all_passed &= approx_eq(evaluate(const_interp_r, 2.0), 10.0)
all_passed &= approx_eq(evaluate(const_interp_r, 4.0), 20.0)
all_passed &= approx_eq(evaluate(const_interp_r, 6.0), 30.0)

# Constant interpolation to left
const_interp_l = create_constant_interpolation_to_left(x_const, y_const)

all_passed &= approx_eq(evaluate(const_interp_l, 0.0), 10.0)
all_passed &= approx_eq(evaluate(const_interp_l, 2.0), 20.0)
all_passed &= approx_eq(evaluate(const_interp_l, 4.0), 30.0)
all_passed &= approx_eq(evaluate(const_interp_l, 6.0), 30.0)

# ============================================================
# G. Schumaker quadratic spline
# ============================================================

x_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0]
schum = Schumaker(x_data, y_data)
spline = create_quadratic_spline(schum)

all_passed &= approx_eq(evaluate(spline, 0.0), 0.0)
all_passed &= approx_eq(evaluate(spline, 1.5), 2.3069154589305167)
all_passed &= approx_eq(evaluate(spline, 2.5), 6.2759585628953385)
all_passed &= approx_eq(evaluate(spline, 4.0), 16.0)
all_passed &= approx_eq(integral(spline, 0.0, 5.0), 41.797927194401176)

# Derivative of spline
dspline = derivative(spline)
all_passed &= approx_eq(evaluate(dspline, 0.5), 0.827254248593737)
all_passed &= approx_eq(evaluate(dspline, 2.5), 4.86939868865599)
all_passed &= approx_eq(evaluate(dspline, 4.5), 9.109619081049441)

# ============================================================
# H. Chebyshev approximation (univariate)
# ============================================================

cheb = create_chebyshev_approximation(sin, 10, 5,
    OrderedDict{Symbol,Tuple{Float64,Float64}}(:default => (0.0, Float64(pi))))

all_passed &= approx_eq(evaluate(cheb, 0.0), 0.0005900371153731987)
all_passed &= approx_eq(evaluate(cheb, Float64(pi)/6), 0.5001895495293156)
all_passed &= approx_eq(evaluate(cheb, Float64(pi)/2), 0.9993965536561874)
all_passed &= approx_eq(evaluate(cheb, Float64(pi)), 0.0005900371153731987)
all_passed &= approx_eq(evaluate(cheb, Float64(pi)/4), 0.7077068050946655)

# ============================================================
# I. Chebyshev approximation (multivariate)
# ============================================================

cheb2d = create_chebyshev_approximation(
    (x,y) -> sin(x)*cos(y), 8, 4,
    OrderedDict{Symbol,Tuple{Float64,Float64}}(:x => (0.0, Float64(pi)), :y => (0.0, Float64(pi)))
)

all_passed &= approx_eq(evaluate(cheb2d, Dict{Symbol,Float64}(:x => 1.0, :y => 0.5)), 0.7389498405316471)
all_passed &= approx_eq(evaluate(cheb2d, Dict{Symbol,Float64}(:x => Float64(pi)/2, :y => 0.0)), 0.9671073703831982)

# ============================================================
# J. OLS Regression (univariate, deterministic)
# ============================================================

x_ols = collect(range(0.0, 2.0, length=50))
y_ols = x_ols .^ 2
ols_model, ols_reg = create_ols_approximation(y_ols, x_ols, 2; intercept=true, base_x=0.0)

ols_rss = sum((ols_reg.rr.mu .- ols_reg.rr.y) .^ 2)
all_passed &= ols_rss < 1e-20

all_passed &= approx_eq(evaluate(ols_model, 2.5), 6.250000000000037)
all_passed &= approx_eq(evaluate(ols_model, 1.0), 1.000000000000006)

# ============================================================
# K. Saturated OLS (multivariate, deterministic)
# ============================================================

n_sat = 100
dd_sat = DataFrame()
dd_sat[!, :a] = collect(range(0.0, 3.0, length=n_sat))
dd_sat[!, :b] = collect(range(1.0, 4.0, length=n_sat))
dd_sat[!, :y] = 2.0 .* dd_sat[!, :a] .+ 3.0 .* dd_sat[!, :b] .+ 1.0
sat_model, sat_reg = create_saturated_ols_approximation(dd_sat, :y, [:a, :b], 1; intercept=true)

sat_rss = sum((sat_reg.rr.mu .- sat_reg.rr.y) .^ 2)
all_passed &= sat_rss < 1e-20

all_passed &= approx_eq(evaluate(sat_model, Dict{Symbol,Float64}(:a => 1.5, :b => 2.5)), 11.500000000000002)

# ============================================================
# L. Recursive Partitioning (deterministic, piecewise constant)
# ============================================================

n_rp = 200
x_rp = collect(range(-5.0, 5.0, length=n_rp))
y_rp = [xi < 0.0 ? 10.0 : 20.0 for xi in x_rp]
dd_rp = DataFrame()
dd_rp[!, :x] = x_rp
dd_rp[!, :y] = y_rp
rp_model, rp_reg = create_recursive_partitioning(dd_rp, :y, Set{Symbol}([:x]), 3; rel_tol = 1e-3)

rp_rss = sum((rp_reg.rr.mu .- rp_reg.rr.y) .^ 2)
all_passed &= rp_rss < 1e-20

all_passed &= approx_eq(evaluate(rp_model, Dict{Symbol,Float64}(:x => -3.0)), 10.0)
all_passed &= approx_eq(evaluate(rp_model, Dict{Symbol,Float64}(:x => 3.0)), 20.0)

# ============================================================
# M. MARS Splines (deterministic)
# ============================================================

n_mars = 200
x_mars = collect(range(-5.0, 5.0, length=n_mars))
y_mars = 3.0 .* max.(0.0, x_mars .- 1.0) .+ 7.0
dd_mars = DataFrame()
dd_mars[!, :x] = x_mars
dd_mars[!, :y] = y_mars
mars_model, mars_reg = create_mars_spline(dd_mars, :y, Set{Symbol}([:x]), 3; rel_tol = 1e-3)

mars_rss = sum((mars_reg.rr.mu .- mars_reg.rr.y) .^ 2)
all_passed &= mars_rss < 1e-15

all_passed &= approx_eq(evaluate(mars_model, Dict{Symbol,Float64}(:x => -2.0)), 6.9999999999998295)
all_passed &= approx_eq(evaluate(mars_model, Dict{Symbol,Float64}(:x => 3.0)), 13.000000000000256)

all_passed
