using MultivariateFunctions
using Dates
using SchumakerSpline: Schumaker

tol = 10*eps()
const global_base_date = Date(2000,1,1)
StartDate = Date(2018, 7, 21)

x = StartDate .+ Dates.Day.(2 .* (1:1000 .- 1))

function ff_interp(x::Date)
    days_between = years_from_global_base(x)
    return log(days_between) + sqrt(days_between)
end
y = ff_interp.(x)

spline = create_linear_interpolation(x,y)
# Test if interpolating
all(abs.(evaluate.(Ref(spline), x) .- y) .< tol)

x_float = years_from_global_base.(x)
coefficient_in_first_interval = (y[2] - y[1])/(x_float[2] - x_float[1])
(coefficient_in_first_interval - spline.functions_[1].functions_[2].multiplier_) < tol
other_coefficient = (y[11] - y[10])/(x_float[11] - x_float[10])
(other_coefficient - spline.functions_[10].functions_[2].multiplier_) < tol

left_const = create_constant_interpolation_to_left(x,y)
right_const = create_constant_interpolation_to_right(x,y)
all(abs.(evaluate.(Ref(left_const), x_float .- tol) .- evaluate.(Ref(right_const), x_float)) .< tol)

# =====================================================
# Float64 input variants
# =====================================================

x_f = collect(range(1.0, 5.0; length=20))
y_f = sin.(x_f)

# Linear interpolation with Float64 input
lin_f = create_linear_interpolation(x_f, y_f)
# Must interpolate at knots
all(abs.(evaluate.(Ref(lin_f), x_f) .- y_f) .< tol)
# Must be between values at midpoints (monotone regions)
mid = (x_f[2] + x_f[3]) / 2
y_mid = evaluate(lin_f, mid)
(y_mid >= min(y_f[2], y_f[3]) - tol) && (y_mid <= max(y_f[2], y_f[3]) + tol)

# Constant interpolation to right with Float64 input
right_f = create_constant_interpolation_to_right(x_f, y_f)
# At a knot, should pick up the value from the left
abs(evaluate(right_f, x_f[5]) - y_f[5]) < tol
# Just after a knot, should still be that value
abs(evaluate(right_f, x_f[5] + 0.001) - y_f[5]) < tol

# Constant interpolation to left with Float64 input
left_f = create_constant_interpolation_to_left(x_f, y_f)
# Just before a knot should give the value AT that knot
abs(evaluate(left_f, x_f[5] - 0.001) - y_f[5]) < tol

# Derivative of constant interpolation is zero everywhere (between knots)
d_const = derivative(right_f)
abs(evaluate(d_const, x_f[3] + 0.01)) < tol

# Derivative of linear interpolation is constant in each interval
d_lin = derivative(lin_f)
expected_slope = (y_f[4] - y_f[3]) / (x_f[4] - x_f[3])
abs(evaluate(d_lin, (x_f[3] + x_f[4]) / 2) - expected_slope) < 1e-10

# Integral of constant interpolation
int_const = integral(right_f, x_f[1], x_f[3])
# Should equal sum of rectangle areas
expected_int = y_f[1] * (x_f[2] - x_f[1]) + y_f[2] * (x_f[3] - x_f[2])
abs(int_const - expected_int) < 1e-10

# =====================================================
# Int input for quadratic spline
# =====================================================

x_int = collect(1:20)
y_int = sqrt.(Float64.(x_int))
spline_int = create_quadratic_spline(x_int, y_int)
# Must interpolate at knots
all(abs.(evaluate.(Ref(spline_int), Float64.(x_int)) .- y_int) .< 1e-10)

# =====================================================
# Float64 input for quadratic spline
# =====================================================

spline_float = create_quadratic_spline(x_f, y_f)
all(abs.(evaluate.(Ref(spline_float), x_f) .- y_f) .< tol)

# Quadratic spline: third derivative should be zero within each interval
d3 = derivative(derivative(derivative(spline_float)))
abs(evaluate(d3, (x_f[5] + x_f[6]) / 2)) < tol

# =====================================================
# Schumaker object input for quadratic spline
# =====================================================

schum_obj = Schumaker(x_f, y_f)
spline_from_schum = create_quadratic_spline(schum_obj)
all(abs.(evaluate.(Ref(spline_from_schum), x_f) .- y_f) .< tol)

# =====================================================
# Custom dim_name for all interpolation methods
# =====================================================

lin_named = create_linear_interpolation(x_f, y_f; dim_name = :t)
underlying_dimensions(lin_named) == Set([:t])
abs(evaluate(lin_named, Dict(:t => x_f[3])) - y_f[3]) < tol

const_named = create_constant_interpolation_to_right(x_f, y_f; dim_name = :t)
underlying_dimensions(const_named) == Set([:t])

# =====================================================
# DatePeriod input variants
# =====================================================

x_period = Dates.Day.(1:10) .* 100
y_period = Float64.(1:10) .* 2.0

lin_period = create_linear_interpolation(x_period, y_period)
right_period = create_constant_interpolation_to_right(x_period, y_period)
left_period = create_constant_interpolation_to_left(x_period, y_period)
spline_period = create_quadratic_spline(x_period, y_period)

# All should be valid Piecewise_Functions
typeof(lin_period) == MultivariateFunctions.Piecewise_Function
typeof(right_period) == MultivariateFunctions.Piecewise_Function
typeof(left_period) == MultivariateFunctions.Piecewise_Function
typeof(spline_period) == MultivariateFunctions.Piecewise_Function

# Should interpolate at knots (convert periods to floats for evaluation)
x_period_float = period_length.(x_period)
all(abs.(evaluate.(Ref(lin_period), x_period_float) .- y_period) .< 1e-10)
