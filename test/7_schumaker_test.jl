using MultivariateFunctions
using Dates

tol = 10*eps()
const global_base_date = Date(2000,1,1)
StartDate = Date(2018, 7, 21)


x = Array{Date}(undef, 100)
for i in 1:100
    x[i] = StartDate +Dates.Day(2* (i-1))
end

function ff(x::Date)
    days_between = years_from_global_base(x)
    return log(days_between) + sqrt(days_between)
end
y = ff.(x)

splinex = create_quadratic_spline(x,y; dim_name = :x)
collect(keys(splinex.thresholds_))[1] == :x

spline = create_quadratic_spline(x,y)
# Test if interpolating
all(abs.(evaluate.(Ref(spline), x) .- y) .< tol)

# Testing third derivatives
third_derivatives = evaluate.(Ref(derivative(derivative(derivative(spline)))), x)
maximum(third_derivatives) < tol

# Testing Integrals
function analytic_integral(lhs,rhs)
    lhs_in_days = years_from_global_base(lhs)
    rhs_in_days = years_from_global_base(rhs)
    return rhs_in_days*log(rhs_in_days) - rhs_in_days + (2/3)*rhs_in_days^(3/2) - (lhs_in_days*log(lhs_in_days) - lhs_in_days)- (2/3)*lhs_in_days^(3/2)
end

lhs = StartDate
rhs = StartDate + Dates.Month(16)
numerical_integral  = integral(spline, lhs,rhs)

analytical = analytic_integral(lhs,rhs)
abs(  analytical - numerical_integral  ) < 0.001
