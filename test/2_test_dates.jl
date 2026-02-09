using MultivariateFunctions
using Dates
tol = 10*eps()

# =====================================================
# Direct tests of date_conversions.jl functions
# =====================================================

# years_between(Date, Date) — basic
d1 = Date(2000,1,1)
d2 = Date(2001,1,1)
yb = years_between(d2, d1)
abs(yb - Dates.days(d2 - d1) / 365.2422) < tol

# years_between antisymmetry: years_between(a,b) == -years_between(b,a)
abs(years_between(d1, d2) + years_between(d2, d1)) < tol

# years_between same date is zero
abs(years_between(d1, d1)) < tol

# years_between(Dates.Day, Dates.Day)
day1 = convert(Dates.Day, d1)
day2 = convert(Dates.Day, d2)
abs(years_between(day2, day1) - years_between(d2, d1)) < tol

# years_from_global_base(Date) — the global base is Date(2000,1,1) so it should be 0 there
abs(years_from_global_base(Date(2000,1,1))) < tol

# years_from_global_base(Date) — known value
abs(years_from_global_base(d2) - years_between(d2, Date(2000,1,1))) < tol

# years_from_global_base(Dates.Day)
abs(years_from_global_base(day2) - years_from_global_base(d2)) < tol

# period_length — a Year should be close to 1.0
pl = period_length(Dates.Year(1))
abs(pl - 365 / 365.2422) < 0.01  # Year(1) = 365 days, so slightly less than 1.0

# period_length — a Day should be 1/365.2422
abs(period_length(Dates.Day(1)) - 1.0 / 365.2422) < tol

# period_length with custom base
pl_custom = period_length(Dates.Month(1), Date(2020, 3, 1))
# March has 31 days
abs(pl_custom - 31 / 365.2422) < tol

# period_length — zero period
abs(period_length(Dates.Day(0))) < tol

# =====================================================
# Original PE_Function date tests
# =====================================================

today = Date(2000,1,1)
pe_func = PE_Function(1.0,2.0,today, 3)
(pe_func.units_[:default].base_ - years_from_global_base(today))   < tol
date_in_2020 = Date(2020,1,1)
pe_func2 = PE_Function(1.0,2.0,date_in_2020, 3)
(pe_func2.units_[:default].base_ - years_from_global_base(date_in_2020))   < tol
abs(evaluate(pe_func, date_in_2020) - evaluate(pe_func, years_from_global_base(date_in_2020)) ) < tol

#Sum of functions
sum_func = Sum_Of_Functions([pe_func, PE_Function(2.0,2.5,today, 3) ])
abs(evaluate(sum_func, date_in_2020) - evaluate(sum_func, years_from_global_base(date_in_2020)) ) < tol

inyear = Date(2001,1,1)
result = integral(pe_func,today,inyear)

left_integral = integral(pe_func, Dict{Symbol,Tuple{Any,Any}}(:default => (today, :default_right)))
abs(evaluate(left_integral, Dict{Symbol,Any}(:default_right => inyear)) - result) < 100 * tol
right_integral = integral(pe_func, Dict{Symbol,Tuple{Any,Any}}(:default => (:default_left, inyear)))
abs(evaluate(right_integral, Dict{Symbol,Any}(:default_left => today)) - result) <  100 * tol
both_integral = integral(pe_func, Dict{Symbol,Tuple{Any,Any}}(:default => (:default_left, :default_right)))
abs(evaluate(both_integral, Dict{Symbol,Any}([:default_left, :default_right] .=> [today, inyear])) - result) <  100 * tol
