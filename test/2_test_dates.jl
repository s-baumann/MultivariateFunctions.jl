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
@test abs(yb - Dates.days(d2 - d1) / 365.2422) < tol

# years_between antisymmetry: years_between(a,b) == -years_between(b,a)
@test abs(years_between(d1, d2) + years_between(d2, d1)) < tol

# years_between same date is zero
@test abs(years_between(d1, d1)) < tol

# years_between(Dates.Day, Dates.Day)
day1 = convert(Dates.Day, d1)
day2 = convert(Dates.Day, d2)
@test abs(years_between(day2, day1) - years_between(d2, d1)) < tol

# years_from_global_base(Date) — the global base is Date(2000,1,1) so it should be 0 there
@test abs(years_from_global_base(Date(2000,1,1))) < tol

# years_from_global_base(Date) — known value
@test abs(years_from_global_base(d2) - years_between(d2, Date(2000,1,1))) < tol

# years_from_global_base(Dates.Day)
@test abs(years_from_global_base(day2) - years_from_global_base(d2)) < tol

# period_length — a Year should be close to 1.0
pl = period_length(Dates.Year(1))
@test abs(pl - 365 / 365.2422) < 0.01  # Year(1) = 365 days, so slightly less than 1.0

# period_length — a Day should be 1/365.2422
@test abs(period_length(Dates.Day(1)) - 1.0 / 365.2422) < tol

# period_length with custom base
pl_custom = period_length(Dates.Month(1), Date(2020, 3, 1))
# March has 31 days
@test abs(pl_custom - 31 / 365.2422) < tol

# period_length — zero period
@test abs(period_length(Dates.Day(0))) < tol

# =====================================================
# Original PE_Function date tests
# =====================================================

today = Date(2000,1,1)
pe_func = PE_Function(1.0,2.0,today, 3)
@test (unit_get(pe_func.units_, :default).base_ - years_from_global_base(today))   < tol
date_in_2020 = Date(2020,1,1)
pe_func2 = PE_Function(1.0,2.0,date_in_2020, 3)
@test (unit_get(pe_func2.units_, :default).base_ - years_from_global_base(date_in_2020))   < tol
@test abs(evaluate(pe_func, date_in_2020) - evaluate(pe_func, years_from_global_base(date_in_2020)) ) < tol

#Sum of functions
sum_func = Sum_Of_Functions([pe_func, PE_Function(2.0,2.5,today, 3) ])
@test abs(evaluate(sum_func, date_in_2020) - evaluate(sum_func, years_from_global_base(date_in_2020)) ) < tol

inyear = Date(2001,1,1)
result = integral(pe_func,today,inyear)

left_integral = integral(pe_func, Dict{Symbol,Tuple{Any,Any}}(:default => (today, :default_right)))
@test abs(evaluate(left_integral, Dict{Symbol,Any}(:default_right => inyear)) - result) < 100 * tol
right_integral = integral(pe_func, Dict{Symbol,Tuple{Any,Any}}(:default => (:default_left, inyear)))
@test abs(evaluate(right_integral, Dict{Symbol,Any}(:default_left => today)) - result) <  100 * tol
both_integral = integral(pe_func, Dict{Symbol,Tuple{Any,Any}}(:default => (:default_left, :default_right)))
@test abs(evaluate(both_integral, Dict{Symbol,Any}([:default_left, :default_right] .=> [today, inyear])) - result) <  100 * tol
