module MultivariateFunctions

using Dates
using SchumakerSpline: Schumaker
using GLM
using DataFrames
using DataStructures: OrderedDict
using Combinatorics: permutations

# This includes timing functions for use of UnivariateFunctions with dates.
include("date_conversions.jl")
export years_between, years_from_global_base
# The abstrat type UnivariateFunction and all structs are implemented here.
# In addition we have operator reversals and some supporting functions.
include("0_structs_and_generic_reversals.jl")
export UnivariateFunction, PE_Function, Sum_Of_Functions, Piecewise_Function, PE_Unit
export change_base, trim_piecewise_function, sort, convert_to_linearly_rescale_inputs
export evaluate, rebadge
include("1_algebra.jl")
include("2_calculus.jl")
export derivative, integral
include("3_1D_splines_and_interpolation.jl")
export create_quadratic_spline, create_constant_interpolation_to_right
export create_constant_interpolation_to_left, create_linear_interpolation
include("chebyshevs.jl")
export get_chevyshevs_up_to, get_chebyshev
include("4_chebyshev_approximation.jl")
export create_chebyshev_approximation
include("5_ols_regression.jl")
export create_ols_approximation, create_saturated_ols_approximation

export +, -, *, /, ^, convert, ≂
end

# In terms of UnivariateFunction types we have the following seniority:
# PE_Function
# Sum_Of_Functions
# Piecewise_Function

# In particular a Sum_Of_Functions contains the first two types of functions but never
# another Sum_Of_Functions (although one can be used to construct a Sum_Of_Functions). If a sum
# of functions contains an undefined function then it contains nothing else.
# A Piecewise_Function can have pieces composed of the first three categories but not another
# Piecewise_Function

# In terms of multiple dispatch rules this means that we:
# * define operators of each type of UnivariateFunction with another of its own type.
# * For each we define operators (f + x) with scalars. The reverse operations (x + f) can be done
#    all at once except for powers, subtraction and division where it matters.
#  All operations between these types should be defined for every pair. The seniority rule means that
#  Where a * b (where a and b are types) is the same as b * a then we define b*a = *(b,a) and put the
# relevent code in b * a place. Undefined_Function should take prevedence over PE_Function and etc
# in the seniority list.
''
