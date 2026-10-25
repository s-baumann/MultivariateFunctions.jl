function next_chebyshev(previous::Sum_Of_Functions, two_previous::Sum_Of_Functions, dim_name::Symbol)
    return PE_Function(2.0, UnitMap([dim_name => PE_Unit(0.0,0.0,1)])) * previous - two_previous
end

# First and second kind Chebyshevs share the same recurrence and differ only in
# the coefficient on the linear (second) polynomial.
function get_Chebyshevs(num::Int, dim_name::Symbol, first_kind::Bool)
    if num < 1
        error("Select a positive number of Chebyshevs to return")
    end
    chebyshevs = Array{Sum_Of_Functions}(undef, num)
    chebyshevs[1] = Sum_Of_Functions([PE_Function(1.0, UnitMap([dim_name => PE_Unit(0.0,0.0,0)]))])
    if num > 1
        linear_coefficient = first_kind ? 1.0 : 2.0
        chebyshevs[2] = Sum_Of_Functions([PE_Function(linear_coefficient, UnitMap([dim_name => PE_Unit(0.0,0.0,1)]))])
    end
    for n in 3:num
        chebyshevs[n] = next_chebyshev(chebyshevs[n-1], chebyshevs[n-2], dim_name)
    end
    return chebyshevs
end

"""
    get_chebyshevs_up_to(num::Int, first_kind::Bool = true; dim_name::Symbol = default_symbol)
    get_chevyshevs_up_to(num::Int, first_kind::Bool = true; dim_name::Symbol = default_symbol)
    Output all chebyshev polynomials up to degree num.
"""
function get_chebyshevs_up_to(num::Int, first_kind::Bool = true; dim_name::Symbol = default_symbol)
    return get_Chebyshevs(num, dim_name, first_kind)
end
const get_chevyshevs_up_to = get_chebyshevs_up_to
