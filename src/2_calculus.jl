function derivative_(u::PE_Unit)# Intentially changing name so this is not exported.
    if u.d_ == 0
        result_array = Array{Tuple{Float64,PE_Unit}}(undef,1)
        result_array[1] = (u.b_, PE_Unit(u.b_, u.base_, u.d_))
    else
        result_array = Array{Tuple{Float64,PE_Unit}}(undef,2)
        result_array[1] = (u.b_, PE_Unit(u.b_, u.base_, u.d_))
        result_array[2] = (u.d_, PE_Unit(u.b_, u.base_, u.d_ - 1))
    end
    return result_array
end
function derivative_(u::Array{Tuple{Float64,PE_Unit},1})# Intentially changing name so this is not exported.
    len = length(u)
    heap = Array{Tuple{Float64,PE_Unit},1}()
    for i in 1:len
        u_result = u[i]
        mult = u_result[1]
        output = derivative_(u_result[2])
        output_mults, output_units =  collect(zip(output...))
        output_mults = mult .* vcat(output_mults...)
        output_units = vcat(output_units...)
        zipped_output = collect(zip(output_mults,output_units))
        append!(heap, zipped_output)
    end
    return heap
end

function derivative(f::PE_Function, derivs::Dict{Symbol,Int})
    # Should always return a Sum_Of_Functions or a PE_Function.
    dims = keys(derivs)
    fdims = keys(f.units_)

    if (length(setdiff(dims,fdims)) > 0) && minimum(get.(Ref(derivs), [setdiff(dims,fdims)...],0)) < 1
        return PE_Function()
    end
    dims_dict = Dict{Symbol,Array{Tuple{Float64,PE_Unit},1}}()
    units = deepcopy(f.units_)
    for dimen in dims
        num_derivs = derivs[dimen]
        if num_derivs > 0
            if dimen in keys(units)
                unit = pop!(units, dimen)
                der = derivative_(unit)
                for i in 2:num_derivs
                    der = derivative_(der)
                end
                dims_dict[dimen] = der
            else
                dims_dict[dimen] = Array{Tuple{Float64,PE_Unit},1}([(0.0, PE_Unit())])
            end
        end
    end
    array_of_tups = [Dict(dims .=> val) for val in (collect(Iterators.product(getindex.((dims_dict,),dims)...))...,)]
    array_of_pes = PE_Function.(1.0, array_of_tups)
    remaining_dims = PE_Function(f.multiplier_, units)
    final_result = remaining_dims .* array_of_pes
    if length(final_result) == 0
        return PE_Function()
    elseif length(final_result) == 1
        return final_result[1]
    else
        return Sum_Of_Functions(final_result)
    end
end
function derivative(f::Sum_Of_Functions, derivs::Dict{Symbol,Int})
    # Should always return a Sum_Of_Functions or a PE_Function.
    deriv_funcs = derivative.(f.functions_, Ref(derivs))
    return Sum_Of_Functions(deriv_funcs)
end

function derivative(f::Piecewise_Function, derivs::Dict{Symbol,Int})
    max_order = maximum(values(derivs))
    if max_order > 0
        derivatives = derivative.(f.functions_, Ref(derivs))
        return Piecewise_Function(derivatives, f.thresholds_)
    elseif max_order == 0
        return f
    else
        error("Not possible to take a negative derivative. Try evaluating an integral instead.")
    end
end

function derivative(f::Missing, derivs::Dict{Symbol,Int})
    return Missing()
end

function derivative(f::MultivariateFunction)
    if length(setdiff(underlying_dimensions(f), Set([default_symbol]))) == 0
        derivs = Dict{Symbol,Int}(default_symbol => 1)
        return derivative(f, derivs)
    else
        error("It is not possible to take the derivative without using a dict unless the only variable is the default one.")
    end
end


## Integration
function indefinite_integral(u::PE_Unit, incoming_multiplier::Float64 = 1.0)# Intentially changing name so this is not exported.
    result_array = Array{Tuple{Float64,PE_Unit}}(undef,1)
    if u.b_ ≂ 0.0
        result_array[1] = (incoming_multiplier/(u.d_+1), PE_Unit(u.b_, u.base_, u.d_+1)) # Note (u.d_+1) > 0 because u.d_ \geq 0
        return result_array
    else
        result_array[1] = (incoming_multiplier/u.b_, PE_Unit(u.b_, u.base_, u.d_))
        if u.d_ > 0
            other_pieces_multiplier = -incoming_multiplier * (u.d_ / u.b_)
            other_pieces_unit = PE_Unit(u.b_, u.base_, u.d_ - 1)
            other_pieces = indefinite_integral(other_pieces_unit, other_pieces_multiplier)
            append!(result_array, other_pieces)
        end
        return result_array
    end
end

function apply_limits(mult::Float64, indef::Array{Tuple{Float64,PE_Unit}}, left::Symbol, right::Symbol)# Intentially changing name so this is not exported.
    converted = (collect(Iterators.product(((indef,))...))...,)
    rights = [Dict{Symbol,Tuple{Float64,PE_Unit}}([right] .=> val) for val in converted]
    lefts  = [Dict{Symbol,Tuple{Float64,PE_Unit}}([left] .=> val) for val in converted]
    return Sum_Of_Functions(PE_Function.(mult, rights)) - Sum_Of_Functions(PE_Function.(mult, lefts))
end
function apply_limits(mult::Float64, indef::Array{Tuple{Float64,PE_Unit}}, left::Float64, right::Symbol)# Intentially changing name so this is not exported.
    converted = (collect(Iterators.product(((indef,))...))...,)
    rights = [Dict{Symbol,Tuple{Float64,PE_Unit}}([right] .=> val) for val in converted]
    lefts  = [Dict{Symbol,Tuple{Float64,PE_Unit}}([:left] .=> val) for val in converted]
    return Sum_Of_Functions(PE_Function.(mult, rights)) - sum(evaluate.(PE_Function.(mult, lefts), Ref(Dict{Symbol,Float64}(:left => left))))
end
function apply_limits(mult::Float64, indef::Array{Tuple{Float64,PE_Unit}}, left::Symbol, right::Float64)# Intentially changing name so this is not exported.
    converted = (collect(Iterators.product(((indef,))...))...,)
    rights = [Dict{Symbol,Tuple{Float64,PE_Unit}}([:right] .=> val) for val in converted]
    lefts  = [Dict{Symbol,Tuple{Float64,PE_Unit}}([left] .=> val) for val in converted]
    return evaluate.(Sum_Of_Functions(PE_Function.(mult, rights)), Ref(Dict{Symbol,Float64}(:right => right))   ) - Sum_Of_Functions(PE_Function.(mult, lefts))
end
function apply_limits(mult::Float64, indef::Array{Tuple{Float64,PE_Unit}}, left::Float64, right::Float64)# Intentially changing name so this is not exported.
    converted = (collect(Iterators.product(((indef,))...))...,)
    converted_again = [Dict{Symbol,Tuple{Float64,PE_Unit}}([default_symbol] .=> val) for val in converted]
    funcs = Sum_Of_Functions(PE_Function.(mult, converted_again))
    return evaluate.(funcs, right) - evaluate.(funcs, left)
end

const IntegrationLimitDict = Union{Dict{Symbol,Tuple{Union{Symbol,Float64},Union{Symbol,Float64}}},Dict{Symbol,Tuple{Symbol,Symbol}},Dict{Symbol,Tuple{Float64,Float64}},Dict{Symbol,Tuple{Symbol,Float64}},Dict{Symbol,Tuple{Float64,Symbol}}}

function integral(f::PE_Function, limits::IntegrationLimitDict)
    if length(f.units_) == 0
        volume_of_cube = 1.0
        for dimen in keys(limits)
            volume_of_cube = volume_of_cube * (limits[dimen][2] - limits[dimen][1])
        end
        return volume_of_cube * f.multiplier_
    end
    units = deepcopy(f.units_)
    result_by_dimension = Array{Union{Float64,Sum_Of_Functions,PE_Function},1}()
    for dimen in keys(limits)
        left_  = limits[dimen][1]
        right_ = limits[dimen][2]
        f_unit = Array{PE_Unit,1}(undef,1)
        if haskey(f.units_, dimen)
            f_unit = pop!(units, dimen)
        else
            f_unit = PE_Unit()
        end
        indef  = indefinite_integral(f_unit)
        result_of_applying_limits = apply_limits(1.0, indef, left_, right_)
        append!(result_by_dimension, [result_of_applying_limits])
    end
    if length(units) == 0
        return f.multiplier_ * prod(result_by_dimension)
    else
        return PE_Function(f.multiplier_, units) * prod(result_by_dimension)
    end
end

function integral(f::Sum_Of_Functions, limits::IntegrationLimitDict)
    funcs = integral.(f.functions_, Ref(limits))
    return sum(funcs)
end

function hypercubes_to_integrate(f::Piecewise_Function, limits::Dict{Symbol,Tuple{Float64,Float64}})
    ks= sort(collect(keys(limits)))
    new_dict = Dict{Symbol,Array{Float64}}()
    for dimen in ks
        lower = limits[dimen][1]
        upper = limits[dimen][2]
        if lower >= upper
            error(string("The lower limit of integration is higher than the upper for dimension ", dim))
        end
        thresholds = f.thresholds_[dimen]
        censored_thresholds = vcat(lower, thresholds[thresholds .> lower])
        censored_thresholds2 = vcat(censored_thresholds[censored_thresholds .< upper] , upper)
        new_dict[dimen] = censored_thresholds2
    end
    lengths_minus_one = length.(get.(Ref(new_dict),ks,0)) .- 1
    index_combinations = vcat.(collect(collect(Iterators.product(range.(1,lengths_minus_one)...))))
    indices = (collect(collect(Iterators.product(range.(1,lengths_minus_one)...)))...,)
    indices = collect(collect.(indices))
    total_len = length(indices)
    new_cubes = Array{Dict{Symbol,Tuple{Float64,Float64}},1}(undef,total_len)
    for i in range(1, total_len)
        new_cube = Dict{Symbol,Tuple{Float64,Float64}}()
        ind = indices[i]
        for j in 1:length(ks)
            key = ks[j]
            indd = ind[j]
            new_cube[key]  =  (new_dict[key][indd], new_dict[key][indd+1])
        end
        new_cubes[i] = new_cube
    end
    return new_cubes
end

function integral(f::Piecewise_Function, limits::Dict{Symbol,Tuple{Float64,Float64}})
    cubes_to_integrate = hypercubes_to_integrate(f, limits)
    ints = Array{Float64,1}(undef,length(cubes_to_integrate))
    for i in 1:length(ints)
        piece_func = get_correct_function_from_piecewise(f, cubes_to_integrate[i])
        ints[i] = integral(piece_func, cubes_to_integrate[i])
    end
    return sum(ints)
end








function integral(f::MultivariateFunction, left_limit::Float64, right_limit::Float64)
    if underlying_dimensions(f) == Set([default_symbol])
        limits_ = Dict{Symbol,Tuple{Float64,Float64}}(default_symbol => Tuple{Float64,Float64}((left_limit,right_limit)))
        return integral(f, limits_)
    else
        error("Cannot evaluate the integral of a Multivariate function without a dictionary set of coordinates unless it is a MultivariateFunction with only the default dimension being used.")
    end
end

function integral(f::MultivariateFunction, left_limit::Date, right_limit::Date)
    if underlying_dimensions(f) == Set([default_symbol])
        limits_ = Dict{Symbol,Tuple{Float64,Float64}}(default_symbol => Tuple{Float64,Float64}((years_from_global_base(left_limit),years_from_global_base(right_limit))))
        return integral(f, limits_)
    else
        error("Cannot evaluate the integral of a Multivariate function without a dictionary set of coordinates unless it is a MultivariateFunction with only the default dimension being used.")
    end
end











"""
function hypercubes_to_integrate(f::Piecewise_Function, limits::Dict{Symbol,Tuple{Float64,Float64}})
    ks= sort(collect(keys(limits)))
    new_dict = Dict{Symbol,Array{Float64}}()
    for dimen in ks
        lower = limits[dimen][1]
        upper = limits[dimen][2]
        if lower >= upper
            error(string("The lower limit of integration is higher than the upper for dimension ", dim))
        end
        thresholds = f.thresholds_[dimen]
        censored_thresholds = vcat(lower, thresholds[thresholds .> lower + 10eps()])
        censored_thresholds2 = vcat(censored_thresholds[censored_thresholds .< upper] , upper)
        new_dict[dimen] = censored_thresholds2
    end
    lengths_minus_one = length.(get.(Ref(new_dict),ks,0)) .- 1
    index_combinations = vcat.(collect(collect(Iterators.product(range.(1,lengths_minus_one)...))))
    indices = (collect(collect(Iterators.product(range.(1,lengths_minus_one)...)))...,)
    indices = collect(collect.(indices))
    total_len = length(indices)
    new_lows = Array{Dict{Symbol,Float64},1}(undef,total_len)
    new_highs = Array{Dict{Symbol,Float64},1}(undef,total_len)
    for i in range(1, total_len)
        new_low = Dict{Symbol,Float64}()
        new_high = Dict{Symbol,Float64}()
        ind = indices[i]
        for j in 1:length(ks)
            key = ks[j]
            in = ind[j]
            new_low[key]  =  new_dict[key][in]
            new_high[key] =  new_dict[key][in+1]
        end
        new_lows[i] = new_low
        new_highs[i] = new_high
    end
    return new_lows, new_highs
end











function evaluate_integral_(u::PE_Unit, left_point::Float64, right_point::Float64)# Intentially changing name so this is not exported.
    indef = indefinite_integral(u,1.0)
    indef_ = collect(zip(indef...))
    mults = vcat(indef_[1]...)
    units = vcat(indef_[2]...)
    val = sum(mults .* (evaluate.(units, right_point) .- evaluate.(units, left_point)))
    return val
end

function right_integral_(u::PE_Unit, left_point::Float64)# Intentially changing name so this is not exported.
    # Here The left side is fixed and right hand side is free.
    indef = indefinite_integral(u,1.0)
    indef_ = collect(zip(indef...))
    mults = vcat(indef_[1]...)
    units = vcat(indef_[2]...)
    leftval = sum(mults .* evaluate.(units, left_point))
    append!(indef, [(-leftval, PE_Unit())])
    return(indef)
end

function left_integral_(u::PE_Unit, right_point::Float64)# Intentially changing name so this is not exported.
    # Here The right side is fixed and left hand side is free.
    indef = right_integral_(u,right_point)
    indef[:][1] = -1.0 * indef[:][1]
    return(indef)
end



function right_integral(f::PE_Function, left_coordinates::Dict{Symbol,Float64})
    dims_dict = Dict{Symbol,Array{Tuple{Float64,PE_Unit},1}}()
    units = deepcopy(f.units_)
    dims = keys(left_coordinates)
    for dim in dims
        if haskey(f.units_, dim)
            left_point = left_coordinates[dim]
            f_unit = pop!(units, dim)
            changed_pes = right_integral_(f_unit, left_point)
            dims_dict[dim] = changed_pes
        else
            dims_dict[dim] = [(0.0,PE_Unit())]
        end
    end
    array_of_tups = [Dict{Symbol,Tuple{Float64,PE_Unit}}(dims .=> val) for val in (collect(Iterators.product(getindex.((dims_dict,),dims)...))...,)]
    array_of_pes = PE_Function(f.multiplier_, units) .* PE_Function.(1.0, array_of_tups)
    return Sum_Of_Functions(array_of_pes)
end

function left_integral(f::PE_Function, right_coordinates::Dict{Symbol,Float64})
    # Here The right side is fixed and left hand side is free.
    indef = right_integral(f,right_coordinates)
    return evaluate(indef, right_coordinates) - indef
end

function evaluate_integral(f::PE_Function, left_coordinates::Dict{Symbol,Float64}, right_coordinates::Dict{Symbol,Float64})
    mult = f.multiplier_
    units = deepcopy(f.units_)
    dims = keys(left_coordinates)
    for dim in dims
        if haskey(f.units_, dim)
            left_point = left_coordinates[dim]
            right_point= right_coordinates[dim]
            f_unit = pop!(units, dim)
            mult = mult * evaluate_integral_(f_unit, left_point, right_point)
        else
            mult = mult * 0.0
        end
    end
    if length(units) == 0
        return mult
    else
        return mult * PE_Function(1.0, units)
    end
end

function indefinite_integral(f::PE_Function, dims::Set{Symbol})
    units = deepcopy(f.units_)
    for dim in dims
        if haskey(f.units_, dim)
            f_unit = pop!(units, dim)
            mult = mult * indefinite_integral_(f_unit)
        else
            mult = mult * 0.0
        end
    end
    if length(units) == 0
        return mult
    else
        return mult * PE_Function(1.0, units)
    end
end

#  1 dimension case

function right_integral(f::PE_Function, left_coordinate::Float64)
    if length(setdiff(underlying_dimensions(f), Set([default_symbol]))) == 0
        coordinates = Dict{Symbol,Float64}(default_symbol => left_coordinate)
        return right_integral(f, coordinates)
    else
        error("It is not possible to use the right_integral method without using a dict unless the only variable is the default one.")
    end
end

function left_integral(f::PE_Function, right_coordinate::Float64)
    if length(setdiff(underlying_dimensions(f), Set([default_symbol]))) == 0
        coordinates = Dict{Symbol,Float64}(default_symbol => right_coordinate)
        return left_integral(f, coordinates)
    else
        error("It is not possible to use the left_integral method without using a dict unless the only variable is the default one.")
    end
end

function evaluate_integral(f::PE_Function, left_coordinate::Float64, right_coordinate::Float64)
    if length(setdiff(underlying_dimensions(f), Set([default_symbol]))) == 0
        left_coordinates = Dict{Symbol,Float64}(default_symbol => left_coordinate)
        right_coordinates = Dict{Symbol,Float64}(default_symbol => right_coordinate)
        return evaluate_integral(f, left_coordinates, right_coordinates)
    else
        error("It is not possible to use the left_integral method without using a dict unless the only variable is the default one.")
    end
end

## Date

function evaluate_integral(f::MultivariateFunction, left_coordinate::Date, right_coordinate::Date)
    left_as_float  = years_from_global_base(left_coordinate)
    right_as_float = years_from_global_base(right_coordinate)
    return evaluate_integral(f, left_as_float, right_as_float)
end
function left_integral(f::MultivariateFunction, right_coordinate::Date)
    right_as_float = years_from_global_base(right_coordinate)
    return left_integral(f, right_as_float)
end
function right_integral(f::MultivariateFunction, left_coordinate::Date)
    left_as_float  = years_from_global_base(left_coordinate)
    return right_integral(f, left_as_float)
end


function left_integral(f::MultivariateFunction, right_coordinates::Dict{Symbol,Any})
    new_coordinates = convert_to_float_dict(coordinates)
    return left_integral(f, new_coordinates)
end
function right_integral(f::MultivariateFunction, left_coordinates::Dict{Symbol,Any})
    new_coordinates = convert_to_float_dict(coordinates)
    return right_integral(f, new_coordinates)
end
"""
