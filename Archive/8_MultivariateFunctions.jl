import Base.+, Base.-, Base./, Base.*, Base.^
import Base.sort, Base.convert

abstract type MultivariateFunction end

function is_constant_function(func::PE_Function)
   return (abs(func.b_) < tol) & (func.d_ == 0)
end

"""
    Multivariate_PE_Function(multiplier::Float64, funcs::Dict{Symbol,PE_Function})

Creates a Multivariate_PE_Function from a multiplier and a set of PE_Functions.
"""
struct Multivariate_PE_Function <: MultivariateFunction
    multiplier_::Float64
    functions_::Dict{Symbol,PE_Function}
    function Multivariate_PE_Function(multiplier::Float64, funcs::Dict{Symbol,PE_Function})
        multiplier_ = multiplier
        functions_ = Dict{Symbol,PE_Function}()
        for dim in keys(funcs)
            multiplier_ = multiplier_ * funcs[dim].a_
            f_new = PE_Function(1.0, funcs[dim].b_, funcs[dim].base_, funcs[dim].d_)
            if !is_constant_function(f_new)
                functions_[dim] = f_new
            end
        end
        if abs(multiplier_) < tol
            return new(0.0,Dict{Symbol,PE_Function}())
        end
        return new(multiplier_, functions_)
    end
end
Base.broadcastable(e::Multivariate_PE_Function) = Ref(e)

function is_constant_function(func::Multivariate_PE_Function)
   return length(func.functions_) == 0
end

function is_zero_function(func::Multivariate_PE_Function)
    return (abs(func.multiplier_) < tol) & is_constant_function(func)
end

function dict_to_array(d::Dict{Symbol,Array{PE_Function}})
    ks= sort(collect(keys(d)));
    return [Dict(ks .=> val) for val in (collect(Iterators.product(getindex.((d,),ks)...))...,)]
end

function sort(funcs::Array{Multivariate_PE_Function,1})
    len = length(funcs)
    lengths = map(f -> length(f.functions_), funcs)
    ordering = convert(Array{Int}, sortslices(hcat(lengths, 1:len) , dims = 1)[:,2])
    return funcs[ordering]
end

"""
    Multivariate_Sum_Of_Functions(funcs)

Creates a Multivariate_Sum_Of_Functions from an array composed of Multivariate_PE_Function and other Multivariate_Sum_Of_Functions structs.
"""
struct Multivariate_Sum_Of_Functions <: MultivariateFunction
    functions_::Array{Multivariate_PE_Function,1}
    function Multivariate_Sum_Of_Functions(funcs::Array{Multivariate_PE_Function,1})
        cleaned_funcs = funcs[is_zero_function.(funcs) .!= true]
        sorted_funcs = sort(cleaned_funcs)
        consts = is_constant_function.(sorted_funcs)
        if sum(consts) == 0
            return new(sorted_funcs)
        else
            new_const = sum(map(f -> f.multiplier_, sorted_funcs[findall(consts)]))
            new_const_func = Multivariate_PE_Function(new_const,Dict{Symbol,PE_Function}())
            if sum(consts) == length(sorted_funcs)
                return new([new_const_func])
            else
                nonconsts = convert(BitArray, 1 .- consts)
                return new(vcat(new_const_func, sorted_funcs[findall(nonconsts)]))
            end
        end
    end
    function Multivariate_Sum_Of_Functions(funcs)
        Sums_Functions  = funcs[typeof.(funcs) .== UnivariateFunctions.Multivariate_Sum_Of_Functions]
        PE_Functions    = funcs[typeof.(funcs) .== UnivariateFunctions.Multivariate_PE_Function]
        if length(Sums_Functions) + length(PE_Functions) < length(funcs)
            error("Multivariate_Sum_Of_Functions can only be created from other Multivariate_Sum_Of_Functions and Multivariate_PE_Functions")
        end
        PE_Functions = convert(Array{Multivariate_PE_Function,1}, PE_Functions)
        for sum in Sums_Functions
            PE_Functions = vcat(PE_Functions, sum.functions_)
        end
        return Multivariate_Sum_Of_Functions(PE_Functions)
    end
    function Multivariate_Sum_Of_Functions(f::Multivariate_PE_Function)
        return Multivariate_Sum_Of_Functions([f])
    end
end
Base.broadcastable(e::Multivariate_Sum_Of_Functions) = Ref(e)

function dim_names(f::Multivariate_Sum_Of_Functions)
    dims = Set()
    for func in f.functions_
        dims = union(dims, keys(f.functions_))
    end
    return dims
end

function convert(::Type{Multivariate_Sum_Of_Functions}, f::Multivariate_PE_Function)
    return Multivariate_Sum_Of_Functions(f)
end
function convert(::Type{Float64}, f::Multivariate_PE_Function)
    if length(f.functions_) == 0
        return f.multiplier_
    else
        error(string("This multivariate function is a function of ", keys(f.functions_)..., " and hence cannot be converted to a scalar. To evaluate the function at certain values use the evaluate function."))
    end
end
function convert(::Type{Float64}, f::Multivariate_Sum_Of_Functions)
    val = 0.0
    for ff in f.functions_
        val = val + convert(Float64, ff)
    end
    return val
end


function +(f::Multivariate_Sum_Of_Functions, number::Float64)
    return Multivariate_Sum_Of_Functions(vcat(f.functions_, Multivariate_PE_Function(number, Dict{Symbol,PE_Function}())))
end
function +(f::Multivariate_PE_Function, number::Float64)
    return Multivariate_Sum_Of_Functions(vcat(f, Multivariate_PE_Function(number, Dict{Symbol,PE_Function}())))
end
function +(f::MultivariateFunction, number::Int)
    number_as_float = convert(Float64,number)
    return +(f,number_as_float)
end
function -(f::MultivariateFunction, number::Union{Float64,Int})
    number_as_float = convert(Float64,number)
    return +(f,-number_as_float)
end
function *(f::Multivariate_PE_Function, number::Union{Float64,Int})
    new_mult = f.multiplier_ * number
    return Multivariate_PE_Function(new_mult,f.functions_)
end
function *(f::Multivariate_Sum_Of_Functions, number::Union{Float64,Int})
    new_funcs = number * f.functions_
    return Multivariate_Sum_Of_Functions(new_funcs)
end
function /(f::MultivariateFunction, number::Union{Float64,Int})
    return f * (1/number)
end
function ^(f::MultivariateFunction, num::Int)
    if num < 0
        error("Cannot raise any MultivariateFunction function to a negative power")
    elseif num == 0
        return Multivariate_PE_Function(1.0, Dict{Symbol,PE_Function}())
    elseif num == 1
        return f
    elseif num == 2
        return f * f
    else
        product = f * f
        for i in 1:(num-2)
            product = product * f
        end
        return product
    end
end
function +(number::Union{Float64,Int}, f::MultivariateFunction)
    return +(f,number)
end
function -(number::Union{Float64,Int}, f::MultivariateFunction)
    return +(number, -1*f)
end
function *(number::Union{Float64,Int}, f::MultivariateFunction)
    return *(f,number)
end
function /(number::Union{Float64,Int}, f::MultivariateFunction)
    error("It is not possible yet to divide scalars by MultivariateFunction")
end
function ^(number::Union{Float64,Int}, f::MultivariateFunction)
    error("It is not possible yet to raise to the power of a MultivariateFunction")
end


function *(f1::Multivariate_PE_Function, f2::Multivariate_PE_Function)
    new_mult = f1.multiplier_ * f2.multiplier_
    keyset   = union(keys(f1.functions_), keys(f2.functions_))
    funcs    = Dict{Symbol,Array{PE_Function}}()
    for dim in keyset
        if haskey(f1.functions_, dim) & haskey(f2.functions_, dim)
            multiple = f1.functions_[dim] * f2.functions_[dim]
            if typeof(multiple) == UnivariateFunctions.Sum_Of_Functions
                funcs[dim] = multiple.functions_
            else
                funcs[dim] = [multiple]
            end
        elseif haskey(f1.functions_, dim)
            funcs[dim] = [f1.functions_[dim]]
        else
            funcs[dim] = [f2.functions_[dim]]
        end
    end
    dict_array = dict_to_array(funcs)
    mv_funcs = Multivariate_PE_Function.(new_mult, dict_array)
    if length(mv_funcs) == 1
        return mv_funcs[1]
    else
        return Multivariate_Sum_Of_Functions(mv_funcs)
    end
end
function *(f1::Multivariate_PE_Function, f2::Multivariate_Sum_Of_Functions)
    new_funcs = f1 .* f2.functions_
    return Multivariate_Sum_Of_Functions(new_funcs)
end
function *(f1::Multivariate_Sum_Of_Functions, f2::Multivariate_Sum_Of_Functions)
    return f1.functions_ * f2.functions_
end
function *(a1::Array{Multivariate_PE_Function,1}, a2::Array{Multivariate_PE_Function,1})
    new_functions_ = Array{Union{Multivariate_PE_Function,Multivariate_Sum_Of_Functions},1}()
    for f in a1
        new_new = f .* a2
        append!(new_functions_, new_new)
    end
    return Multivariate_Sum_Of_Functions(new_functions_)
end

function *(f1::Multivariate_Sum_Of_Functions, f2::Multivariate_PE_Function)
    return f2 * f1
end
function +(f1::MultivariateFunction, f2::MultivariateFunction)
    return Multivariate_Sum_Of_Functions(vcat(f1,f2))
end
function -(f1::MultivariateFunction, f2::MultivariateFunction)
    neg_f2 = -1*f2
    return Multivariate_Sum_Of_Functions(vcat(f1,neg_f2))
end
function /(f1::MultivariateFunction, f2::MultivariateFunction)
    error("It is not possible yet to divide functions by other functions")
end




"""
    evaluate(f::MultivariateFunction, coordinates::Dict{Symbol,Float64})

Evaluate a multivariate functions at coordinates specified by a dictionary. Returns a float64.
"""
function evaluate(f::Multivariate_PE_Function, coordinates::Dict{Symbol,Float64})
    vals = f.multiplier_
    contents = deepcopy(f.functions_)
    for dim in keys(contents)
        if haskey(coordinates, dim)
            vals = vals * evaluate(contents[dim], coordinates[dim])
            delete!(contents, dim)
        end
    end
    if length(contents) == 0
        return vals
    else
        return Multivariate_PE_Function(vals, contents)
    end
end
function evaluate(f::Multivariate_Sum_Of_Functions, coordinates::Dict{Symbol,Float64})
    vals = evaluate.(f.functions_, Ref(coordinates))
    if length(vals) == 0
        return 0.0
    else
        return sum(vals)
    end
end
function evaluate(f::MultivariateFunction, coordinates::Dict{Symbol,Any})
    new_coordinates = Dict{Symbol,Float64}()
    for dim in keys(coordinates)
        val = coordinates[dim]
        if typeof(val) == Float64
            new_coordinates[dim] = val
        elseif typeof(val) == Int
            new_coordinates[dim] = convert(Float64,val)
        elseif typeof(val) == Date
            new_coordinates[dim] = years_from_global_base(val)
        else
            error(string("This package does not know how to convert a ", typeof(val),  " to a Float64. Convert the type yourself to a Float64 and then try again."))
        end
    end
    return evaluate(f, new_coordinates)
end



function evaluate(f::Multivariate_PE_Function, coordinates::DataFrame)
    result = Array{Float64}(undef,size(coordinates)[1])
    result .= f.multiplier_
    if length(f.functions_) == 0
        return result
    end
    funcs = deepcopy(f.functions_)
    col_names = Symbol.(names(coordinates))
    for col in col_names
        if haskey(funcs, col)
            ff = pop!(funcs, col)
            result .= result .* evaluate.(Ref(ff), coordinates[Symbol(col)])
        end
    end
    if length(funcs) == 0
        return result
    else
        return Multivariate_PE_Function.(result, funcs)
    end
end

function evaluate(f::Multivariate_Sum_Of_Functions, coordinates::DataFrame)
    results = evaluate.(f.functions_, Ref(coordinates))
    return sum(results)
end

"""
    derivative(f::MultivariateFunction, dim::Symbol, n::Int = 1)

Return the derivative of function f. The n'th partial derivative in the dim dimension is taken. So if n = 2 and dim = "x" then the second derivative of f is taken wtih respect to x. This is returned as a MultivariateFunction.
"""
function derivative(f::Multivariate_PE_Function, dim::Symbol, n::Int = 1)
    if n > 0
        if !(haskey(f.functions_, dim))
            return Multivariate_PE_Function(0.0,Dict{Symbol,PE_Function}())
        end
        funcs = deepcopy(f.functions_)
        deriv = derivative(funcs[dim])
        if typeof(deriv) == UnivariateFunctions.Sum_Of_Functions
            func_array = Array{Multivariate_PE_Function}(undef, length(deriv.functions_))
            for i in 1:length(deriv.functions_)
                funcs[dim] = deriv.functions_[i]
                ff = Multivariate_PE_Function(f.multiplier_, deepcopy(funcs))
                func_array[i] = ff
            end
            new_func = Multivariate_Sum_Of_Functions(func_array)
            return derivative(new_func, dim, n-1)
        else
            funcs[dim] = deriv
            new_func = Multivariate_PE_Function(f.multiplier_, funcs)
            return derivative(new_func, dim, n-1)
        end
    elseif n == 0
        return f
    else
        error("Not possible to take a negative derivative. Try evaluating an integral instead.")
    end
end
function derivative(f::Multivariate_Sum_Of_Functions, dim::Symbol, num::Int = 1)
    if num > 0
        new_funcs = derivative.(f.functions_, dim, num)
        new_func = Multivariate_Sum_Of_Functions(new_funcs)
        return new_func
    elseif num == 0
        return f
    else
        error("Not possible to take a negative derivative. Try evaluating an integral instead.")
    end
end
"""
    derivative(f::MultivariateFunction, derivs::Dict{Symbol,Int})

Calculates a partial derivative of the f function specified by the derivs dictionary. If derivs = Dict(["x", "y"] .=> [2,1]) then the derivative ``\\frac{x}{y}`` will be returned as a multivariate function.
"""
function derivative(f::MultivariateFunction, derivs::Dict{Symbol,Int})
    f_new = f
    for dim in keys(derivs)
        f_new = derivative(f_new, dim, derivs[dim])
    end
    return f_new
end

## Errors from combining univariate and multivariate functions:
const multiunierror = "Algebraic operations are not allowed between univariate and multivariate functions. This
       is because the dimension name of a univariate function is not defined. To do this first convert the univariate function
       into a multivariate function by specifying a dimension name. Then algebraic operations are available."
function +(f1::UnivariateFunction, f2::MultivariateFunction)
    error(multiunierror)
end
function -(f1::UnivariateFunction, f2::MultivariateFunction)
    error(multiunierror)
end
function *(f1::UnivariateFunction, f2::MultivariateFunction)
    error(multiunierror)
end
function /(f1::UnivariateFunction, f2::MultivariateFunction)
    error(multiunierror)
end
function +(f1::MultivariateFunction, f2::UnivariateFunction)
    error(multiunierror)
end
function -(f1::MultivariateFunction, f2::UnivariateFunction)
    error(multiunierror)
end
function *(f1::MultivariateFunction, f2::UnivariateFunction)
    error(multiunierror)
end
function /(f1::MultivariateFunction, f2::UnivariateFunction)
    error(multiunierror)
end

function +(f1::Undefined_Function, f2::MultivariateFunction)
    return Undefined_Function()
end
function -(f1::Undefined_Function, f2::MultivariateFunction)
    return Undefined_Function()
end
function *(f1::Undefined_Function, f2::MultivariateFunction)
    return Undefined_Function()
end
function /(f1::Undefined_Function, f2::MultivariateFunction)
    return Undefined_Function()
end
function +(f1::MultivariateFunction, f2::Undefined_Function)
    return Undefined_Function()
end
function -(f1::MultivariateFunction, f2::Undefined_Function)
    return Undefined_Function()
end
function *(f1::MultivariateFunction, f2::Undefined_Function)
    return Undefined_Function()
end
function /(f1::MultivariateFunction, f2::Undefined_Function)
    return Undefined_Function()
end
