
#################################################################################

struct Multivariate_Piecewise_Function <: MultivariateFunction
    # Here we use ints to represent the function i n each subcube of the space. This saves on size and computation in
    # cases where the same function is in more than one cuve.
    functions_::Array{Union{Undefined_Function,Multivariate_Sum_Of_Functions}}
    labels_::Array{Symbol,1}
    thresholds_::Dict{Symbol,Array{Float64}}
end

function get_correct_function_from_piecewise(f::Multivariate_Piecewise_Function, coordinates::Dict{Symbol,Float64})
    segment_coordinates = Array{Int,1}(undef,length(f.labels_))
    for i in 1:length(f.labels_)
        dim = f.labels_[i]
        point = coordinates[dim]
        segment_coordinates[i] = searchsortedlast(f.thresholds_[dim], coordinates[dim])
    end
    if 0 in segment_coordinates
        return Undefined_Function()
    else
        func = getindex(f.functions_, segment_coordinates...)
        return func
    end
end

function evaluate(f::Multivariate_Piecewise_Function, coordinates::Dict{Symbol,Float64})
    func = get_correct_function_from_piecewise(f, coordinates)
    return evaluate(func, coordinates)
end

function derivative(f::Multivariate_Piecewise_Function, dim::Symbol, num::Int = 1)
    if num > 0
        derivatives = derivative.(f.functions_, dim, num)
        return Multivariate_Piecewise_Function(derivatives, f.labels_, f.thresholds_)
    elseif num == 0
        return f
    else
        error("Not possible to take a negative derivative. Try evaluating an integral instead.")
    end
end

function indefinite_integral(f::Multivariate_Piecewise_Function)
    indef_integrals = indefinite_integral.( f.functions_)
    return Piecewise_Function(f.starts_, indef_integrals)
end

function +(f::Multivariate_Piecewise_Function,number::Float64)
    functions_with_addition = f.functions_ .+ number
    return Multivariate_Piecewise_Function(functions_with_addition, f.labels_, f.thresholds_)
end
function -(f::Multivariate_Piecewise_Function,number::Float64)
    functions_with_subtraction = f.functions_ .- number
    return Multivariate_Piecewise_Function(functions_with_subtraction, f.labels_, f.thresholds_)
end
function *(f::Multivariate_Piecewise_Function,number::Float64)
    functions_with_multiplication = f.functions_ .* number
    return Multivariate_Piecewise_Function(functions_with_multiplication, f.labels_, f.thresholds_)
end
function /(f::Multivariate_Piecewise_Function,number::Float64)
    functions_with_division = f.functions_ ./ number
    return Multivariate_Piecewise_Function(functions_with_division, f.labels_, f.thresholds_)
end
function +(f::Multivariate_Piecewise_Function,number::Int)
    number_as_float = convert(Float64, number)
    return +(f,number_as_float)
end
function -(f::Multivariate_Piecewise_Function,number::Int)
    number_as_float = convert(Float64, number)
    return -(f,number_as_float)
end
function *(f::Multivariate_Piecewise_Function,number::Int)
    number_as_float = convert(Float64, number)
    return *(f,number_as_float)
end
function /(f::Multivariate_Piecewise_Function,number::Int)
    number_as_float = convert(Float64, number)
    return /(f,number_as_float)
end

function get_threshold_dict(f1::Multivariate_Piecewise_Function,f2::Multivariate_Piecewise_Function)
    keyset = unique(vcat(f1.labels_, f2.labels_))
    theshold_dict = Dict{Symbol,Array{Float64,1}}()
    for k in keyset
        if (k in keys(f1.thresholds_)) & (k in keys(f2.thresholds_))
            theshold_dict[k] = unique(sort!(vcat(f1.thresholds_[k], f2.thresholds_[k])))
        elseif (k in keys(f1.thresholds_))
            theshold_dict[k] = f1.thresholds_[k]
        else
            theshold_dict[k] = f2.thresholds_[k]
        end
    end
    return theshold_dict
end

function create_common_pieces(f1::Multivariate_Piecewise_Function,f2::Multivariate_Piecewise_Function)
    thresholds_ = get_threshold_dict(f1,f2)
    labels_     = collect(keys(thresholds_))
    lengths     = length.(get.(Ref(thresholds_), labels_, 0))
    functions1_ = Array{Union{Undefined_Function,Multivariate_Sum_Of_Functions},length(labels_)}(undef, lengths...)
    functions2_ = Array{Union{Undefined_Function,Multivariate_Sum_Of_Functions},length(labels_)}(undef, lengths...)
    for i in CartesianIndices(functions1_)
        starts = Dict(labels_ .=>  getindex.(  get.(Ref(thresholds_), labels_,0)  , Tuple(i)) .+ 100eps()  ) # The 100eps() here is to ensure we are inside the hypercube rather than on a boundary.
        functions1_[i] = get_correct_function_from_piecewise(f1, starts)
        functions2_[i] = get_correct_function_from_piecewise(f2, starts)
    end
    return Multivariate_Piecewise_Function(functions1_, labels_, thresholds_) , Multivariate_Piecewise_Function(functions2_, labels_, thresholds_)
end


function *(f1::Multivariate_Piecewise_Function, f2::MultivariateFunction)
    funcs = deepcopy(f1.functions_)
    funcs = funcs .* f2
    return Multivariate_Piecewise_Function(funcs, f1.labels_, f1.thresholds_)
end
function +(f1::Multivariate_Piecewise_Function, f2::MultivariateFunction)
    funcs = deepcopy(f1.functions_)
    funcs = funcs .+ f2
    return Multivariate_Piecewise_Function(funcs, f1.labels_, f1.thresholds_)
end
function -(f1::Multivariate_Piecewise_Function, f2::MultivariateFunction)
    neg_f2 = -1*f2
    return +(f1,neg_f2)
end
function *(f1::MultivariateFunction, f2::Multivariate_Piecewise_Function)
    return *(f2,f1)
end
function +(f1::MultivariateFunction, f2::Multivariate_Piecewise_Function)
    return +(f2,f1)
end
function -(f1::MultivariateFunction, f2::Multivariate_Piecewise_Function)
    neg_f2 = -1*f2
    return +(neg_f2,f1)
end

function +(f1::Multivariate_Piecewise_Function,f2::Multivariate_Piecewise_Function)
    c_f1, c_f2 = create_common_pieces(f1,f2)
    functions_  = c_f1.functions_ .+ c_f2.functions_
    return Multivariate_Piecewise_Function(functions_, c_f1.labels_, c_f1.thresholds_)
end
function *(f1::Multivariate_Piecewise_Function,f2::Multivariate_Piecewise_Function)
    c_f1, c_f2 = create_common_pieces(f1,f2)
    functions_ = c_f1.functions_ .* c_f2.functions_
    return Multivariate_Piecewise_Function(functions_, c_f1.labels_, c_f1.thresholds_)
end
function -(f1::Multivariate_Piecewise_Function,f2::Multivariate_Piecewise_Function)
    c_f1, c_f2 = create_common_pieces(f1,f2)
    functions_ = c_f1.functions_ .- c_f2.functions_
    return Multivariate_Piecewise_Function(functions_, c_f1.labels_, c_f1.thresholds_)
end
