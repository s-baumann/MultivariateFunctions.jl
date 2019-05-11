"""
    create_quadratic_spline(x::Array{Date,1},y::Array{Float64,1} ; gradients::Union{missing,Array{Float64,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,Float64} = missing, right_gradient::Union{Missing,Float64} = missing, dim_name::Symbol = default_symbol)
    create_quadratic_spline(x::Array{Int,1},y::Array{Float64,1} ; gradients::Union{missing,Array{Float64,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,Float64} = missing, right_gradient::Union{Missing,Float64} = missing, dim_name::Symbol = default_symbol)
    create_quadratic_spline(x::Array{Float64,1},y::Array{Float64,1} ; gradients::Union{missing,Array{Float64,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,Float64} = missing, right_gradient::Union{Missing,Float64} = missing, dim_name::Symbol = default_symbol)
    create_quadratic_spline(schum::Schumaker; dim_name::Symbol = default_symbol)

Create a quadratic spline. The spline is a Schumaker shape-preserving spline which is taken from the SchumakerSpline.jl package.
"""
function create_quadratic_spline(x::Array{Date,1},y::Array{T,1} ; gradients::Union{Missing,Array{T,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,T} = missing, right_gradient::Union{Missing,T} = missing, dim_name::Symbol = default_symbol) where T<:Real
    x_as_Ts = convert.(Ref(T), years_from_global_base.(x))
    return create_quadratic_spline(x_as_Ts, y; gradients = gradients, extrapolation = extrapolation, left_gradient = left_gradient, right_gradient = right_gradient, dim_name = dim_name)
end

function create_quadratic_spline(x::Array{DatePeriod,1},y::Array{T,1} ; gradients::Union{Missing,Array{T,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,T} = missing, right_gradient::Union{Missing,T} = missing, dim_name::Symbol = default_symbol)  where T<:Real
    x_as_Ts = convert.(Ref(T), period_length.(x))
    return create_quadratic_spline(x_as_Ts, y; gradients = gradients, extrapolation = extrapolation, left_gradient = left_gradient, right_gradient = right_gradient, dim_name = dim_name)
end

function create_quadratic_spline(x::Array{R,1},y::Array{T,1} ; gradients::Union{Missing,Array{T,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,T} = missing, right_gradient::Union{Missing,T} = missing, dim_name::Symbol = default_symbol)  where T<:Real  where R<:Real
    x_as_Ts = convert.(Ref(T), x)
    return create_quadratic_spline(x_as_Float64s, y; gradients = gradients, extrapolation = extrapolation, left_gradient = left_gradient, right_gradient = right_gradient, dim_name = dim_name)
end

function create_quadratic_spline(x::Array{T,1},y::Array{T,1} ; gradients::Union{Missing,Array{T,1}} = missing, extrapolation::Schumaker_ExtrapolationSchemes = Curve, left_gradient::Union{Missing,T} = missing, right_gradient::Union{Missing,T} = missing, dim_name::Symbol = default_symbol)  where T<:Real
    schum = Schumaker(x, y; gradients = gradients, extrapolation = extrapolation, left_gradient = left_gradient, right_gradient = right_gradient)
    return create_quadratic_spline(schum; dim_name = dim_name)
end

function create_quadratic_spline(schum::Schumaker{T}; dim_name::Symbol = default_symbol) where T<:Real
    starts_ = schum.IntStarts_
    coefficients = schum.coefficient_matrix_
    number_of_intervals = size(coefficients)[1]
    funcs_ = Array{Sum_Of_Functions{T},1}(undef, number_of_intervals)
    for i in 1:number_of_intervals
        quadratic = PE_Function(coefficients[i,1], Dict{Symbol,PE_Unit{T}}(dim_name => PE_Unit(0.0, starts_[i], 2)))
        linear    = PE_Function(coefficients[i,2], Dict{Symbol,PE_Unit{T}}(dim_name => PE_Unit(0.0, starts_[i], 1)))
        constant  = PE_Function(coefficients[i,3], Dict{Symbol,PE_Unit{T}}(dim_name => PE_Unit(0.0, 0.0       , 0)))
        polynomial = Sum_Of_Functions([quadratic, linear, constant])
        funcs_[i] = polynomial
    end
    thresholds = OrderedDict{Symbol,Array{T,1}}(dim_name => starts_)
    return Piecewise_Function(funcs_, thresholds)
end

"""
    create_constant_interpolation_to_right(x::Array{Date,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    create_constant_interpolation_to_right(x::Array{DatePeriod,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    create_constant_interpolation_to_right(x::Array{T,1},y::Array{R,1}; dim_name::Symbol = default_symbol) where T<:Real where R<:Real

Create a piecewise constant one-dimensional function which carries values from the left to the right.
"""
function create_constant_interpolation_to_right(x::Array{Date,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    x_T = convert.(Ref(T), years_from_global_base.(x))
    return create_constant_interpolation_to_right(x_T,y; dim_name = dim_name)
end

function create_constant_interpolation_to_right(x::Array{DatePeriod,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    x_as_Ts = convert.(Ref(T), period_length.(x))
    return create_constant_interpolation_to_right(x_as_Ts, y; dim_name = dim_name)
end

function create_constant_interpolation_to_right(x::Array{T,1},y::Array{R,1}; dim_name::Symbol = default_symbol) where T<:Real where R<:Real
    promo_type = promote_type(T,R)
    x_ = convert(Array{promo_type,1}, vcat(-Inf,x))
    y  = convert(Array{promo_type,1}, vcat(y[1], y))
    funcs_ = PE_Function.(y)
    thresholds_ = OrderedDict{Symbol,Array{promo_type,1}}(dim_name => x_)
    return Piecewise_Function(funcs_, thresholds_)
end

"""
    create_constant_interpolation_to_left(x::Array{Date,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    create_constant_interpolation_to_left(x::Array{DatePeriod,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    create_constant_interpolation_to_left(x::Array{T,1},y::Array{R,1}; dim_name::Symbol = default_symbol) where T<:Real where R<:Real

Create a piecewise constant one-dimensional function which carries values from the right to the left.
"""
function create_constant_interpolation_to_left(x::Array{Date,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    x_T = convert.(Ref(T), years_from_global_base.(x))
    return create_constant_interpolation_to_left(x_T , y ; dim_name = dim_name)
end

function create_constant_interpolation_to_left(x::Array{DatePeriod,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    x_as_Ts = convert.(Ref(T), period_length.(x))
    return create_constant_interpolation_to_left(x_as_Ts, y; dim_name = dim_name)
end

function create_constant_interpolation_to_left(x::Array{T,1},y::Array{R,1}; dim_name::Symbol = default_symbol)  where T<:Real where R<:Real
    promo_type = promote_type(T,R)
    x_ = convert(Array{promo_type,1}, vcat(-Inf,x[1:(length(x)-1)]))
    y2 = convert(Array{promo_type,1}, y)
    funcs_ = PE_Function.(y2)
    thresholds_ = OrderedDict{Symbol,Array{Float64,1}}(dim_name => x_)
    return Piecewise_Function(funcs_, thresholds_)
end

"""
    create_linear_interpolation(x::Array{Date,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    create_linear_interpolation(x::Array{DatePeriod,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    create_linear_interpolation(x::Array{T,1},y::Array{R,1}; dim_name::Symbol = default_symbol) where T<:Real where R<:Real

Create a piecewise linear one-dimensional function which interpolates linearly between datapoints.
"""
function create_linear_interpolation(x::Array{Date,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    x_Float = convert.(Ref(T), years_from_global_base.(x))
    return create_linear_interpolation(x_Float,y; dim_name = dim_name)
end

function create_linear_interpolation(x::Array{DatePeriod,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    x_as_Ts = convert.(Ref(T), period_length.(x))
    return create_linear_interpolation(x_as_Ts, y; dim_name = dim_name)
end

function create_linear_interpolation(x::Array{T,1},y::Array{R,1}; dim_name::Symbol = default_symbol) where T<:Real where R<:Real
    promo_type = promote_type(T,R)
    x_as_promos = convert.(Ref(promo_type), x)
    y_as_promos = convert.(Ref(promo_type), y)
    return create_linear_interpolation(x_as_promos, y_as_promos; dim_name = dim_name)
end

function create_linear_interpolation(x::Array{T,1},y::Array{T,1}; dim_name::Symbol = default_symbol) where T<:Real
    len = length(x)
    if len < 2
        error("Insufficient data to linearly interpolate")
    end
    starts_ = Array{T,1}(undef, len-1)
    funcs_  = Array{Sum_Of_Functions{T},1}(undef, len-1)
    coefficient = (y[2] - y[1])/(x[2] - x[1])
    starts_[1] = -Inf
    con = PE_Function(y[1])
    lin = PE_Function(coefficient,Dict{Symbol,PE_Unit{T}}(dim_name => PE_Unit(0.0, x[1], 1)))
    funcs_[1]  = con + lin
    if len > 2
        for i in 2:(len-1)
            starts_[i] = x[i]
            coefficient = (y[i+1] - y[i])/(x[i+1] - x[i])
            con = PE_Function(y[i])
            lin = PE_Function(coefficient,Dict{Symbol,PE_Unit{T}}(dim_name => PE_Unit(0.0, x[i], 1)))
            funcs_[i]  = con + lin
        end
    end
    thresholds_ = OrderedDict{Symbol,Array{T,1}}(dim_name => starts_)
    return Piecewise_Function(funcs_, thresholds_)
end
