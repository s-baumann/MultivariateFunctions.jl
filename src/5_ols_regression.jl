"""
function create_ols_approximation(y::Array{Float64,1}, x::Array{Float64,1}, base_x::Float64 = 0.0, degree::Int = 1; intercept::Bool = true, dim_name::Symbol = default_symbol)
    obs = length(y)
    if degree < 0
        error("Cannot approximate with OLS with a degree that is negative")
    end
    x = x .- base_x
    if intercept
        X = ones(obs)
        for i in 1:degree
            X = hcat(X, (x .^ i))
        end
    else
        X = x
        for i in 2:degree
            X = hcat(X, (x .^ i))
        end
    end

    lm1 = fit(LinearModel,  hcat(X), y)
    beta = lm1.pp.beta0
    func_array = Array{PE_Function,1}(undef,convert(Int, intercept) + degree)
    if intercept
        func_array[1] = PE_Function(beta[1], Dict{Symbol,PE_Unit}(dim_name => PE_Unit(0.0, base_x, 0)))
    end
    for d in 1:degree
        func_array[d+convert(Int, intercept)] = PE_Function(beta[d+convert(Int, intercept)], Dict{Symbol,PE_Unit}(dim_name => PE_Unit(0.0, base_x, d)))
    end
    return Sum_Of_Functions(func_array)
end


function create_ols_approximation(y::Array{Float64,1}, x::Array{Date,1}, base_x::Date = global_base_date, degree::Int = 1, intercept::Bool = true)
    base = years_from_global_base.(base_x)
    xx   = years_from_global_base.(x)
    return create_ols_approximation(y, xx, base, degree, intercept)
end
"""
function create_ols_approximation(dd::DataFrame, y::Symbol, model::MultivariateFunction)
    X = hcat(evaluate.(model.functions_, Ref(dd))...)
    y = dd[y]
    reg = lm(X,y)
    coefficients = reg.pp.beta0
    updated_model = Sum_Of_Functions(model.functions_ .* coefficients)
    return updated_model, reg
end


function create_saturated_ols_approximation(dd::DataFrame, y::Symbol, x_variables::Array{Symbol}, degree::Int; intercept::Bool = true,  bases::Dict{Symbol,Float64} = Dict{Symbol,Float64}(x_variables .=> repeat([0.0],length(x_variables))))
    model = Array{PE_Function,1}()
    if intercept
        append!(model, [PE_Function(1.0,Dict{Symbol,PE_Unit}())] )
    end
    if degree > 0
        number_of_variables = length(x_variables)
        linear_set = Array{PE_Function,1}(undef, number_of_variables)
        for i in 1:length(x_variables)
            linear_set[i] = PE_Function(1.0,Dict{Symbol,PE_Unit}(x_variables[i] => PE_Unit(0.0,bases[x_variables[i]],1) ))
        end
        higher_order_terms = Array{Array{PE_Function,1},1}(undef,degree)
        higher_order_terms[1] = linear_set
        for i in 2:degree
            degree_terms = Array{PE_Function,1}()
            for j in 1:number_of_variables
                degree_terms = vcat(degree_terms, linear_set[j] .* hcat(higher_order_terms[i-1]))
            end
            higher_order_terms[i] = linear_set
        end
        append!(model, vcat(higher_order_terms...))
    end
    # This is buggy because we get multiple terms like (x^{2}z) comes out twice or more times. So perfect multicolinearity.
    # Easiest way will be to have the  Multivariate_Sum_Of_Functions constructor clean up duplicates. The multiplication to the multiplier_ will
    # not matter as it will get baked into the OLS coefficient anyway.
    return create_ols_approximation(dd, y, Sum_Of_Functions(model))
end

# 1D convenience functions
function create_ols_approximation(y::Array{Float64,1}, x::Array{Float64,1}, degree::Int; intercept::Bool = true, dim_name::Symbol = default_symbol, base_x::Float64 = 0.0)
    dd = DataFrame()
    dd[dim_name] = x
    dd[:y]       = y
    base_dict = Dict{Symbol,Float64}(dim_name => base_x)
    return create_saturated_ols_approximation(dd, :y, [dim_name], degree; intercept = intercept, bases = base_dict)
end

function create_ols_approximation(y::Array{Float64,1}, x::Array{Date,1}, degree::Int; intercept::Bool = true, dim_name::Symbol = default_symbol)
    return  create_ols_approximation(y, years_from_global_base.(x), degree; intercept = intercept, dim_name = dim_name, base_x = 0.0)
end
