
"""
    create_saturated_ols_approximation(dd::DataFrame, y::Symbol, x_variables::Array{Symbol,1}, degree::Int; intercept::Bool = true, bases::Dict{Symbol,Float64} = ..., weights::Union{Nothing, Vector{Float64}} = nothing)
This creates a MultivariateFunction from an OLS regression predicting some variable. You input a dataframe and specify what column in that dataframe
is to be predicted by inputting a symbol y. You also put in an array of what x_variables should be used in prediction. A saturated OLS model is then calculated up to the specified degree.

Returns a tuple `(model, regression)` where `model` is a `Sum_Of_Piecewise_Functions` and `regression` is the GLM `LinearModel` object.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted least squares regression is performed using frequency weights.
"""
function create_saturated_ols_approximation(dd::DataFrame, y::Symbol, x_variables::Array{Symbol,1}, degree::Int; intercept::Bool = true,  bases::Dict{Symbol,Float64} = Dict{Symbol,Float64}(x_variables .=> repeat([0.0],length(x_variables))), weights::Union{Nothing, Vector{Float64}} = nothing)
    model = Array{PE_Function,1}()
    if intercept
        push!(model, PE_Function(1.0))
    end
    if degree > 0
        number_of_variables = length(x_variables)
        linear_set = Array{PE_Function,1}(undef, number_of_variables)
        for i in 1:length(x_variables)
            linear_set[i] = PE_Function(1.0, UnitMap([x_variables[i] => PE_Unit(0.0,bases[x_variables[i]],1)]))
        end
        higher_order_terms = Array{Array{PE_Function,1},1}(undef,degree)
        higher_order_terms[1] = linear_set
        for i in 2:degree
            degree_terms = Array{PE_Function,1}()
            for j in 1:number_of_variables
                append!(degree_terms, linear_set[j] .* higher_order_terms[i-1])
            end
            higher_order_terms[i] = degree_terms
        end
        append!(model, vcat(higher_order_terms...))
    end
    sum_of_functions = Sum_Of_Functions(model) # We put it through here to remove duplicates.
    return create_ols_approximation(dd, y, sum_of_functions; weights=weights)
end

"""
    create_ols_approximation(dd::DataFrame, y::Symbol, model::Array; dropcollinear = true, weights = nothing)
    create_ols_approximation(dd::DataFrame, y::Symbol, model::Sum_Of_Functions; weights = nothing)
    create_ols_approximation(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions; weights = nothing)
Creates a MultivariateFunction from an OLS regression predicting some variable. You input a dataframe and specify what column in that dataframe
is to be predicted by inputting a symbol y. You also input the regression model as an array of basis functions.
Each function that is input will be multiplied by the OLS coefficient and will return a new function with these coefficients
incorporated.

Returns a tuple `(model, regression)` where `model` is a `Sum_Of_Piecewise_Functions` and `regression` is the GLM `LinearModel` object.

If `weights` is provided (a `Vector{Float64}` with one non-negative entry per row), a weighted least squares regression is performed using frequency weights.
"""
function create_ols_approximation(dd::DataFrame, y::Symbol, model::Array; dropcollinear = true, weights::Union{Nothing, Vector{Float64}} = nothing)
    X = hcat(evaluate.(model, Ref(dd))...)
    y = dd[!, y]
    if weights === nothing
        reg = fit(LinearModel, X, y; dropcollinear=dropcollinear)
    else
        reg = fit(LinearModel, X, y; dropcollinear=dropcollinear, weights=FrequencyWeights(weights))
    end
    coefficients = reg.pp.beta0
    updated_model = Sum_Of_Piecewise_Functions(model .* coefficients)
    return updated_model, reg
end
function create_ols_approximation(dd::DataFrame, y::Symbol, model::Sum_Of_Functions; weights::Union{Nothing, Vector{Float64}} = nothing)
    return create_ols_approximation(dd, y, model.functions_; weights=weights)
end
function create_ols_approximation(dd::DataFrame, y::Symbol, model::Sum_Of_Piecewise_Functions; weights::Union{Nothing, Vector{Float64}} = nothing)
    return create_ols_approximation(dd, y, vcat( model.functions_, model.global_funcs_ ); weights=weights)
end
"""
    create_ols_approximation(y::Array{Float64,1}, x::Array{Float64,1}, degree::Int; intercept::Bool = true, dim_name::Symbol = default_symbol, base_x::Float64 = 0.0, weights = nothing)
    create_ols_approximation(y::Array{Float64,1}, x::Array{Date,1}, degree::Int; intercept::Bool = true, dim_name::Symbol = default_symbol, base_date::Date = global_base_date, weights = nothing)

This predicts a linear relationship between the y and x arrays and creates a MultivariateFunction containing the approximation function. The degree specifies how many higher
order terms of x should be used (for instance degree 2 implies x and x^2 are both used to predict y).

If `weights` is provided, a weighted least squares regression is performed.
"""
function create_ols_approximation(y::Array{Float64,1}, x::Array{Float64,1}, degree::Int; intercept::Bool = true, dim_name::Symbol = default_symbol, base_x::Float64 = 0.0, weights::Union{Nothing, Vector{Float64}} = nothing)
    dd = DataFrame()
    dd[!, dim_name] = x
    dd[!, :y]       = y
    base_dict = Dict{Symbol,Float64}(dim_name => base_x)
    return create_saturated_ols_approximation(dd, :y, [dim_name], degree; intercept = intercept, bases = base_dict, weights = weights)
end

function create_ols_approximation(y::Array{Float64,1}, x::Array{Date,1}, degree::Int; intercept::Bool = true, dim_name::Symbol = default_symbol, base_date::Date = global_base_date, weights::Union{Nothing, Vector{Float64}} = nothing)
    return  create_ols_approximation(y, years_from_global_base.(x), degree; intercept = intercept, dim_name = dim_name, base_x = years_from_global_base(base_date), weights = weights)
end
