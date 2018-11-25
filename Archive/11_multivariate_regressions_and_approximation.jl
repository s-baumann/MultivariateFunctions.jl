function create_saturated_ols_approximation(dd::DataFrame, y::Symbol, x_variables::Array{Symbol}, degree::Int, intercept::Bool = true)
    if length(x_variables) == 1
         x = x_variables[1]
         univariate_reg = create_ols_approximation(dd[y], dd[x], 0.0, degree, intercept)
         funcs = Dict{Symbol,PE_Function}.( Ref(x) .=> univariate_reg.functions_)
         multivariate_reg = Multivariate_PE_Function.(0.0, funcs)
         return Multivariate_Sum_Of_Functions(multivariate_reg)
    end
    model = Array{Multivariate_PE_Function,1}()
    if intercept
        append!(model, [Multivariate_PE_Function(1.0,Dict{Symbol,PE_Function}())] )
    end
    if degree > 0
        linear     = PE_Function(1.0,0.0,0.0,1)
        linear_set = Multivariate_PE_Function.(1.0,Dict{Symbol,PE_Function}.(x_variables .=> Ref(linear)))
        higher_order_terms = Array{Array{Multivariate_PE_Function,1},1}(undef,degree)
        higher_order_terms[1] = linear_set
        for i in 2:degree
             terms = linear_set * higher_order_terms[i-1]
             higher_order_terms[i] = terms.functions_
        end
        append!(model, vcat(higher_order_terms...))
    end
    # This is buggy because we get multiple terms like (x^{2}z) comes out twice or more times. So perfect multicolinearity.
    # Easiest way will be to have the  Multivariate_Sum_Of_Functions constructor clean up duplicates. The multiplication to the multiplier_ will
    # not matter as it will get baked into the OLS coefficient anyway.
    return create_ols_approximation(dd, y, Multivariate_Sum_Of_Functions(model))
end

function create_ols_approximation(dd::DataFrame, y::Symbol, model::MultivariateFunction)
    X = hcat(evaluate.(model.functions_, Ref(dd))...)
    y = dd[y]
    reg = lm(X,y)
    coefficients = reg.pp.beta0
    updated_model = Multivariate_Sum_Of_Functions(model.functions_ .* coefficients)
    return updated_model, reg
end


#TODO
# OLS and Chebyshev approximations
# MARS regressions.
