function marginal_integral(f::Multivariate_PE_Function, upper_lim::Float64, lower_lim::Float64, dimension_to_integrate::Symbol)
    functions    = deepcopy(f.functions_)
    dim_function = pop!(functions, dimension_to_integrate)
    int          = evaluate_integral(dim_function, lower_lim, upper_lim)
    return Multivariate_PE_Function(int * f.multiplier_, functions)
end

function marginal_integral(f::Multivariate_Sum_Of_Functions, upper_lim::Float64, lower_lim::Float64, dimension_to_integrate::Symbol)
    functions    = deepcopy(f.functions_)
    ints          = evaluate_integral.(functions, lower_lim, upper_lim)
    return Multivariate_Sum_Of_Functions(ints)
end

function marginal_integral(f::Multivariate_PE_Function, upper_lims::Dict{Symbol,Float64}, lower_lims::Dict{Symbol,Float64})
    g = deepcopy(f)
    dimensions_to_integrate = keys(upper_lims)
    for d in dimensions_to_integrate
        upper_limit = upper_lims[d]
        lower_limit = lower_lims[d]
        g = marginal_integral(g, upper_limit, lower_limit, d)
    end
    return g
end

function marginal_integral(f::Multivariate_Sum_Of_Functions, upper_lims::Dict{Symbol,Float64}, lower_lims::Dict{Symbol,Float64})
    return sum(marginal_integral.(f.functions_, Ref(upper_lims), Ref(lower_lims)))
end

function evaluate_integral(f::Multivariate_PE_Function, upper_lims::Dict{Symbol,Float64}, lower_lims::Dict{Symbol,Float64})
    return convert(Float64, marginal_integral(f,upper_lims, lower_lims))
end

function evaluate_integral(f::Multivariate_Sum_Of_Functions, upper_lims::Dict{Symbol,Float64}, lower_lims::Dict{Symbol,Float64})
    return convert(Float64, marginal_integral(f,upper_lims, lower_lims))
end

function hypercubes_to_integrate(f::Multivariate_Piecewise_Function, upper_lims::Dict{Symbol,Float64}, lower_lims::Dict{Symbol,Float64})
    ks= sort(collect(keys(upper_lims)))
    new_dict = Dict{Symbol,Array{Float64}}()
    for dim in ks
        lower = lower_lims[dim]
        upper = upper_lims[dim]
        if lower >= upper
            error(string("The lower limit of integration is higher than the upper for dimension ", dim))
        end
        thresholds = f.thresholds_[dim]
        censored_thresholds = vcat(lower, thresholds[thresholds .> lower + 10eps()])
        censored_thresholds2 = vcat(censored_thresholds[censored_thresholds .< upper] , upper)
        new_dict[dim] = censored_thresholds2
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

function evaluate_integral(f::Multivariate_Piecewise_Function, upper_lims::Dict{Symbol,Float64}, lower_lims::Dict{Symbol,Float64})
    new_lows, new_highs = hypercubes_to_integrate(f, upper_lims, lower_lims)
    ints = Array{Float64,1}(undef,length(new_lows))
    for i in 1:length(ints)
        ff = get_correct_function_from_piecewise(f, new_lows[i])
        ints[i] = marginal_integral(ff, new_highs[i], new_lows[i])
    end
    return sum(ints)
end
