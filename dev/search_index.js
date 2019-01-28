var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "MultivariateFunctions",
    "title": "MultivariateFunctions",
    "category": "page",
    "text": ""
},

{
    "location": "#MultivariateFunctions-1",
    "page": "MultivariateFunctions",
    "title": "MultivariateFunctions",
    "category": "section",
    "text": "This implements single algebra and evaluation on Multivariate functions. There are a few ways in which it can be used.This can be used for approximation functions. It can currently implement OLS functions, chebyshev polynomials, the schumaker shape preserving spline and basic interpolation schemes. It can also implement Recursive Partitioning and create Multivariate Adaptive Regression (MARS) Splines. It could be extended to implement other approximation schemes.\nAs in the StochasticIntegrals.jl package this package can be used to define functions that will be the integrands in stochastic integrals. This has the benefit that the means, variances & covariances implied by these stochastic integrals can be found analytically.\nAll basic algebra and calculus on a MultivariateFunction can be done analytically.\nThe Newton\'s method is implemented so that roots and optima can be found using analytical Jacobians and Hessians."
},

{
    "location": "#Contents-1",
    "page": "MultivariateFunctions",
    "title": "Contents",
    "category": "section",
    "text": "pages = [\"index.md\",\n         \"1_structs_and_limitations.md\",\n         \"2_interpolation_methods.md\",\n         \"3_approximation_methods.md\",\n         \"4_examples_algebra.md\",\n         \"5_examples_interpolation.md\",\n         \"6_examples_approximation.md\",\n         \"99_refs.md\"]\nDepth = 2"
},

{
    "location": "1_structs_and_limitations/#",
    "page": "Structs",
    "title": "Structs",
    "category": "page",
    "text": ""
},

{
    "location": "1_structs_and_limitations/#Structs-1",
    "page": "Structs",
    "title": "Structs",
    "category": "section",
    "text": "There are four main MultivariateFunction structs that are part of this package. These are:PE_Function - This is the basic function type. It has a form of ae^b(x-base) (x-base)^d.\nSum_Of_Functions - This is an array of PE_Functions. Note that by adding PE_Functions we can replicate any given polynomial. Hence from Weierstrass\' approximation theorem we can approximate any continuous function on a bounded domain to any desired level of accuracy (whether this is practical in numerical computing depends on the function being approximated).\nPiecewise_Function - This defines a different MultivariateFunction for each part of the x domain.\nSum_Of_Piecewise_Functions - Mathematically this does the same job as a Piecewise_Function but is dramatically more efficient when the contribution of different dimensions to the Piecewise_Function is additively separable.It is possible to perform any additions, subtractions, multiplications between any two MultivariateFunctions and between Ints/Floats and any MultivariateFunction. No division is allowed and it is not possible to raise a MultivariateFunction to a negative power. This is to ensure that all Multivariatefunctions are analytically integrable and differentiable. This may change in future releases."
},

{
    "location": "1_structs_and_limitations/#Major-limitations-1",
    "page": "Structs",
    "title": "Major limitations",
    "category": "section",
    "text": "It is not possible to divide by Multivariate functions or raise them by a negative power.\nWhen multiplying PE_Functions with different base dates there is often an issue of very high or very low numbers that go outside machine precision. If one were trying to change a PE_Function from base 2010 to 50, this would not generally be possible. This is because to change ae^x-2020 to qe^x- 50 we need to premultiply the first expression by e^-1950 which is often a tiny number. In these cases it is better to do the algebra on paper and rewriting the code accordingly as often base changes cancel out on paper. It is also good to change bases as rarely as possible. If different Multivariate functions use different bases then there is a need to base change when multiplying them which can result in errors. Note that if base changes are segment in the x domain by means of a piecewise function then they should never interact meaning it is ok to use different bases here."
},

{
    "location": "1_structs_and_limitations/#Date-Handling-1",
    "page": "Structs",
    "title": "Date Handling",
    "category": "section",
    "text": "All base dates are immediately converted to floats and are not otherwise saved. Thus there is no difference between a PE_Function created with a base as a float and one created with the matching date. This is done to simplify the code. All date conversions is done by finding the year fractions between the date and the global base date of Date(2000,1,1). This particular global base date should not affect anything as long as it is consistent. It is relatively trivial to change it (in the date_conversions.jl file) and recompile however if desired."
},

{
    "location": "2_interpolation_methods/#",
    "page": "Univariate Interpolation Methods",
    "title": "Univariate Interpolation Methods",
    "category": "page",
    "text": ""
},

{
    "location": "2_interpolation_methods/#Univariate-Interpolation-Methods-1",
    "page": "Univariate Interpolation Methods",
    "title": "Univariate Interpolation Methods",
    "category": "section",
    "text": "So far this package support the following interpolation schemes for one dimensional interpolation:Constant interpolation from the left to the right. Such a Piecewise_Function spline can be constructed by the create_constant_interpolation_to_right method.\nConstant interpolation from the right to the left. Such a Piecewise_Function spline can be constructed by the create_constant_interpolation_to_left method.\nLinear interpolation. Such a Piecewise_Function spline can be constructed by the create_linear_interpolation method.\nSchumaker shape preserving spline - Such a Piecewise_Function spline can be constructed by the create_quadratic_spline method. See Judd (1998) for details on how this is done.Note that interpolation in higher dimensions is hard and no such methods are yet available in this package. There are some approximation schemes that might work in this case however as described in the next section."
},

{
    "location": "3_approximation_methods/#",
    "page": "Supported Approximation Methods",
    "title": "Supported Approximation Methods",
    "category": "page",
    "text": ""
},

{
    "location": "3_approximation_methods/#Supported-Approximation-Methods-1",
    "page": "Supported Approximation Methods",
    "title": "Supported Approximation Methods",
    "category": "section",
    "text": "In addition the following approximation schemes are available, each of which can be used in any number of dimensions (subject to having enough computational power)OLS regression - Performs an OLS regression of the data and generates a Sum_Of_Functions containing the resultant approximation. This should work well in many dimensions.\nChebyshev polynomials - Creates a Sum_Of_Functions that uses Chebyshev polynomials to approximate a certain function. Unlike the other approximation schemes this does not take in an arbitrary collection of datapoints but rather takes in a function that it evaluates at certain points in a grid to make an approximation function. This might be useful if the original function is expensive (so you want a cheaper one). You might also want to numerically integrate a function by getting a Chebyshev approximation function that can be analytically integrated. See Judd (1998) for details on how this is done.\nMars regression spline - Creates a Sum_Of_Piecewise_Functions representing a MARS regression spline. See Friedman (1991) for an explanation of the spline."
},

{
    "location": "4_examples_algebra/#",
    "page": "Examples - Algebra",
    "title": "Examples - Algebra",
    "category": "page",
    "text": ""
},

{
    "location": "4_examples_algebra/#Examples-Algebra-1",
    "page": "Examples - Algebra",
    "title": "Examples - Algebra",
    "category": "section",
    "text": ""
},

{
    "location": "4_examples_algebra/#Univariate:-Basic-algebra-1",
    "page": "Examples - Algebra",
    "title": "Univariate: Basic algebra",
    "category": "section",
    "text": "Consider we have a two functions f and g and want to add them, multiply them by some other function h, then square it and finally integrate the result between 2.0 and 2.8. This can be done analytically with MultivariateFunctions:f = PE_Function(1.0, 2.0, 4.0, 5)\ng = PE_Function(1.3, 2.0, 4.3, 2)\nh = PE_Function(5.0, 2.2, 1.0,0)\nresult_of_operations = (h*(f+g))^2\nintegral(result_of_operations, 2.0, 2.8)"
},

{
    "location": "4_examples_algebra/#Multivariate:-Basic-algebra-1",
    "page": "Examples - Algebra",
    "title": "Multivariate: Basic algebra",
    "category": "section",
    "text": "Consider we have a three functions f(x) = x^2 - 8 and g(y) = e^y and want to add them, multiply them by some other function h(xy) = 4 x e^y, then square it and finally integrate the result between 2.0 and 2.8 in the x domain and 2 and 3 in the y domain. This can be done analytically with MultivariateFunctions.The additional complication from the univariate case here is that we need to define the names of the dimensions as we have more than one dimension.f = PE_Function(1.0, Dict(:x => PE_Unit(0.0,0.0,2))) - 8\ng = PE_Function(1.0, Dict(:y => PE_Unit(1.0,0.0,0)))\nh = PE_Function(4.0, Dict([:x, :y] .=> [PE_Unit(0.0,0.0,1), PE_Unit(1.0,0.0,0)]))\nresult_of_operations = (h*(f+g))^2\nintegration_limits = Dict([:x, :y] .=> [(2.0,2.8), (2.0,3.0)])\nintegral(result_of_operations, integration_limits)"
},

{
    "location": "5_examples_interpolation/#",
    "page": "Examples - Data interpolation",
    "title": "Examples - Data interpolation",
    "category": "page",
    "text": ""
},

{
    "location": "5_examples_interpolation/#Examples-Data-interpolation-1",
    "page": "Examples - Data interpolation",
    "title": "Examples - Data interpolation",
    "category": "section",
    "text": "Suppose we have want to approximate some function with some sampled points. First to generate some pointsconst global_base_date = Date(2000,1,1)\nStartDate = Date(2018, 7, 21)\nx = Array{Date}(undef, 20)\nfor i in 1:20\n    x[i] = StartDate +Dates.Day(2* (i-1))\nend\nfunction ff(x::Date)\n    days_between = years_from_global_base(x)\n    return log(days_between) + sqrt(days_between)\nend\ny = ff.(x)Now we can generate a function that can be used to easily interpolate from the sampled points:func = create_quadratic_spline(x,y)And we can evaluate from this function and integrate it and differentiate it in the normal way:evaluate(func, Date(2020,1,1))\nevaluate.(Ref(func), [Date(2020,1,1), Date(2021,1,2)])\nevaluate(derivative(func), Date(2021,1,2))\nintegral(func, Date(2020,1,1), Date(2021,1,2))If we had wanted to interpolate instead with a constant method(from left or from right) or by linearly interpolating then we could have just generated func with a different method: create_constant_interpolation_to_left, create_constant_interpolation_to_right or create_linear_interpolation."
},

{
    "location": "6_examples_approximation/#",
    "page": "Examples - Approximation",
    "title": "Examples - Approximation",
    "category": "page",
    "text": ""
},

{
    "location": "6_examples_approximation/#Examples-Approximation-1",
    "page": "Examples - Approximation",
    "title": "Examples - Approximation",
    "category": "section",
    "text": ""
},

{
    "location": "6_examples_approximation/#OLS-approximation-1",
    "page": "Examples - Approximation",
    "title": "OLS approximation",
    "category": "section",
    "text": "If we have lots of data that we want to summarise with OLS# Generating example data\nusing Random\nusing Distributions\nusing DataStructures\nRandom.seed!(1)\nobs = 1000\nX = rand(obs)\ny = X .+ rand(Normal(),obs) .+ 7\n# And now making an approximation function\napproxFunction = create_ols_approximation(y, X, 2)"
},

{
    "location": "6_examples_approximation/#Numerical-Integration-with-Chebyshev-polynomials-1",
    "page": "Examples - Approximation",
    "title": "Numerical Integration with Chebyshev polynomials",
    "category": "section",
    "text": "And if we want to approximate the sin function in the [2.3, 5.6] bound with 7 polynomial terms and 20 approximation nodes:chebyshevs = create_chebyshev_approximation(sin, 20, 7, OrderedDict{Symbol,Tuple{Float64,Float64}}(:default => (2.3, 5.6)))We can integrate the above term in the normal way to achieve Gauss-Chebyshev quadrature:integral(chebyshevs, 2.3, 5.6)"
},

{
    "location": "6_examples_approximation/#Multivariate:-MARS-Spline-for-approximation-1",
    "page": "Examples - Approximation",
    "title": "Multivariate: MARS Spline for approximation",
    "category": "section",
    "text": "First we will generate some example data.using MultivariateFunctions\nusing Random\nusing DataFrames\nusing Distributions\nusing DataStructures\n\nRandom.seed!(1992)\nnObs = 1000\ndd = DataFrame()\ndd[:x] = rand( Normal(),nObs) + 0.1 .* rand( Normal(),nObs)\ndd[:z] = rand( Normal(),nObs) + 0.1 .* rand( Normal(),nObs)\ndd[:w] = (0.5 .* rand( Normal(),nObs)) .+ 0.7.*(dd[:z] .- dd[:x]) + 0.1 .* rand( Normal(),nObs)\ndd[:y] = (dd[:x] .*dd[:w] ) .* (dd[:z] .- dd[:w]) .+ dd[:x] + rand( Normal(),nObs)\ndd[7,:y] = 1.0\ny = :y\nx_variables = Set{Symbol}([:w, :x, :z])It is important to note here that we have a set of symbols for x_variables. This is the set of columns in the dataframe that we will use to predict y - the dependent variable.We can then create an approximation with recursive partitioning:number_of_divisions = 7\nrp_4, rp_reg_4 = create_recursive_partitioning(dd, y, x_variables, number_of_divisions; rel_tol = 1e-3)We can also create a MARS approximation spline:rp_1, rp_reg_1 = create_mars_spline(dd, y, x_variables, number_of_divisions; rel_tol = 1e-3)Note that the rel_tol here is the tolerance in the optimisation step for hinges (or divisions in the recursive partitioning case). In most applied cases it generally doesn\'t matter much if there is a hinge at 1.0006 or at 1.0007 so in most settings this can be set higher than you would generally set the tolerance for a numerical optimiser. For this reason the default value is 1e-02."
},

{
    "location": "99_refs/#",
    "page": "References",
    "title": "References",
    "category": "page",
    "text": ""
},

{
    "location": "99_refs/#References-1",
    "page": "References",
    "title": "References",
    "category": "section",
    "text": "Friedman, Jerome (1991) Multivariate Adaptive Regression Splines. The annals of Statistics 19(1). pp. 1-141.Judd, Kenneth (1998) Numerical Methods in Economics. 9780262100717. MIT Press."
},

]}
