using MultivariateFunctions
using DataStructures: OrderedDict
using DataFrames

tol = 1e-14

# =====================================================
# Direct tests of chebyshevs.jl — first kind
# =====================================================

# Test get_chebyshevs_up_to returns correct number
cheb1 = get_chebyshevs_up_to(1)
length(cheb1) == 1

# T_0(x) = 1
abs(evaluate(cheb1[1], 0.5) - 1.0) < tol
abs(evaluate(cheb1[1], -3.0) - 1.0) < tol

# T_0 through T_4
cheb5 = get_chebyshevs_up_to(5)
length(cheb5) == 5

# T_1(x) = x
abs(evaluate(cheb5[2], 0.7) - 0.7) < tol
abs(evaluate(cheb5[2], -0.3) - (-0.3)) < tol

# T_2(x) = 2x^2 - 1
x_test = 0.6
abs(evaluate(cheb5[3], x_test) - (2*x_test^2 - 1)) < tol

# T_3(x) = 4x^3 - 3x
abs(evaluate(cheb5[4], x_test) - (4*x_test^3 - 3*x_test)) < tol

# T_4(x) = 8x^4 - 8x^2 + 1
abs(evaluate(cheb5[5], x_test) - (8*x_test^4 - 8*x_test^2 + 1)) < tol

# Chebyshev identity: T_n(cos(θ)) = cos(nθ)
θ = 0.8
for n in 0:4
    abs(evaluate(cheb5[n+1], cos(θ)) - cos(n*θ)) < 1e-10
end

# =====================================================
# Second kind Chebyshevs
# =====================================================

cheb2nd = get_chebyshevs_up_to(5, false)
length(cheb2nd) == 5

# U_0(x) = 1
abs(evaluate(cheb2nd[1], x_test) - 1.0) < tol

# U_1(x) = 2x
abs(evaluate(cheb2nd[2], x_test) - 2*x_test) < tol

# U_2(x) = 4x^2 - 1
abs(evaluate(cheb2nd[3], x_test) - (4*x_test^2 - 1)) < tol

# U_3(x) = 8x^3 - 4x
abs(evaluate(cheb2nd[4], x_test) - (8*x_test^3 - 4*x_test)) < tol

# U_4(x) = 16x^4 - 12x^2 + 1
abs(evaluate(cheb2nd[5], x_test) - (16*x_test^4 - 12*x_test^2 + 1)) < tol

# Second kind identity: U_n(cos(θ)) = sin((n+1)θ) / sin(θ)
for n in 0:4
    abs(evaluate(cheb2nd[n+1], cos(θ)) - sin((n+1)*θ)/sin(θ)) < 1e-10
end

# =====================================================
# Custom dim_name
# =====================================================

cheb_custom = get_chebyshevs_up_to(3; dim_name = :t)
underlying_dimensions(cheb_custom[2]) == Set([:t])
abs(evaluate(cheb_custom[2], Dict(:t => 0.5)) - 0.5) < tol

# =====================================================
# Old spelling alias
# =====================================================

cheb_old = get_chevyshevs_up_to(3)
abs(evaluate(cheb_old[3], x_test) - evaluate(cheb5[3], x_test)) < tol

# =====================================================
# Chebyshev approximation
func = sin
nodes  =  12
degree =  8
left   = -2.0
right  =  5.0
limits =  OrderedDict{Symbol,Tuple{Float64,Float64}}([:default] .=> [(left, right)])
approxim = create_chebyshev_approximation(func, nodes, degree, limits)
X = convert(Array{Float64,1}, left:0.01:right)
y = func.(X)
y_approx = evaluate.(Ref(approxim), X)
maximum(abs.(y .- y_approx)) < 0.01

func = exp
nodes  =  12
degree =  8
left   =  1.0
right  =  5.0
limits =  OrderedDict{Symbol,Tuple{Float64,Float64}}([:default] .=> [(left, right)])
approxim = create_chebyshev_approximation(func, nodes, degree, limits)
X = convert(Array{Float64,1}, left:0.01:right)
y = func.(X)
y_approx = evaluate.(Ref(approxim), X)
maximum(abs.(y .- y_approx)) < 0.01


function func1(a::Float64,b::Float64,c::Float64)
    return sin(a)* c + a * sqrt(b)
end
func = func1
nodes  =  8
degree =  4
function_takes_Dict = false
limits = OrderedDict{Symbol,Tuple{Float64,Float64}}([:a, :b, :c] .=> [(-2.0,0.5), (0.0,4.0), (5.0,11.0)])
approxim = create_chebyshev_approximation(func, nodes, degree, limits, function_takes_Dict)
st =  0.99
d = vcat(Iterators.product(convert(Array{Float64,1}, limits[:a][1]:st:limits[:a][2]),convert(Array{Float64,1}, limits[:b][1]:st:limits[:b][2]),convert(Array{Float64,1}, limits[:c][1]:st:limits[:c][2]))...)
dd = DataFrame()
dd[!, :a], dd[!, :b], dd[!, :c]  = vcat.(d...)
y = Array{Float64,1}(undef, size(dd)[1])
for i in 1:length(y)
    y[i] = func.(dd[i,:a], dd[i,:b], dd[i,:c])
end
y_approx = evaluate(approxim, dd)
maximum(abs.(y .- y_approx)) < 0.5
# This is commented out to avoid the extra dependency. It should be true though.
#using Distributions
#cor(y,y_approx) > 0.9999

function func2(dd::Dict{Symbol,Float64})
    return sin(dd[:a])* cos(dd[:c])/ ((1+dd[:b])^2) +  sqrt(1.0+dd[:b])
end
func = func2
nodes  =  24
degree =  9
function_takes_Dict = true
limits = OrderedDict{Symbol,Tuple{Float64,Float64}}([:a, :b, :c] .=> [(-2.0,1.0), (0.1,0.15), (5.0,11.0)])
approxim = create_chebyshev_approximation(func, nodes, degree, limits, function_takes_Dict)
st =  0.6
d = vcat(Iterators.product(convert(Array{Float64,1}, limits[:a][1]:st:limits[:a][2]),convert(Array{Float64,1}, limits[:b][1]:st:limits[:b][2]),convert(Array{Float64,1}, limits[:c][1]:st:limits[:c][2]))...)
dd = DataFrame()
dd[!, :a], dd[!, :b], dd[!, :c]  = vcat.(d...)
y = Array{Float64,1}(undef, size(dd)[1])
for i in 1:length(y)
    dic = Dict{Symbol,Float64}([:a,:b,:c] .=> [dd[i,:a], dd[i,:b], dd[i,:c]])
    y[i] = func(dic)
end
y_approx = evaluate(approxim, dd)
y .- y_approx
maximum(abs.(y .- y_approx)) < 0.001
#using Distributions
#cor(y,y_approx) > 0.999999
