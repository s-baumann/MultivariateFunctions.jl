using UnivariateFunctions
using Test

# Run tests

println("Test of Univariate Functions.")
@time @test include("1_test_univariate.jl")
println("Test of MultivariateFunctions taking a date.")
@time @test include("2_test_dates.jl")
println("Test of Multivariate Functions.")
@time @test include("3_test_multivariate.jl")
println("Test of Univariate Piecewise Functions.")
@time @test include("4_piecewise_tests.jl")
println("Test of Multivariate Piecewise Functions.")
@time @test include("5_test_piecewise_multivariate.jl")
println("Test of Schumaker spline with a univariate function.")
@time @test include("6_schumaker_test.jl")
println("Test of interpolation with a univariate function.")
@time @test include("7_interpolation_test.jl")
println("Test of chebyshev fitting with univariate and multivariate functions.")
@time @test include("8_chebyshev_tests.jl")


@time @test include("6_test_regressions.jl")
@time @test include("7_test_multivariate.jl")
@time @test include("8_test_piecewise_multivariate.jl")
