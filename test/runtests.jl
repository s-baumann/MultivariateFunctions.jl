using MultivariateFunctions
using Test

# Run tests. Each file's assertions are @test lines; the @testset wrappers mean every
# assertion is reported with its file and line, and all files run even if one fails.

@testset "MultivariateFunctions" begin
    println("Test of Univariate Functions.")
    @time @testset "Univariate Functions" begin include("1_test_univariate.jl") end
    println("Test of MultivariateFunctions taking a date.")
    @time @testset "Dates" begin include("2_test_dates.jl") end
    println("Test of Multivariate Functions.")
    @time @testset "Multivariate Functions" begin include("3_test_multivariate.jl") end
    println("Test of Univariate Piecewise Functions.")
    @time @testset "Univariate Piecewise" begin include("4_piecewise_tests.jl") end
    println("Test of Multivariate Piecewise Functions.")
    @time @testset "Multivariate Piecewise" begin include("5_test_piecewise_multivariate.jl") end
    println("Test of Rootfinding and optimisation spline.")
    @time @testset "Rootfinding and optimisation" begin include("6_optimisation_and_rootfinder.jl") end
    println("Test of interpolation with a univariate function.")
    @time @testset "Schumaker spline" begin include("7_schumaker_test.jl") end
    println("Test of additional interpolation methods.")
    @time @testset "Interpolation" begin include("8_interpolation_test.jl") end
    println("Test of chebyshev fitting with univariate and multivariate functions.")
    @time @testset "Chebyshev approximation" begin include("9_chebyshev_tests.jl") end
    println("Test of ols regressions with univariate and multivariate functions.")
    @time @testset "OLS regressions" begin include("10_test_regressions.jl") end
    println("Test of high dimensional algorithms.")
    @time @testset "High dimensional algorithms" begin include("11_test_Piecewise_ML_Algos.jl") end
    println("Regression test of exact numerical outputs.")
    @time @testset "Numerical regression pins" begin include("12_regression_test.jl") end
    println("Test of monotonic MARS spline.")
    @time @testset "Monotonic MARS" begin include("13_test_monotonic_mars.jl") end
    println("Test of multivariate fitters.")
    @time @testset "Multivariate fitters" begin include("14_test_multivariate_fitters.jl") end
    println("Test of observation weights.")
    @time @testset "Observation weights" begin include("15_test_weights.jl") end
end
