# API Reference

## Types

```@docs
PE_Unit
PE_Function
Sum_Of_Functions
Piecewise_Function
Sum_Of_Piecewise_Functions
```

## Date Handling

```@docs
years_from_global_base
years_between
period_length
```

## Evaluation and Inspection

```@docs
evaluate
underlying_dimensions
≂
rebadge
change_base
```

## Algebra

```@docs
Base.:+
Base.:-
Base.:*
Base.:/
Base.:^
```

## Calculus

```@docs
derivative
all_derivatives
integral
MultivariateFunctions.Hessian
jacobian
uniroot
find_local_optima
MultivariateFunctions.hypercubes_to_integrate
```

## Interpolation

```@docs
create_quadratic_spline
create_linear_interpolation
create_constant_interpolation_to_left
create_constant_interpolation_to_right
```

## Approximation

```@docs
get_chebyshevs_up_to
create_chebyshev_approximation
create_ols_approximation
create_saturated_ols_approximation
create_mars_spline
create_recursive_partitioning
trim_mars_spline
```
