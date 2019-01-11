## Supported Univariate Interpolation Methods
So far this package support the following interpolation schemes for one dimensional interpolation:
* Constant interpolation from the left to the right. Such a Piecewise_Function spline can be constructed by the create_constant_interpolation_to_right method.
* Constant interpolation from the right to the left. Such a Piecewise_Function spline can be constructed by the create_constant_interpolation_to_left method.
* Linear interpolation. Such a Piecewise_Function spline can be constructed by the create_linear_interpolation method.
* Schumaker shape preserving spline - Such a Piecewise_Function spline can be constructed by the create_quadratic_spline method. See Judd (1998) for details on how this is done.

Note that interpolation in higher dimensions is hard and not such methods are yet available. There are some approximation shemes that might work in this case however:

## Supported Approximation Methods

In addition the following approximation schemes are available, each of which can be used in any number of dimensions (subject to having enough computational power)
* OLS regression - Performs an OLS regression of the data and generates a Sum_Of_Functions containing the resultant approximation. This should work well in many dimensions.
* Chebyshev polynomials - Creates a Sum_Of_Functions that uses chebyshev polynomials to approximate a certain function. Unlike the other approximation schemes this does not take in an arbitrary collection of datapoints but rather takes in a function that it evaluates at certain points in a grid to make an approximation function. This might be useful if the original function is expensive (so you want a cheaper one). You might also want to numerically integrate a function by getting a chebyshev approximation function that can be analytically integrated. See Judd (1998) for details on how this is done.
* Mars regression spline - Creates a Sum_Of_Piecewise_Functions representing a MARS regression spline. See Friedman (1991) for an explanation of the spline.
