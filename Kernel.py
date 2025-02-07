import sys, os
import jax.numpy as jnp

def L(x):
    """_Laguerre polynomial, 
    Double the polynomial order, plus 2,
    This is the order of convergence of the vortex method_

    Args:
        x (_type_): _typically r_

    Returns:
        _type_: _Laguerre polynomial evaluated at x_
    """
    ans = 1 - 3*x + 3*x**2 - x**3/6 # 3rd order double then plus 2 is the order of method
    #ans = 1 - x # 1st order laguere poly, is a quadratic in r, and a 4th order method. 
    #ans = 1/720 * (720 - 4320*x + 5400*x**2 - 2400*x**3 + 450*x**4 - 36*x**5 + 1*x**6) #6, 12, 14.
    return ans