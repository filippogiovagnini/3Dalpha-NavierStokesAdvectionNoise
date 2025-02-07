import sys, os
import numpy as np
from jax import *
from Kernel import *


def VelocityUV(pos, vec, delta, NM):
    vel = Velocity_at_Field(pos, vec, delta, NM)
    return vel

def Velocity_at_Field(pos, vec, delta, NM):
    """
    we extend VelocityUV, to evaluate at an arbitrary set of values defined by xarrd,yarrd.
    
    Args:
        pos (array): (N, N, N, 3) 
        vec (array): (N, N, N, 3) 
        carr (array): (N, N, N) 
        delta (float): this is the parameter that controls the smoothness of the kernel
    """

    # diffs[i, j, k, i', j', k', :]: pos[i, j, k, :] - pos[i', j', k', :]
    # rsq[i, j, k, i', j', k'] = || pos[i, j, k] - pos[i', j', k'] ||^2
    diffs = pos[:, :, :, jnp.newaxis, jnp.newaxis, jnp.newaxis, :] - pos[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :, :]
    rsq = ((diffs)**2).sum(axis=-1)
    denominator1 = jnp.where(rsq<1e-12, 1e-12, rsq) # prevent division by zero numerically, sensitive
    denominator2 = jnp.power(denominator1, 3/2)
    #mol = (1 - L(rsq /delta**2 )*jnp.exp(-rsq /delta**2)) # uses p-th order kernel. specified by L

    # In the following vec_x has to be an array with shape (NM^3,). Same thing for vec_y and vec_z.
    # Instead, x_diff_2, y_diff_2 and z_diff_2 are arrays with shape (NM^3, NM^3)
    
    # crosses[i, j, k, i', j', k', :] = (pos[i, j, k, :] - pos[i', j', k', :]) x vec[i', j', k', :]
    vec_enhanced = vec[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :, :]
    cross_product = jnp.cross(diffs, vec_enhanced, axisa=-1, axisb=-1)
    crosses_divided = cross_product / denominator2[..., jnp.newaxis]
    vel = (1/NM)*(crosses_divided).sum(axis=(-4, -3, -2))
    return vel

def step(pos, vec_matrix, initial_positions, dt, delta, NM):
    """
    Forward Euler flow map
    
    Args:
        pos (array): (N, N, N, 3) 
        vec (array): (N, N, N, 3) 
        carr (array): (N, N, N) 
        dt (float): 
        delta (float): 
    """
    vec = jnp.einsum('...ij,...j->...i', vec_matrix, initial_positions)
    vel = Velocity_at_Field(pos, vec, delta, NM)
    grad_vel = gradient_of_flow_at_a_point(pos, vec, dt, delta, NM)
    pos = pos + dt*vel
    vec_matrix = vec_matrix + dt * (grad_vel @ vec_matrix)
    return pos, vec_matrix

def gradient_of_flow_at_a_point(pos, vec, dt, delta, NM):

    # diffs has shape (NM, NM, NM, NM, NM, NM, 3)
    diffs = pos[:, :, :, jnp.newaxis, jnp.newaxis, jnp.newaxis, :] - pos[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :, :]
    rsq = ((diffs)**2).sum(axis=-1)
    denominator1 = jnp.where(rsq<1e-12, 1e-12, rsq) # prevent division by zero numerically, sensitive
    denominator2 = jnp.power(denominator1, 3/2)
    denominator3 = jnp.power(denominator1, 5/2)
    #mol = (1 - L(rsq /delta**2 )*jnp.exp(-rsq /delta**2)) # uses p-th order kernel. specified by L
    denominator2_expanded = denominator2[..., jnp.newaxis, jnp.newaxis]
    denominator3_expanded = denominator3[..., jnp.newaxis, jnp.newaxis]
    # In the following vec_x has to be an array with shape (NM^3,). Same thing for vec_y and vec_z.
    # Instead, x_diff_2, y_diff_2 and z_diff_2 are arrays with shape (NM^3, NM^3)

    # crosses[i, j, k, i', j', k', :] = (pos[i, j, k, :] - pos[i', j', k', :]) x vec[i', j', k', :]
    identity_matrix = jnp.eye(3)
    expanded_identity = identity_matrix[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]
    grad_K_1 = expanded_identity / denominator2_expanded
    grad_K_2 = jnp.einsum('...i,...j->...ij', diffs, diffs) / denominator3_expanded
    grad_K = grad_K_1 - 3 * grad_K_2

    first_column = grad_K[..., 0]
    second_column = grad_K[..., 1]
    third_column = grad_K[..., 2]
    first_column = jnp.cross(first_column, vec[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :, :], axisa=-1, axisb=-1)
    second_column = jnp.cross(second_column, vec[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :, :], axisa=-1, axisb=-1)
    third_column = jnp.cross(third_column, vec[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :, :], axisa=-1, axisb=-1)

    vel = jnp.stack([first_column, second_column, third_column], axis=-1)

    vel = (1/NM)*(vel).sum(axis=(-5, -4, -3))
    return vel


def integrate(step, pos, vec_matrix, initial_positions, dt, delta, NM):
    """_integration of scheme_
    """
    init = stacked(pos, vec_matrix, initial_positions)
    T = 2 #2**-4 # affect nt. 
    dt = 2 ** -3 # affect nt. # -4 seems ecessive
    visc = 1 #this is viscosity and causes points to drift away from another, see the scale on the x,y axis. 
    nt = int(T/dt)
    dts = jnp.ones(nt) * dt

    def body(init, _):
        delta = 0.1
        NM = 10
        dt = 0.125
        pos, vec_matrix, initial_positions = unstacked(init)
        pos, vec_matrix = step(pos, vec_matrix, initial_positions, dt, delta, NM)
        init = stacked(pos, vec_matrix, initial_positions)
        return init, init

    carry, all = jax.lax.scan(body, init, dts)
    return carry, all

# TO BE MODIFIEEEEEEEDDDD!!!!!!!!
def stacked(pos, vec_matrix, initial_positions):
    first = pos # Shape (10, 10, 10, 3)
    second = vec_matrix # Shape (10, 10, 10, 3, 3)
    third = initial_positions # Shape (10, 10, 10, 3)
    # Reshape tensors to match desired final shape
    first_exp = first[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]  # (10, 10, 10, 3, 1, 1, 1)
    second_exp = second[:, :, :, jnp.newaxis, :, :, jnp.newaxis]  # (10, 10, 10, 1, 3, 3, 1)
    third_exp = third[:, :, :, jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (10, 10, 10, 1, 1, 1, 3)
    # Broadcasting automatically aligns dimensions
    result = first_exp + second_exp + third_exp  # Final shape: (10, 10, 10, 3, 3, 3, 3)
    return result

def unstacked(init):
    # Extract `first`: Take the slice where second and third dimensions are at index 0
    first_recovered = init[..., 0, 0, 0]  # Shape (10, 10, 10, 3)

    # Extract `second`: Take the slice where first and third dimensions are at index 0
    second_recovered = init[..., :, :, 0, 0]  # Shape (10, 10, 10, 3, 3)

    # Extract `third`: Take the slice where first and second dimensions are at index 0
    third_recovered = init[..., 0, 0, 0, :]  # Shape (10, 10, 10, 3)
    return first_recovered, second_recovered, third_recovered

if __name__ == '__main__':
    # Test the functions
    # Define the parameters
    NM = 10
    delta = 0.1
    dt = 0.1
    key = random.PRNGKey(0)
    pos = random.uniform(key, shape=(NM, NM, NM, 3))
    vec = random.uniform(key, shape=(NM, NM, NM, 3))
    carr = random.uniform(key, shape=(NM, NM, NM))
    vec_matrix = random.uniform(key, shape=(NM, NM, NM, 3, 3))
    print('Test the function CFE')
    initial_positions = pos
    
    # Test the function VelocityUV
    #A = gradient_of_flow_at_a_point(pos, vec, carr, dt, delta, NM)
    B = integrate(step, pos, vec_matrix, initial_positions, dt, delta, NM)
    pos, vec_matrix, initial_positions = unstacked(B)
    print(pos.shape)

