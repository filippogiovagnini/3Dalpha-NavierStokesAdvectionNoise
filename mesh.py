import sys, os
import jax.numpy as jnp


def mesh_creation(nx,
                  ny,
                  nz,
                  xmin,
                  xmax,
                  ymin,
                  ymax,
                  zmin,
                  zmax):
    # ----------Mesh input (including edge)-----------------#
    nfx = nx + 1
    nfy = ny + 1
    nfz = nz + 1
    # -----------------derived parameters-------------------#
    dx = (xmax - xmin) / (nx)
    dy = (ymax - ymin) / (ny)
    dz = (zmax - zmin) / (nz)
    # ----------------Mesh Creation------------------#
    # xc is the vector of the cell centers
    # xf is the vector of the faces with first removed (boundary conditions)
    xc = jnp.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx) # assumes periodic in other code 
    yc = jnp.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ny)
    zc = jnp.linspace(zmin + 0.5 * dz, zmax - 0.5 * dz, nz)
    xf = jnp.linspace(xmin + dx, xmax, nfx - 1)
    yf = jnp.linspace(ymin + dy, ymax, nfy - 1)
    zf = jnp.linspace(zmin + dz, zmax, nfz - 1)
    # xv, yv = torch.meshgrid(xf, yf, sparse=False, indexing='xy')  # the v ertices.
    # xxf, xyf = jnp.meshgrid(xf, yc, indexing='xy')  # the x faces positions
    # yxf, yyf = jnp.meshgrid(xc, yf, indexing='xy')  # the y faces positions
    xxc, yyc, zzc = jnp.meshgrid(xc, yc, zc, indexing='xy')  # the c enters
    print("Mesh created")
    return xxc, yyc, zzc


def mesh_creation_2d(nx,
                  ny,
                  xmin,
                  xmax,
                  ymin,
                  ymax):
    # ----------Mesh input (including edge)-----------------#
    nfx = nx + 1
    nfy = ny + 1
    # -----------------derived parameters-------------------#
    dx = (xmax - xmin) / (nx)
    dy = (ymax - ymin) / (ny)
    # ----------------Mesh Creation------------------#
    # xc is the vector of the cell centers
    # xf is the vector of the faces with first removed (boundary conditions)
    xc = jnp.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, nx) # assumes periodic in other code 
    yc = jnp.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ny)
    xf = jnp.linspace(xmin + dx, xmax, nfx - 1)
    yf = jnp.linspace(ymin + dy, ymax, nfy - 1)
    # xv, yv = torch.meshgrid(xf, yf, sparse=False, indexing='xy')  # the v ertices.
    # xxf, xyf = jnp.meshgrid(xf, yc, indexing='xy')  # the x faces positions
    # yxf, yyf = jnp.meshgrid(xc, yf, indexing='xy')  # the y faces positions
    xxc, yyc = jnp.meshgrid(xc, yc, indexing='xy')  # the c enters
    arguments = [nx, ny, dx, dy, xc, yc, xxc, yyc]
    return arguments