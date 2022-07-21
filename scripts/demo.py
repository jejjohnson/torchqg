import sys, os
from pyprojroot import here


# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(root))

# -----------------

import sys
import math
import tqdm
import xarray as xr
import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torchqg import to_spectral, to_physical, QgModel, MLdiv, Constant

# import workflow

plt.rcParams.update({'mathtext.fontset':'cm'})

# A framework for the evaluation of turbulence closures used in mesoscale ocean large-eddy simulations.
# Graham and Ringler (2013).  


# WIND STRESS FORCING

# UNITS FOR LAYER
def t_unit():
  return 1.2e6

def l_unit():
  return (504e4 / math.pi)

def Fs(i, sol, dt, t, grid):
    phi_x = math.pi * math.sin(1.2e-6 / t_unit()**(-1) * t)
    phi_y = math.pi * math.sin(1.2e-6 * math.pi / t_unit()**(-1) * t / 3)
    y = torch.cos(4 * grid.y + phi_y).view(grid.Ny, 1) - torch.cos(4 * grid.x + phi_x).view(1, grid.Nx)

    yh = to_spectral(y)
    K = torch.sqrt(grid.krsq)
    yh[K < 3.0] = 0
    yh[K > 5.0] = 0
    yh[0, 0] = 0

    e0 = 1.75e-18 / t_unit()**(-3)
    ei = 0.5 * grid.int_sq(yh) / (grid.Lx * grid.Ly)
    yh *= torch.sqrt(e0 / ei)
    return yh


#######################
# PARAMETERS
#######################

domain_factor = 1

Lx = 2 * math.pi * domain_factor
Ly = 2 * math.pi * domain_factor
Nx = 512
Ny = 512
scale = 4
Nxl = int(Nx / scale)
Nyl = int(Ny / scale)

iters = 10_000
steps = 1_500

t0=0.0
dt = 480 / t_unit() # 480s
B=0.0    # Planetary vorticity y-gradient
mu = 1.25e-8 / l_unit()**(-1) # Linear drag, 1.25e-8m^-1
nu = 352 / l_unit()**2 / t_unit()**(-1) # Viscosity coefficient, 22m^2s^-1 for the simulation (2048^2)
nv=1     # Hyperviscous order (nv=1 is viscosity)
eta = torch.zeros([Ny, Nx], dtype=torch.float64, requires_grad=True)  # Topographic PV
eta_m = torch.zeros([Nyl, Nxl], dtype=torch.float64, requires_grad=True)  # Topographic PV
source=Fs # Source term
sgs=Constant(c=0.0) # Subgrid-scale term (replace with yours)

# CREATE DATAARRAY
# High res model.
ds_grid = xr.Dataset(
    # {
    #     "r": (("steps", "Nx", "Ny"), fdns[:, 0, ...]),
    #     "q": (("steps", "Nx", "Ny"), fdns[:, 1, ...]),
    #     "p": (("steps", "Nx", "Ny"), fdns[:, 2, ...]),
    #     "u": (("steps", "Nx", "Ny"), fdns[:, 3, ...]),
    #     "v": (("steps", "Nx", "Ny"), fdns[:, 4, ...]),
    # },
    coords={
        "Nx": np.arange(-Lx/2, Lx/2, Lx/Nx),
        "Ny": np.arange(-Ly/2, Ly/2, Ly/Ny),
        "Nxl": np.arange(-Lx/2, Lx/2, Lx/Nxl),
        "Nyl": np.arange(-Ly/2, Ly/2, Ly/Nyl),
        "steps": np.arange(steps),
    },
)

# experimental attributes
ds_grid.attrs["mu"] = mu 
ds_grid.attrs["t0"] = t0 
ds_grid.attrs["B"] = B 
ds_grid.attrs["nu"] = nu 
ds_grid.attrs["dt"] = dt 
ds_grid.attrs["nv"] = nv 
ds_grid.attrs["forcing"] = "wind_stress" 
ds_grid.attrs["sgs"] = "constant" 
ds_grid.attrs["scale"] = 1 
ds_grid.attrs["iters"] = iters
ds_grid.attrs["steps"] = steps
ds_grid.attrs["num_pts_x"] = Nx
ds_grid.attrs["num_pts_y"] = Ny
ds_grid.attrs["x_minmax"] = Lx
ds_grid.attrs["y_minmax"] = Ly
ds_grid.attrs["step_size"] = Lx/Nx
ds_grid.attrs["step_size_l"] = Lx/Nxl

# High res model.
h = QgModel(
    name='\\mathcal{F}',
    Nx=Nx,
    Ny=Ny,
    Lx=Lx,
    Ly=Ly,
    dt=dt,
    t0=0.0,
    B=0.0,    # Planetary vorticity y-gradient
    mu=mu,    # Linear drag
    nu=nu,    # Viscosity coefficient
    nv=1,     # Hyperviscous order (nv=1 is viscosity)
    eta=eta,  # Topographic PV
    source=Fs,# Source term
    # sgs=Constant(c=0.0),
    
)


# Initial conditions.
h.init_randn(0.01, [3.0, 5.0])
# Set up spectral filter kernel.
h.kernel = h.grid.cutoff


# High res model.
m1 = QgModel(
    name='',
    Nx=Nxl,
    Ny=Nyl,
    Lx=Lx,
    Ly=Ly,
    dt=dt,
    t0=0.0,
    B=0.0,    # Planetary vorticity y-gradient
    mu=mu,    # Linear drag
    nu=nu,    # Viscosity coefficient
    nv=1,     # Hyperviscous order (nv=1 is viscosity)
    eta=eta_m,  # Topographic PV
    source=Fs,# Source term
    sgs=Constant(c=0.0),
    
)

# Initialize from DNS vorticity field.
m1.pde.sol = h.filter(m1.grid, scale, h.pde.sol)



dir = "output/"
name = "geo"
scale = scale
models = []
system = h

t0 = system.pde.cur.t
store_les = int(iters / steps)
store_dns = store_les * scale
fdns = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)
dns  = torch.zeros([steps, 4, Ny,  Nx ], dtype=torch.float64)

time = torch.zeros([steps])


if models:
    sgs_grid = models[-1].grid

# LES
les = {}
for m in models:
    les[m.name] = torch.zeros([steps, 5, Nyl, Nxl], dtype=torch.float64)



def visitor_dns(m, cur, it):
  # High res
  if it % store_dns == 0:
      i = int(it / store_dns)
      q, p, u, v = m.update()

      # Exact sgs
      if models:
          r = m.R(sgs_grid, scale)
          fdns[i, 0] = qg.to_physical(r)
          fdns[i, 1] = m.filter_physical(sgs_grid, scale, q).view(1, Nyl, Nxl)
          fdns[i, 2] = m.filter_physical(sgs_grid, scale, p).view(1, Nyl, Nxl)
          fdns[i, 3] = m.filter_physical(sgs_grid, scale, u).view(1, Nyl, Nxl)
          fdns[i, 4] = m.filter_physical(sgs_grid, scale, v).view(1, Nyl, Nxl)

      dns[i] = torch.stack((q, p, u, v))

      # step time
      time[i] = cur.t - t0
  return None



def enstropy(q):
    return 0.5 * torch.mean(q**2)

def energy(u, v):
    return 0.5 * torch.mean(u**2 + v**2)



with torch.no_grad():
    # high resolution model
    for it in tqdm.tqdm(range(iters * scale)):
      system.pde.step(system)
      visitor_dns(system, system.pde.cur, it)
      for m in models:
        if it % scale == 0:
          m.pde.step(m)
          visitor_dns(m, m.pde.cur, it / scale)



ds_grid = xr.Dataset(
    {
        # HIGH RES MODEL
        "q": (("steps", "Nx", "Ny"), dns[:, 0, ...]),
        "p": (("steps", "Nx", "Ny"), dns[:, 1, ...]),
        "u": (("steps", "Nx", "Ny"), dns[:, 2, ...]),
        "v": (("steps", "Nx", "Ny"), dns[:, 3, ...]),
        # # LOW RES MODEL
        # "rl": (("steps", "Nxl", "Nyl"), fdns[:, 0, ...]),
        # "ql": (("steps", "Nxl", "Nyl"), fdns[:, 1, ...]),
        # "pl": (("steps", "Nxl", "Nyl"), fdns[:, 2, ...]),
        # "ul": (("steps", "Nxl", "Nyl"), fdns[:, 3, ...]),
        # "vl": (("steps", "Nxl", "Nyl"), fdns[:, 4, ...]),
    },
    coords={
        "Nx": np.arange(-Lx/2, Lx/2, Lx/Nx),
        "Ny": np.arange(-Ly/2, Ly/2, Ly/Ny),
        # "Nxl": np.arange(-Lx/2, Lx/2, Lx/Nxl),
        # "Nyl": np.arange(-Ly/2, Ly/2, Ly/Nyl),
        "steps": np.arange(steps),
    },
)



# experimental attributes
ds_grid.attrs["mu"] = mu 
ds_grid.attrs["t0"] = t0 
ds_grid.attrs["B"] = B 
ds_grid.attrs["nu"] = nu 
ds_grid.attrs["dt"] = dt 
ds_grid.attrs["nv"] = nv 
ds_grid.attrs["forcing"] = "wind_stress" 
ds_grid.attrs["sgs"] = "constant" 
ds_grid.attrs["scale"] = scale
ds_grid.attrs["iters"] = iters
ds_grid.attrs["steps"] = steps
ds_grid.attrs["num_pts_x"] = Nx
ds_grid.attrs["num_pts_y"] = Ny
ds_grid.attrs["x_minmax"] = Lx
ds_grid.attrs["y_minmax"] = Ly
ds_grid.attrs["step_size"] = Lx/Nx


save_path = root.joinpath("data/cutout_res.nc")
ds_grid.to_zarr(save_path, mode="w")