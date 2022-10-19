

# !ls /mnt/meom/workdir/johnsonj/data

from pathlib import Path
import xarray as xr
import numpy as np

root = Path("/home/johnsonj/projects/torchqg")

data_name = root.joinpath("data/qgsim_forcing_128x128.zarr")
save_path = root.joinpath("data/qgsim_forcing_128x128.nc")
# save_path = root.joinpath("temp.nc")

ds_grid = xr.open_zarr(data_name)

def reformat_time(ds, dt):
    
    ds = ds - ds[0]
    ds /= dt
    
    return ds


def preprocessing(ds, noise: float=0.01, dt: float=1.0):
    
    # slice timesteps
    ds = ds.isel(steps=slice(500,511))
    
    # reformat time
    time_coords = ds.steps.values.astype(np.float64)

    time_coords = reformat_time(
        time_coords, dt
    )
    time_coords = time_coords.astype(np.float64)
    ds["steps"] = time_coords
    
    # add noise to observations
    rng = np.random.RandomState(123)

    ds["q_obs"] = ds["q"] + noise * rng.randn(*ds["q"].shape)
    ds["p_obs"] = ds["p"] + noise * rng.randn(*ds["p"].shape)
    
    return ds

ds_grid = preprocessing(ds_grid)

ds_grid.to_netcdf(save_path)