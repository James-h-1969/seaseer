"""Data loading for SeaSeer.

Loads ERA5 heat flux, NOAA SST, and CMEMS ocean current NetCDF files,
aligns them to a common lat/lon grid and daily timestep, then serves
(input, target) pairs where input is the state at time t and target
is the state at time t+1.
"""

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path(__file__).parents[1] / "data"

# Variables per source that become input channels
ERA5_VARIABLES = [
    "mean_surface_direct_short_wave_radiation_flux",
    "mean_surface_downward_long_wave_radiation_flux",
    "mean_surface_latent_heat_flux",
    "mean_surface_sensible_heat_flux",
]

NOAA_VARIABLE = "sst"

CMEMS_DATASETS = {
    "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i": ["utotal", "vtotal"],
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m": ["mlotst", "tob"],
    "cmems_obs-mob_glo_phy-uvw_nrt_0.25deg_P7D-i": ["wo"],
}

# Target grid resolution (degrees) — all sources get interpolated here
TARGET_LAT = np.arange(-25, -10, 0.25)
TARGET_LON = np.arange(142, 154, 0.25)


def _load_era5() -> xr.Dataset | None:
    """Load and concatenate yearly ERA5 files."""
    files = sorted(DATA_DIR.glob("era5_*.nc"))
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine="by_coords")
    # Resample hourly data to daily means
    ds = ds.resample(time="1D").mean()
    return ds


def _load_noaa_sst() -> xr.DataArray | None:
    """Load and concatenate yearly NOAA OISST files."""
    files = sorted(DATA_DIR.glob("sst.day.mean.*.nc"))
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine="by_coords")
    sst = ds[NOAA_VARIABLE]
    # NOAA uses 0-360 longitude; convert to -180..180 if needed
    if float(sst.lon.max()) > 180:
        sst = sst.assign_coords(lon=(sst.lon + 180) % 360 - 180)
        sst = sst.sortby("lon")
    return sst


def _load_cmems() -> xr.Dataset | None:
    """Load CMEMS subset files."""
    copernicus_dir = DATA_DIR / "copernicus"
    files = sorted(copernicus_dir.glob("CMEMS-*.nc"))
    if not files:
        return None
    datasets = [xr.open_dataset(f) for f in files]
    return xr.merge(datasets)


def _regrid(da: xr.DataArray) -> xr.DataArray:
    """Interpolate a DataArray onto the common target grid."""
    return da.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")


def build_tensor() -> tuple[torch.Tensor, list[str]]:
    """Load all sources, regrid, and stack into a single tensor.

    Returns:
        tensor: Shape (T, C, H, W) with all channels aligned in time.
        channel_names: List of channel name strings.
    """
    arrays = []
    channel_names = []

    # ERA5 heat fluxes
    era5 = _load_era5()
    if era5 is not None:
        for var in ERA5_VARIABLES:
            if var in era5:
                arr = _regrid(era5[var].squeeze(drop=True))
                arrays.append(arr)
                channel_names.append(f"era5_{var}")

    # NOAA SST
    sst = _load_noaa_sst()
    if sst is not None:
        arr = _regrid(sst.squeeze(drop=True))
        arrays.append(arr)
        channel_names.append("noaa_sst")

    # CMEMS ocean variables
    cmems = _load_cmems()
    if cmems is not None:
        for variables in CMEMS_DATASETS.values():
            for var in variables:
                if var in cmems:
                    da = cmems[var]
                    # Average over depth if present
                    if "depth" in da.dims:
                        da = da.mean(dim="depth")
                    arr = _regrid(da.squeeze(drop=True))
                    arrays.append(arr)
                    channel_names.append(f"cmems_{var}")

    if not arrays:
        msg = f"No data files found in {DATA_DIR}"
        raise FileNotFoundError(msg)

    # Align all arrays to their shared time range
    aligned = xr.align(*arrays, join="inner")

    # Stack into (T, C, H, W)
    stacked = np.stack([a.values for a in aligned], axis=1)
    # Replace NaNs with 0 (land/missing)
    stacked = np.nan_to_num(stacked, nan=0.0)

    return torch.tensor(stacked, dtype=torch.float32), channel_names


class SeaSeerDataset(Dataset):
    """Dataset that yields (state_t, state_t+1) pairs.

    Args:
        data: Tensor of shape (T, C, H, W).
        lead_time: Number of timesteps ahead for the target.
    """

    def __init__(self, data: torch.Tensor, lead_time: int = 1):
        self.data = data
        self.lead_time = lead_time

    def __len__(self) -> int:
        return len(self.data) - self.lead_time

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx + self.lead_time]


def create_dataloader(
    batch_size: int = 8,
    lead_time: int = 1,
    train_split: float = 0.8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Build train and validation DataLoaders from downloaded data.

    Args:
        batch_size: Batch size.
        lead_time: How many days ahead to predict.
        train_split: Fraction of data for training.
        num_workers: DataLoader workers.

    Returns:
        train_loader, val_loader, channel_names
    """
    data, channel_names = build_tensor()

    n_train = int(len(data) * train_split)
    train_data = data[:n_train]
    val_data = data[n_train:]

    train_loader = DataLoader(
        SeaSeerDataset(train_data, lead_time),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        SeaSeerDataset(val_data, lead_time),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, channel_names
