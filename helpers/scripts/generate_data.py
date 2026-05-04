"""
Script to generate all the required data for training and evaluating SeaSeer.

To note: all data is stored in NetCDF4 format, a standard self
describing scientific data formatter. All data is from 1993-2018.
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import cdsapi
import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(threadName)-10s] %(levelname)s: %(message)s",
)

OUTPUT_DIR = Path(__file__).parents[2] / "data"

BOUNDARY_COORDS = {"north": -10, "south": -25, "east": 154, "west": 142}


T = TypeVar("T")


class Downloader(ABC, Generic[T]):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def download(self, dataset: T) -> None:
        pass


class ERA5Downloader(Downloader[str]):
    START_YEAR = 1993
    END_YEAR = 2018

    def download(self, dataset: str) -> None:
        logging.info("Downloading ERA5 variable: %s", dataset)
        era5_dataset = "reanalysis-era5-single-levels"
        client = cdsapi.Client()

        for year in range(self.START_YEAR, self.END_YEAR + 1):
            logging.info("ERA5 %s: requesting year %d", dataset, year)
            request = {
                "product_type": ["reanalysis"],
                "variable": [dataset],
                "year": [str(year)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [
                    BOUNDARY_COORDS["north"],
                    BOUNDARY_COORDS["west"],
                    BOUNDARY_COORDS["south"],
                    BOUNDARY_COORDS["east"],
                ],
            }
            target = OUTPUT_DIR / f"era5_{dataset}_{year}.nc"
            client.retrieve(era5_dataset, request).download(target)


class NOAAOISSTDownloader(Downloader[str]):
    BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/"

    def download(self, dataset: str) -> None:
        logging.info("Downloading NOAA dataset")
        year = 1981

        while year < 2019:
            filename = f"sst.day.mean.{year}.nc"
            url = f"https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/{filename}"

            print(f"Starting download for {filename}...")

            # stream=True prevents loading the entire 700MB file into memory at once
            with requests.get(url, stream=True) as response:
                # Check if the URL actually exists (status code 200)
                response.raise_for_status()

                with open(filename, "wb") as file:
                    # Download in 8KB chunks
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            print(f"Download complete! Saved as {os.path.abspath(filename)}")

            year += 1


class CMEMSDownloader(Downloader[list]):
    def download(self, dataset: list) -> None:
        import copernicusmarine

        copernicusmarine.subset(
            dataset_id=dataset[0],
            variables=dataset[1],
            minimum_longitude=BOUNDARY_COORDS["west"],
            maximum_longitude=BOUNDARY_COORDS["east"],
            minimum_latitude=BOUNDARY_COORDS["south"],
            maximum_latitude=BOUNDARY_COORDS["north"],
            start_datetime="2022-01-01",
            end_datetime="2026-01-31",
            minimum_depth=0,
            maximum_depth=10,
            output_filename=f"CMEMS-{dataset[0]}.nc",
            output_directory=str(OUTPUT_DIR / "copernicus"),
        )
        print("Finished downloading the CMEMs data")


def download_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaders_with_dataset: list[tuple[Downloader, str | list]] = [
        (
            ERA5Downloader("ERA5"),
            "mean_surface_direct_short_wave_radiation_flux",
        ),  # Net longwave Radiation
        (
            ERA5Downloader("ERA5"),
            "mean_surface_downward_long_wave_radiation_flux",
        ),  # Net shortwave Radiation
        (ERA5Downloader("ERA5"), "mean_surface_latent_heat_flux"),  # Sensible heat flux
        (ERA5Downloader("ERA5"), "mean_surface_sensible_heat_flux"),  # Latent heat flux
        (NOAAOISSTDownloader("NOAA"), ""),  # Sea Surface Temperature
        (
            CMEMSDownloader("CMEMS"),
            ["cmems_mod_glo_phy_anfc_merged-uv_PT1H-i", ["utotal", "vtotal"]],
        ),  # Meridional, Zonal Ocean Current
        (
            CMEMSDownloader("CMEMS"),
            ["cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ["mlotst", "tob"]],
        ),  # Mixed layer depth, Temperature at bottom of floor
        (
            CMEMSDownloader("CMEMS (OMEGA3D)"),
            ["cmems_obs-mob_glo_phy-uvw_nrt_0.25deg_P7D-i", ["wo"]],
        ),  # Vertical Ocean Current
    ]

    threads = []
    for downloader, dataset in downloaders_with_dataset:
        thread = threading.Thread(target=downloader.download, args=(dataset,))
        threads.append(thread)
        print(f"Making a thread for the {downloader.name} {dataset} dataset")

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    download_data()
