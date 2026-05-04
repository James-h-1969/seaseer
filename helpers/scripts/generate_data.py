"""
Script to generate all the required data for training and evaluating SeaSeer.

To note: all data is stored in NetCDF4 format, a standard self
describing scientific data formatter. All data is from 1993-2018.
"""

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cdsapi
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(threadName)-10s] %(levelname)s: %(message)s",
)

OUTPUT_DIR = Path(__file__).parent / "raw"

BOUNDARY_COORDS = {"north": -10, "south": -25, "east": 154, "west": 142}


class Downloader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def download(self, dataset: str) -> None:
        pass


class ERA5Downloader(Downloader):
    def download(self, dataset: str) -> None:
        logging.info("Downloading ERA5")
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": [
                "reanalysis",
            ],
            "variable": [dataset],
            "year": [
                "1993",
                "1994",
                "1995",
                "1996",
                "1997",
                "1998",
                "1999",
                "2000",
                "2001",
                "2002",
                "2003",
                "2004",
                "2005",
                "2006",
                "2007",
                "2008",
                "2009",
                "2010",
                "2011",
                "2012",
                "2013",
                "2014",
                "2015",
                "2016",
                "2017",
                "2018",
            ],
            "month": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
            ],
            "day": [
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
            ],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [
                BOUNDARY_COORDS["north"],
                BOUNDARY_COORDS["west"],
                BOUNDARY_COORDS["south"],
                BOUNDARY_COORDS["east"],
            ],
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download()


class NOAAOISSTDownloader(Downloader):
    BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/"

    def download(self, dataset: str) -> None:
        logging.info("Downloading NOAA dataset")
        archive_url = self.BASE_URL + dataset
        if not archive_url.endswith("/"):
            archive_url += "/"
        r = requests.get(archive_url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        output_dir = OUTPUT_DIR / "noaa" / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for link in soup.find_all("a")[1:]:
            href = link["href"]
            file_r = requests.get(archive_url + href)
            file_r.raise_for_status()
            with open(output_dir / href, "wb") as f:
                f.write(file_r.content)


class CMEMSDownloader(Downloader):
    def download(self, dataset: str) -> None:
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
            output_directory="data/copernicus-data",
        )
        print("Finished downloading the CMEMs data")


def download_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaders_with_dataset: list[tuple[Downloader, Any]] = [
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
        # (NOAAOISSTDownloader("NOAA"), "") # Sea Surface Temperature
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
