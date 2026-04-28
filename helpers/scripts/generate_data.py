"""
Script to generate all the required data for training and evaluating SeaSeer.

To note: all data is stored in NetCDF4 format, a standard self
describing scientific data formatter. All data is from 1993-2018.
"""

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import cdsapi
import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(threadName)-10s] %(levelname)s: %(message)s",
)

OUTPUT_DIR = Path(__file__).parent / "raw"

BOUNDARY_COORDS = {"north": 20, "south": 50, "east": 110, "west": 180}


class Downloader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def download(self, dataset: str) -> None:
        pass


class ERA5Downloader(Downloader):
    def download(self, dataset: str) -> None:
        logging.info("Downloading ERA5")
        request: dict = {
            "product_type": ["reanalysis"],
            "variable": ["surface_latent_heat_flux"],
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
                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
                "2025",
                "2026",
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
            "time": ["00:00", "12:00"],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [
                BOUNDARY_COORDS["north"],
                -BOUNDARY_COORDS["east"],
                -BOUNDARY_COORDS["south"],
                BOUNDARY_COORDS["west"],
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
            # TODO: decide on boundary conditions
            # minimum_longitude=BOUNDARY_COORDS["east"],
            # maximum_longitude=-BOUNDARY_COORDS["west"],
            # minimum_latitude=-BOUNDARY_COORDS["south"],
            # maximum_latitude=BOUNDARY_COORDS["north"],
            start_datetime="2022-01-01",
            end_datetime="2026-01-31",
            minimum_depth=0,
            maximum_depth=10,
            output_filename=f"CMEMS-{dataset[0]}.nc",
            output_directory="data/copernicus-data",
        )
        print("Finished downloading the CMEMs data")


class OMEGA3DDownloader(Downloader):
    def download(self, dataset: str) -> None:
        logging.info("NEED TO IMPLEMENT OMEGA")


def download_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaders_with_dataset: list[tuple[Downloader, str]] = [
        # (ERA5Downloader("ERA5"), ""), # Net longwave Radiation
        # (ERA5Downloader("ERA5"), ""), # Net shortwave Radiation
        # (ERA5Downloader("ERA5"), ""), # Sensible heat flux
        # (ERA5Downloader("ERA5"), ""), # Latent heat flux
        # (NOAAOISSTDownloader("NOAA"), "") # Sea Surface Temperature
        (
            CMEMSDownloader("CMEMS"),
            ["cmems_mod_glo_phy_anfc_merged-uv_PT1H-i", ["utotal", "vtotal"]],
        ),  # Meridional, Zonal Ocean Current
        (
            CMEMSDownloader("CMEMS"),
            ["cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ["mlotst", "tob"]],
        ),  # Mixed layer depth, Temperature at bottom of floor
        # (OMEGA3DDownloader("OMEGA"), ""), # Mixed layer depth
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
