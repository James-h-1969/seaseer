"""
Script to generate all the required data for training and evaluating SeaSeer
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
                "1940",
                "1941",
                "1942",
                "1943",
                "1944",
                "1945",
                "1946",
                "1947",
                "1948",
                "1949",
                "1950",
                "1951",
                "1952",
                "1953",
                "1954",
                "1955",
                "1956",
                "1957",
                "1958",
                "1959",
                "1960",
                "1961",
                "1962",
                "1963",
                "1964",
                "1965",
                "1966",
                "1967",
                "1968",
                "1969",
                "1970",
                "1971",
                "1972",
                "1973",
                "1974",
                "1975",
                "1976",
                "1977",
                "1978",
                "1979",
                "1980",
                "1981",
                "1982",
                "1983",
                "1984",
                "1985",
                "1986",
                "1987",
                "1988",
                "1989",
                "1990",
                "1991",
                "1992",
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
            "area": [28, -130, -55, 110],
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
        logging.info("NEED TO IMPLEMENT CMEMS")


class OMEGA3DDownloader(Downloader):
    def download(self, dataset: str) -> None:
        logging.info("NEED TO IMPLEMENT OMEGA")


def download_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaders_with_dataset: list[tuple[Downloader, str]] = [
        (ERA5Downloader("ERA5"), ""),
        (ERA5Downloader("ERA5"), ""),
        (ERA5Downloader("ERA5"), ""),
        (ERA5Downloader("ERA5"), ""),
        (NOAAOISSTDownloader("NOAA"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (OMEGA3DDownloader("OMEGA"), ""),
    ]

    threads = []
    for downloader, dataset in downloaders_with_dataset:
        thread = threading.Thread(target=downloader.download, args=(dataset,))
        threads.append(thread)
        print(f"Making a thread to download the {downloader.name} {dataset} dataset")

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(f"Downloading {len(threads)} datasets has finished.")


if __name__ == "__main__":
    download_data()
