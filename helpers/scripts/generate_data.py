"""
Script to generate all the required data for training and evaluating SeaSeer
"""

import threading
from abc import ABC, abstractmethod
from pathlib import Path

import cdsapi
import requests
from bs4 import BeautifulSoup

OUTPUT_DIR = Path(__file__).parent / "raw"


class Downloader(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def download(self, dataset: str) -> None:
        pass


class ERA5Downloader(Downloader):
    def download(self, dataset: str) -> None:
        request: dict = {}  # TODO: fill in ERA5 request params
        client = cdsapi.Client()
        client.retrieve(dataset, request).download()


class NOAAOISSTDownloader(Downloader):
    BASE_URL = "https://noaadata.apps.nsidc.org/NOAA/"

    def __init__(self):
        super().__init__("NOAA")

    def download(self, dataset: str) -> None:
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
        raise NotImplementedError


class OMEGA3DDownloader(Downloader):
    def __init__(self):
        super().__init__("OMEGA")

    def download(self, dataset: str) -> None:
        raise NotImplementedError


def download_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaders_with_dataset: list[tuple[Downloader, str]] = [
        (ERA5Downloader("ERA5"), ""),  # TODO: dataset names
        (ERA5Downloader("ERA5"), ""),
        (ERA5Downloader("ERA5"), ""),
        (ERA5Downloader("ERA5"), ""),
        (NOAAOISSTDownloader(), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (CMEMSDownloader("CMEMS"), ""),
        (OMEGA3DDownloader(), ""),
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
