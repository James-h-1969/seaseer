from pathlib import Path
from unittest.mock import MagicMock, patch

from helpers.scripts.generate_data import (
    CMEMSDownloader,
    ERA5Downloader,
    NOAAOISSTDownloader,
    OMEGA3DDownloader,
    download_data,
)


class TestERA5Downloader:
    def test_name(self):
        d = ERA5Downloader("ERA5")
        assert d.name == "ERA5"

    @patch("helpers.scripts.generate_data.cdsapi.Client")
    def test_download_calls_client(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        d = ERA5Downloader("ERA5")
        d.download("reanalysis-era5-single-levels")
        mock_client.retrieve.assert_called_once()
        args = mock_client.retrieve.call_args
        assert args[0][0] == "reanalysis-era5-single-levels"
        assert "variable" in args[0][1]
        mock_client.retrieve.return_value.download.assert_called_once()

    @patch("helpers.scripts.generate_data.cdsapi.Client")
    def test_request_contains_expected_keys(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        ERA5Downloader("ERA5").download("ds")
        request = mock_client.retrieve.call_args[0][1]
        for key in ["product_type", "variable", "year", "month", "day", "time", "area"]:
            assert key in request

    @patch("helpers.scripts.generate_data.cdsapi.Client")
    def test_area_bounds(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        ERA5Downloader("ERA5").download("ds")
        area = mock_client.retrieve.call_args[0][1]["area"]
        assert area == [28, -130, -55, 110]


class TestNOAAOISSTDownloader:
    def test_name(self):
        d = NOAAOISSTDownloader("NOAA")
        assert d.name == "NOAA"

    @patch("helpers.scripts.generate_data.OUTPUT_DIR", Path("/tmp/test_raw"))
    @patch("helpers.scripts.generate_data.requests.get")
    def test_download_fetches_index_page(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "<html><body></body></html>"
        mock_get.return_value = mock_response
        NOAAOISSTDownloader("NOAA").download("some/path")
        first_url = mock_get.call_args_list[0][0][0]
        assert first_url == "https://noaadata.apps.nsidc.org/NOAA/some/path/"

    @patch("helpers.scripts.generate_data.OUTPUT_DIR", Path("/tmp/test_raw"))
    @patch("builtins.open", new_callable=MagicMock)
    @patch("helpers.scripts.generate_data.requests.get")
    def test_download_saves_linked_files(self, mock_get, mock_open):
        index_html = """
        <html><body><a href="parent">..</a><a href="file1.nc">file1.nc</a></body></html>
        """
        index_resp = MagicMock()
        index_resp.text = index_html
        file_resp = MagicMock()
        file_resp.content = b"data"
        mock_get.side_effect = [index_resp, file_resp]

        with patch.object(Path, "mkdir"):
            NOAAOISSTDownloader("NOAA").download("dataset")

        assert mock_get.call_count == 2

    @patch("helpers.scripts.generate_data.requests.get")
    def test_download_raises_on_http_error(self, mock_get):
        mock_get.return_value.raise_for_status.side_effect = Exception("404")
        try:
            NOAAOISSTDownloader("NOAA").download("bad")
        except Exception as e:
            assert "404" in str(e)


class TestCMEMSDownloader:
    def test_name(self):
        d = CMEMSDownloader("CMEMS")
        assert d.name == "CMEMS"

    def test_download_logs_not_implemented(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            CMEMSDownloader("CMEMS").download("ds")
        assert "NEED TO IMPLEMENT CMEMS" in caplog.text


class TestOMEGA3DDownloader:
    def test_name(self):
        d = OMEGA3DDownloader("OMEGA")
        assert d.name == "OMEGA"

    def test_download_logs_not_implemented(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            OMEGA3DDownloader("OMEGA").download("ds")
        assert "NEED TO IMPLEMENT OMEGA" in caplog.text


class TestDownloadData:
    @patch("helpers.scripts.generate_data.threading.Thread")
    @patch("helpers.scripts.generate_data.OUTPUT_DIR", Path("/tmp/test_raw"))
    def test_creates_threads_for_each_downloader(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        with patch.object(Path, "mkdir"):
            download_data()
        assert mock_thread_cls.call_count == 10

    @patch("helpers.scripts.generate_data.threading.Thread")
    @patch("helpers.scripts.generate_data.OUTPUT_DIR", Path("/tmp/test_raw"))
    def test_starts_and_joins_all_threads(self, mock_thread_cls):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        with patch.object(Path, "mkdir"):
            download_data()
        assert mock_thread.start.call_count == 10
        assert mock_thread.join.call_count == 10
