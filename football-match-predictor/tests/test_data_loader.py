"""Tests for src.data_loader. These run offline: no requests reach the website."""

import pytest

from src import data_loader
from src.data_loader import (
    SeasonFile,
    download_file,
    parse_season_links,
    season_label,
    season_start_year,
)

SAMPLE_HTML = """
<a href="mmz4281/2526/I1.csv">Serie A</a>
<a href="mmz4281/2526/I2.csv">Serie B</a>
<a href="mmz4281/9900/I1.csv">Serie A</a>
<a href="mmz4281/9394/I1.csv">Serie A</a>
<a href="mmz4281/9394/I1.csv">Serie A</a>
<a href="italym.php">Not a CSV</a>
"""


@pytest.mark.parametrize(
    "code, year, label",
    [
        ("9394", 1993, "1993-94"),
        ("9900", 1999, "1999-00"),
        ("0001", 2000, "2000-01"),
        ("2425", 2024, "2024-25"),
    ],
)
def test_season_code_conversion(code, year, label):
    assert season_start_year(code) == year
    assert season_label(code) == label


def test_invalid_season_code_raises():
    with pytest.raises(ValueError):
        season_start_year("24-25")


def test_parse_season_links_keeps_only_serie_a_sorted_and_unique():
    files = parse_season_links(SAMPLE_HTML)

    assert [f.season_code for f in files] == ["9394", "9900", "2526"]
    assert all("/I1.csv" in f.url for f in files)  # Serie B link ignored
    assert files[0].url == "https://www.football-data.co.uk/mmz4281/9394/I1.csv"
    assert files[0].filename == "I1_9394.csv"


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, content: bytes = b"Div,Date,HomeTeam\nI1,01/01/24,Inter\n"):
        self.content = content
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return FakeResponse(self.content)


def make_file():
    return SeasonFile("2425", "https://example.test/mmz4281/2425/I1.csv")


def test_download_file_saves_csv(tmp_path):
    session = FakeSession()

    path, downloaded = download_file(make_file(), tmp_path, session)

    assert downloaded
    assert path == tmp_path / "I1_2425.csv"
    assert path.read_bytes() == session.content


def test_download_file_skips_existing_unless_forced(tmp_path):
    session = FakeSession()
    download_file(make_file(), tmp_path, session)

    _, downloaded = download_file(make_file(), tmp_path, session)
    assert not downloaded
    assert session.calls == 1

    _, downloaded = download_file(make_file(), tmp_path, session, force=True)
    assert downloaded
    assert session.calls == 2


def test_download_file_rejects_empty_response(tmp_path):
    with pytest.raises(RuntimeError):
        download_file(make_file(), tmp_path, FakeSession(content=b"  \n"))


@pytest.fixture
def fake_site(monkeypatch):
    """Serve SAMPLE_HTML's links and count download requests."""
    files = parse_season_links(SAMPLE_HTML)
    monkeypatch.setattr(data_loader, "fetch_season_links", lambda session: files)
    session = FakeSession()
    monkeypatch.setattr(data_loader.requests, "Session", lambda: session)
    return session


def test_download_seasons_refreshes_only_latest_season(tmp_path, fake_site):
    data_loader.download_seasons(dest_dir=tmp_path, delay=0)
    assert fake_site.calls == 3  # three Serie A seasons in the sample

    data_loader.download_seasons(dest_dir=tmp_path, delay=0)
    assert fake_site.calls == 4  # second run only re-fetches 2025-26


def test_download_seasons_from_season_filter(tmp_path, fake_site):
    paths = data_loader.download_seasons(from_season=1999, dest_dir=tmp_path, delay=0)

    assert [p.name for p in paths] == ["I1_9900.csv", "I1_2526.csv"]
