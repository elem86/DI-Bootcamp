"""Download historical Serie A results from Football-Data.co.uk.

The Italy page (https://www.football-data.co.uk/italym.php) links one CSV per
season, e.g. ``mmz4281/2425/I1.csv`` for Serie A 2024/25 ("I1" is
Football-Data's code for Serie A). This module scrapes those links and saves
each season to::

    data/raw/I1_<season code>.csv      e.g. data/raw/I1_2425.csv

Usage (from the project folder)::

    python -m src.data_loader                      # every Serie A season
    python -m src.data_loader --from-season 2005   # 2005/06 onwards only
    python -m src.data_loader --force              # re-download everything
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

import requests

from src.config import RAW_DATA_DIR

BASE_URL = "https://www.football-data.co.uk/"
ITALY_PAGE_URL = urljoin(BASE_URL, "italym.php")
SERIE_A_CODE = "I1"

# Identify the client honestly and keep request volume low: the site is a
# free resource, so we pause between downloads.
HEADERS = {"User-Agent": "football-match-predictor (educational project)"}
REQUEST_TIMEOUT = 30  # seconds
DELAY_BETWEEN_DOWNLOADS = 1.0  # seconds

# Matches Serie A links such as "mmz4281/2425/I1.csv" (Serie B "I2" is ignored).
CSV_LINK_PATTERN = re.compile(
    rf'href="(mmz4281/(\d{{4}})/{SERIE_A_CODE}\.csv)"', re.IGNORECASE
)


@dataclass(frozen=True)
class SeasonFile:
    """One downloadable Serie A season CSV."""

    season_code: str  # Football-Data's 4-digit code, e.g. "2425"
    url: str

    @property
    def start_year(self) -> int:
        return season_start_year(self.season_code)

    @property
    def season_label(self) -> str:
        return season_label(self.season_code)

    @property
    def filename(self) -> str:
        return f"{SERIE_A_CODE}_{self.season_code}.csv"


def season_start_year(season_code: str) -> int:
    """Convert a season code to the year the season started.

    "9394" -> 1993, "9900" -> 1999, "0001" -> 2000, "2425" -> 2024.
    """
    if not re.fullmatch(r"\d{4}", season_code):
        raise ValueError(f"Invalid season code: {season_code!r}")
    first_two = int(season_code[:2])
    # Football-Data's archive starts in 1993, so 90-99 can only mean the 1990s.
    return 1900 + first_two if first_two >= 90 else 2000 + first_two


def season_label(season_code: str) -> str:
    """Human-readable season name: "2425" -> "2024-25"."""
    start = season_start_year(season_code)
    return f"{start}-{str(start + 1)[-2:]}"


def parse_season_links(html: str, base_url: str = BASE_URL) -> list[SeasonFile]:
    """Extract every Serie A season CSV link from the Italy page HTML.

    Returns files sorted oldest season first, with duplicates removed.
    """
    files = {
        code: SeasonFile(season_code=code, url=urljoin(base_url, href))
        for href, code in CSV_LINK_PATTERN.findall(html)
    }
    return sorted(files.values(), key=lambda f: f.start_year)


def fetch_season_links(session: requests.Session | None = None) -> list[SeasonFile]:
    """Download the Italy page and return all Serie A season CSV links on it."""
    session = session or requests.Session()
    response = session.get(ITALY_PAGE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    links = parse_season_links(response.text)
    if not links:
        raise RuntimeError(
            f"No Serie A CSV links found on {ITALY_PAGE_URL}; "
            "the page layout may have changed."
        )
    return links


def download_file(
    season_file: SeasonFile,
    dest_dir: Path,
    session: requests.Session,
    force: bool = False,
) -> tuple[Path, bool]:
    """Download one season CSV into ``dest_dir``.

    Returns the file path and whether a download actually happened
    (False when the file already existed and ``force`` was not set).
    """
    target = dest_dir / season_file.filename
    if target.exists() and not force:
        return target, False

    response = session.get(season_file.url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    if not response.content.strip():
        raise RuntimeError(f"Empty file returned for {season_file.url}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Write raw bytes: some older seasons are not UTF-8, and decoding is the
    # job of the import/standardisation step, not the downloader.
    tmp = target.with_suffix(".csv.part")
    tmp.write_bytes(response.content)
    tmp.replace(target)
    return target, True


def download_seasons(
    from_season: int | None = None,
    dest_dir: Path = RAW_DATA_DIR,
    force: bool = False,
    refresh_latest: bool = True,
    delay: float = DELAY_BETWEEN_DOWNLOADS,
) -> list[Path]:
    """Download Serie A season CSVs from Football-Data.co.uk.

    Args:
        from_season: Only seasons starting in this year or later.
        dest_dir: Folder for raw files.
        force: Re-download files that already exist.
        refresh_latest: Always re-download the most recent season, because
            the in-progress season's file is updated after every matchday.
        delay: Seconds to wait between downloads.

    Returns:
        Paths of all selected files (downloaded or already present).
    """
    session = requests.Session()
    files = [
        f
        for f in fetch_season_links(session)
        if from_season is None or f.start_year >= from_season
    ]
    if not files:
        print("No seasons matched the filters.")
        return []

    latest_year = max(f.start_year for f in files)
    paths = []
    for season_file in files:
        refresh = force or (refresh_latest and season_file.start_year == latest_year)
        path, downloaded = download_file(season_file, dest_dir, session, force=refresh)
        status = "downloaded" if downloaded else "already present"
        print(f"Serie A {season_file.season_label}: {status} ({path.name})")
        paths.append(path)
        if downloaded:
            time.sleep(delay)

    print(f"\n{len(paths)} Serie A season files available in {dest_dir}")
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download Serie A results from Football-Data.co.uk."
    )
    parser.add_argument(
        "--from-season",
        type=int,
        default=None,
        help="Earliest season start year to download, e.g. 2005 for 2005/06.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that already exist."
    )
    args = parser.parse_args(argv)
    download_seasons(args.from_season, force=args.force)


if __name__ == "__main__":
    main()
