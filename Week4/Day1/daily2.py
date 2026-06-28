import requests
import time


def measure_page_load(url: str, timeout: int = 30) -> float:
    """Return the time in seconds it takes to get a full response from `url`."""
    start = time.perf_counter()
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return time.perf_counter() - start


def test_sites(sites=None):
    if sites is None:
        sites = [
            "https://www.google.com",
            "https://www.ynet.co.il",
            "https://www.imdb.com",
            "https://www.python.org",
        ]

    results = {}
    for url in sites:
        try:
            results[url] = measure_page_load(url)
        except Exception as error:
            results[url] = f"ERROR: {error}"
    return results


def main():
    print("Page load timing tester")
    print("Testing multiple sites...")
    results = test_sites()
    for url, value in results.items():
        if isinstance(value, float):
            print(f"{url}: {value:.3f} seconds")
        else:
            print(f"{url}: {value}")


if __name__ == "__main__":
    main()
