from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "ml-latest-small.zip"
    print("Downloading MovieLens...")
    urlretrieve(URL, archive)
    with ZipFile(archive) as zipped:
        for name in ("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv"):
            zipped.extract(name, DATA)
    archive.unlink()
    print(f"Ready: {DATA / 'ml-latest-small'}")


if __name__ == "__main__":
    main()

