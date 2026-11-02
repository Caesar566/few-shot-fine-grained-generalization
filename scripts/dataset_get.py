"""Download a CUB copy into the project-local cache only when CUB is absent."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    official_layout = PROJECT_ROOT / "data/raw/CUB_200_2011"
    if all((official_layout / name).exists() for name in
           ("classes.txt", "images.txt", "image_class_labels.txt", "images")):
        print(f"CUB already present: {official_layout}")
        return official_layout
    cache = PROJECT_ROOT / "data/cache/kagglehub"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(cache)
    import kagglehub  # Cache must be set before import.

    path = kagglehub.dataset_download("wenewone/cub2002011")
    print(f"CUB KaggleHub copy cached at: {path}")
    print("Check its layout before using it as data/raw/CUB_200_2011")
    return path


if __name__ == "__main__":
    main()
