from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_ID = (
    "gaurav41/"
    "voxceleb1-audio-wav-files-for-india-celebrity"
)

OUTPUT_DIRECTORY = Path("data/raw/voxceleb1_indian")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {DATASET_ID}")
    print(f"Destination: {OUTPUT_DIRECTORY.resolve()}")

    api.dataset_download_files(
        dataset=DATASET_ID,
        path=str(OUTPUT_DIRECTORY),
        unzip=True,
        quiet=False,
    )

    wav_files = list(OUTPUT_DIRECTORY.rglob("*.wav"))

    print("\nDownload completed.")
    print(f"WAV files found: {len(wav_files)}")
    print(f"Dataset location: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    main()