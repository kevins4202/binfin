import os
import csv
import datasets
import random

# You can add citations and links if you have them
_CITATION = ""
_DESCRIPTION = "Dataset of audio files labeled as 'fake' or 'genuine'."
_HOMEPAGE = (
    "https://huggingface.co/datasets/kevins4202/binfin-project"  # Link to your dataset
)
_LICENSE = "MIT"

# Define the URLs for your data files on the Hub
_URLS = {
    "meta": "https://huggingface.co/datasets/kevins4202/binfin-project/resolve/main/meta.txt",
    "data": "https://huggingface.co/datasets/kevins4202/binfin-project/resolve/main/data.zip",
}


class AudioDeepfakeDataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "label": datasets.ClassLabel(names=["0", "1"]),
                    "filename": datasets.Value("string"),
                }
            ),
            supervised_keys=("audio", "label"),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager downloads and caches the files for you
        # It returns the path to the cached file
        meta_path = dl_manager.download(_URLS["meta"])

        # download_and_extract handles both downloading and unzipping
        # It returns the path to the extracted directory
        data_archive = dl_manager.download_and_extract(_URLS["data"])

        # The audio files are now inside a 'data' folder within the cached directory
        audio_dir = os.path.join(data_archive, "data")

        # Read meta.csv from the cached path
        # with open(meta_path, "r", encoding="utf-8") as f:
        with open(meta_path, "r") as f:
            lines = f.readlines()
            meta_csv = [line.strip().split("\t") for line in lines]
            examples = [
                {
                    "filename": row[0],
                    "label": row[1].strip(),
                }
                for row in meta_csv
            ]

        # Shuffle and split
        random.seed(42)
        random.shuffle(examples)
        n = len(examples)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        # Pass the correct audio directory path to the generators
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"examples": examples[:train_end], "audio_dir": audio_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "examples": examples[train_end:val_end],
                    "audio_dir": audio_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"examples": examples[val_end:], "audio_dir": audio_dir},
            ),
        ]

    def _generate_examples(self, examples, audio_dir):
        # audio_dir is the correct path passed from _split_generators
        for idx, ex in enumerate(examples):
            audio_path = os.path.join(audio_dir, ex["filename"])

            # This check is good practice
            if not os.path.exists(audio_path):
                print(f"Missing file: {audio_path}")
                continue

            yield idx, {
                "audio": audio_path,
                "label": ex["label"],
                "filename": ex["filename"],
            }
