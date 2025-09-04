
from datasets import Dataset, Audio
import json

def load_dataset():
    data_list = []
    with open("metadata.json", "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line.strip()))

    dataset = Dataset.from_list(data_list)
    # Cast to Audio feature for automatic loading
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    return dataset
