import os
import numpy as np
from data_processing.feature_extraction import extract_mfcc


def create_feature_dataset(dataset_path):

    X = []
    y = []

    train_audio_path = os.path.join(
        dataset_path,
        "LA",
        "LA",
        "ASVspoof2019_LA_train",
        "flac"
    )

    protocol_file = os.path.join(
        dataset_path,
        "LA",
        "LA",
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt"
    )

    labels = {}

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            label = parts[4]

            labels[file_id] = 1 if label == "spoof" else 0

    for file_name in os.listdir(train_audio_path):

        if not file_name.endswith(".flac"):
            continue

        file_id = file_name.replace(".flac", "")

        if file_id not in labels:
            continue

        file_path = os.path.join(train_audio_path, file_name)

        mfcc = extract_mfcc(file_path)

        X.append(mfcc)
        y.append(labels[file_id])

    X = np.array(X)
    y = np.array(y)

    return X, y