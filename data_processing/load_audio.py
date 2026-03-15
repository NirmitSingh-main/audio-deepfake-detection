import os
import librosa


def get_audio_files(dataset_path):
    """
    Scan dataset directory and collect all .flac files
    """

    audio_files = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                audio_files.append(os.path.join(root, file))

    return audio_files


def load_audio(file_path, sample_rate=22050):
    """
    Load audio file using librosa
    """

    audio, sr = librosa.load(file_path, sr=sample_rate)

    return audio, sr