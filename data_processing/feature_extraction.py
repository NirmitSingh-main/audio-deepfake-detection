import librosa
import numpy as np


def extract_mfcc(file_path, n_mfcc=40):
    """
    Extract MFCC features from audio file
    """

    audio, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    # take mean across time axis
    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean