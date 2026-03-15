

# import librosa
# import numpy as np

# def extract_mfcc(file_path, n_mfcc=40):

#     audio, sr = librosa.load(file_path, sr=22050)

#     mfcc = librosa.feature.mfcc(
#         y=audio,
#         sr=sr,
#         n_mfcc=n_mfcc
#     )

#     mfcc_mean = np.mean(mfcc.T, axis=0)

#     return mfcc_mean


# # -------- TEST PART --------

# file = "DATASET/LA/LA/ASVspoof2019_LA_train/flac/LA_T_1000137.flac"

# features = extract_mfcc(file)

# print("MFCC vector length:", len(features))
# print(features)





from data_processing.create_dataset import create_feature_dataset

X, y = create_feature_dataset("DATASET")

print("Feature dataset shape:", X.shape)
print("Labels shape:", y.shape)