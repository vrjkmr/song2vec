# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams
import pickle

# set hyperparameters
WINDOW_SIZE = 2
RANDOM_SEED = 42

# load dataframe
print("loading dataframe...")
filename = "data/spotify_playlists/combined_dataset.csv"
dataframe = pd.read_csv(filename, sep="\t", index_col=0)

# separate track names by playlist
unique_playlists = dataframe["playlist_name"].unique()
print("# unique playlists:", len(unique_playlists))
print("building list of tracks by playlist...")
tracks_by_playlist = []
for playlist in unique_playlists:
    tracks = dataframe[dataframe["playlist_name"] == playlist]["track_full_name"].tolist()
    tracks_by_playlist.append(tracks)

# tokenize track names
print("tokenizing tracks...")
tokenizer = Tokenizer(split=None)
tokenizer.fit_on_texts(tracks_by_playlist)
sequences = tokenizer.texts_to_sequences(tracks_by_playlist)

# extract vocabulary
vocabulary_size = len(tokenizer.word_index) + 1
print("vocabulary size (# tracks):", vocabulary_size)

# store tokenizer
print("saving tokenizer...")
pickle.dump(tokenizer, open("data/tokenizer.pkl", "wb"))

# generate skip-grams
print("generating negative sampling skip-gram pairs...")
skip_grams = [skipgrams(sequence, vocabulary_size, window_size=WINDOW_SIZE, seed=RANDOM_SEED) for sequence in sequences]

# store skip-grams
print("saving pairs...")
pickle.dump(skip_grams, open("data/skipgrams.pkl", "wb"))

# prepare training data
print("preparing training data...")
targets = np.array([], dtype="int32")
contexts = np.array([], dtype="int32")
y = np.array([], dtype="int32")

for i, skip_gram in enumerate(skip_grams):
    pairs = list(zip(*skip_gram[0]))
    if len(pairs) > 0:
        target_idxs = np.array(pairs[0], dtype="int32")
        context_idxs = np.array(pairs[1], dtype="int32")
        labels = np.array(skip_gram[1], dtype="int32")
        targets = np.concatenate((targets, target_idxs), axis=0)
        contexts = np.concatenate((contexts, context_idxs), axis=0)
        y = np.concatenate((y, labels), axis=0)

X = [targets, contexts]

# store dataset
print("saving training data...")
pickle.dump(X, open("data/X.pkl", "wb"))
pickle.dump(y, open("data/y.pkl", "wb"))
print("done.")