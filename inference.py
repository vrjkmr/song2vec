# -*- coding: utf-8 -*-
import pickle
import numpy as np
from tensorflow.keras.models import Model, load_model

# load tokenizer
print("loading tokenizer...")
tokenizer = pickle.load(open("data/tokenizer.pkl", "rb"))

# load trained model
model = load_model("model")

# extract embeddings
target_embedding_layer = model.layers[2]
embedding_weights = target_embedding_layer.get_weights()[0]
print("embedding shape:", embedding_weights.shape)

# specify track name and top n
track_name = "john mayer - free fallin' - live at the nokia theatre" # specify track name
n = 10 # get top n most similar songs

# get track embedding
track_idx = tokenizer.word_index[track_name]
track_vector = embedding_weights[track_idx, :].reshape(1, -1)

# compute similarities against other tracks
similarities = np.dot(track_vector, embedding_weights.T) / (np.linalg.norm(track_vector) * np.linalg.norm(embedding_weights, axis=1))
similarities = similarities.reshape(-1)

# get most similar tracks
most_similar_idxs = np.argpartition(similarities, -n)[-n:]
most_similar_idxs = most_similar_idxs[np.argsort(similarities[most_similar_idxs])][::-1]

# print most similar items
print("top {} tracks most similar to '{}':".format(n, track_name))
for idx in most_similar_idxs:
    print("- ({:.3f}): '{}' ({})".format(similarities[idx], tokenizer.index_word[idx], idx))
