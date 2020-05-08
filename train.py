# -*- coding: utf-8 -*-
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense

# set hyperparameters
RANDOM_SEED = 42
EMBEDDING_DIM = 100
NUM_EPOCHS = 20

# load tokenizer
print("loading tokenizer...")
tokenizer = pickle.load(open("data/tokenizer.pkl", "rb"))
track2idx = tokenizer.word_index
vocabulary_size = len(track2idx) + 1

# load training data
print("loading training data...")
X = pickle.load(open("data/X.pkl", "rb"))
y = pickle.load(open("data/y.pkl", "rb"))

# set seed
tensorflow.random.set_seed(RANDOM_SEED)

# build model architecture
print("building model...")
target_inp = Input(shape=(1,))
target_emb = Embedding(vocabulary_size, EMBEDDING_DIM)(target_inp)
target_emb = Flatten()(target_emb)

context_inp = Input(shape=(1,))
context_emb = Embedding(vocabulary_size, EMBEDDING_DIM)(context_inp)
context_emb = Flatten()(context_emb)

x = Dot(axes=1)([target_emb, context_emb])
x = Dense(1, activation="sigmoid")(x)

model = Model([target_inp, context_inp], x)
print(model.summary())

# compile model
optimizer = "adam"
loss = "binary_crossentropy"
model.compile(optimizer=optimizer, loss=loss)

# use gpu if available
device_config = "/CPU:0"
physical_devices = tensorflow.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    device_config = "/GPU:0"

# fit model
print("training model...")
with tensorflow.device(device_config):
    r = model.fit(X, y, epochs=NUM_EPOCHS)

# store model and training data
print("done. saving model and training data...")
model.save("model")
pickle.dump(r.history, open("training_history", "wb"))
print("done.")