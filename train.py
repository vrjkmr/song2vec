# -*- coding: utf-8 -*-
import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam
import pickle

# training function
def build_and_train_model(experiment_name, embedding_dim, learning_rate, num_epochs, random_seed=42):

    # load tokenizer
    tokenizer = pickle.load(open(experiment_name + "/data/tokenizer.pkl", "rb"))
    track2idx = tokenizer.word_index
    vocabulary_size = len(track2idx) + 1

    # load training data
    X = pickle.load(open(experiment_name + "/data/X.pkl", "rb"))
    y = pickle.load(open(experiment_name + "/data/y.pkl", "rb"))

    # set seed
    tensorflow.random.set_seed(random_seed)

    # build model architecture
    target_inp = Input(shape=(1,)) 
    target_emb = Embedding(vocabulary_size, embedding_dim)(target_inp)
    target_emb = Flatten()(target_emb)

    context_inp = Input(shape=(1,))
    context_emb = Embedding(vocabulary_size, embedding_dim)(context_inp)
    context_emb = Flatten()(context_emb)

    x = Dot(axes=1)([target_emb, context_emb])
    x = Dense(1, activation="sigmoid")(x)

    model = Model([target_inp, context_inp], x)
    print(model.summary())

    # compile model
    optimizer = Adam(learning_rate=learning_rate)
    loss = "binary_crossentropy"
    model.compile(optimizer=optimizer, loss=loss)

    # use gpu if available
    device_config = "/CPU:0"
    physical_devices = tensorflow.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        device_config = "/GPU:0"

    # fit model
    with tensorflow.device(device_config):
        r = model.fit(X, y, epochs=num_epochs, verbose=2)

    # store model
    filepath = experiment_name + "/model"
    model.save(filepath)
    print("exported trained model to {path}.".format(path=filepath))

    # store embedding weights
    target_embedding_layer = model.layers[2]
    embedding_weights = target_embedding_layer.get_weights()[0]
    filepath = experiment_name + "/embeddings.pkl"
    pickle.dump(embedding_weights, open(filepath, "wb"))
    print("exported track embeddings to {path}.".format(path=filepath))