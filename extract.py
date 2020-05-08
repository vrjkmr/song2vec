# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os

# hyperparameters
NUM_ROWS_TO_EXTRACT = 2000 # set to -1 to extract entire dataset
STRIDE_LENGTH = 1000

# load dataframe
print("loading dataframe...")
filename = "data/spotify_playlists/spotify_dataset.csv"
dataframe = pd.read_csv(filename, sep="\t", skiprows=1, names=["all_data"])

# extract data from dataframe
def extract(row):
    user_id, track = row["all_data"].split(",", 1)
    f_strip_quotes = lambda x: x.strip('"')
    artist_name, track_name, playlist_name = map(f_strip_quotes, track.split('","'))
    row["user_id"] = user_id
    row["artist_name"] = artist_name.lower()
    row["track_name"] = track_name.lower()
    row["playlist_name"] = playlist_name.lower()
    row["track_full_name"] = str(row["artist_name"]) + " - " + str(row["track_name"])
    row = row.drop("all_data")
    return row

# iterate over dataframe
print("cleaning dataframe...")
if NUM_ROWS_TO_EXTRACT == -1:
    NUM_ROWS_TO_EXTRACT = len(dataframe)
num_strides = NUM_ROWS_TO_EXTRACT // STRIDE_LENGTH
for i in range(num_strides):
    start_idx = i * STRIDE_LENGTH
    end_idx = ((i + 1) * STRIDE_LENGTH) - 1
    print("- processing {}/{}...".format(i, num_strides))
    clean_sub_dataframe = dataframe.loc[start_idx:end_idx, :].apply(extract, axis=1)
    filename = "data/{}-{}.csv".format(start_idx, end_idx)
    clean_sub_dataframe.to_csv(filename, sep="\t")

# merge sub-dataframes
print("done. merging cleaned dataframes...")
filepaths = glob.glob("data/*[-]*.csv")
filepaths.sort(key=lambda x: int(x.split("-")[0].split("/")[-1]))
combined_dataframe = pd.read_csv(filepaths[0], sep="\t", index_col=0)
if len(filepaths) > 1:
    for i in range(1, len(filepaths)):
        next_dataframe = pd.read_csv(filepaths[i], sep="\t", index_col=0)
        combined_dataframe = pd.concat([combined_dataframe, next_dataframe])

# remove sub-dataframe files
for filepath in filepaths:
    os.remove(filepath)

# export combined dataframe
print("exporting final dataframe...")
filename = "data/spotify_playlists/combined_dataset.csv"
combined_dataframe.to_csv(filename, sep="\t")
print("done.")