# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import glob
import os

# cleaning function
def clean_raw_data(experiment_name, csv_filepath, num_rows, num_batches=4):

    # load dataframe
    dataframe = pd.read_csv(csv_filepath, sep="\t", skiprows=1, names=["all_data"])

    # extract data from dataframe
    def _process_row(row):
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
    num_rows = len(dataframe) if num_rows == -1 else num_rows
    batch_length = num_rows // num_batches
    for i in range(num_batches):
        start_idx = i * batch_length
        end_idx = ((i + 1) * batch_length) - 1
        clean_sub_dataframe = dataframe.loc[start_idx:end_idx, :].apply(_process_row, axis=1)
        filepath = experiment_name + "/data/{start}-{end}.csv".format(start=start_idx, end=end_idx)
        clean_sub_dataframe.to_csv(filepath, sep="\t")

    # merge sub-dataframes
    filepaths = glob.glob(experiment_name + "/data/*[-]*.csv")
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
    filepath = experiment_name + "/data/combined_dataset.csv"
    combined_dataframe.to_csv(filepath, sep="\t")
    print("exported clean dataset to {path}.".format(path=filepath))