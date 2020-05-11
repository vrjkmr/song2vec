# Skipgramophone

This repository contains the code for a skip-gram model for learning distributed representations of songs based on the context in which they appear in a playlist.

### Motivation

This project was based on a [2017 talk by Spotify ML](https://youtu.be/HKW_v0xLHH4) on using AI to build predictive platforms. One of the topics in the talk revolved around leveraging NLP research on word embeddings ([word2vec](https://arxiv.org/abs/1301.3781)) to model track similarity. By modelling track similarity, a user can be recommeded new and similar tracks based on the tracks they have listened to in the past.

**Main idea:** Based on the assumption that users tend to listen to similar tracks in a sequence, one could build a word2vec model that learns a vector embedding for any given track, using the tracks that appear before and after it. Once trained, the embedding weights would be organized in such a way that similar tracks would have similar vector embeddings.

### Approach

[Zenodo's Spotify Playlists Dataset](https://zenodo.org/record/2594557) contains 1.2GB of playlists along with the track list for each playlist. I extracted a small fraction of the dataset, then built and trained a skip-gram model with negative sampling using the [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) library.

### Results

The model seems to have not only learn similar vectors for tracks that appear close to one another in a given playlist, but was also able to learn some interesting relations between songs that appear nowhere near each other in the dataset.

Below is an example showing songs similar to John Mayer's cover of Tom Petty's "Free Fallin'" post-training.

```
top 10 tracks most similar to 'john mayer - free fallin' - live at the nokia theatre' (pos. 4602):
- (sim. 0.562): 'the martini's - free' (pos. 4600)
- (sim. 0.446): 'the cranberries - free to decide' (pos. 4605)
- (sim. 0.389): 'säkert! - fredrik' (pos. 4599)
- (sim. 0.388): 'rascals - freakbeat phantom' (pos. 4597)
- (sim. 0.388): 'tom petty - free fallin'' (pos. 206)                     <---- original track
- (sim. 0.377): 'cary brothers - free like you make me' (pos. 4603)
- (sim. 0.350): 'robot koch - nitesky feat. john lamonica' (pos. 6755)
- (sim. 0.331): 'håkan hellström - brännö serenad' (pos. 3480)
- (sim. 0.330): 'justice - the party' (pos. 264)
- (sim. 0.323): 'muse - resistance' (pos. 7376)
```

As can be seen above, the model was able to learn the similarity between John Mayer's cover and Tom Petty's original version, even though both tracks do not appear close to each other in the dataset. John Mayer's cover appears in position 4602 and Tom Petty's original appears in position 206.

### Project structure

[TODO](): Write this section.

## Running the project

### Set up

1. Clone the project
2. Create and activate a new conda environment
3. Run `pip install -r requirements.txt`

### Make inferences using the trained model

To visualize the similarities learned by the trained model, open `notebooks/Inference.ipynb`.

### Run your own experiment

The script `run_experiment.py` enables you to build and run your own experiment. To view the list of parameters, run `python run_experiment.py -h`.

```
usage: run_experiment.py [-h] [--csv_path CSV_PATH] [-nr NUM_ROWS]
                         [-wsz WINDOW_SIZE] [-emb EMBEDDING_DIM] [-e EPOCHS]
                         [-lr LEARNING_RATE] [-rs RANDOM_SEED]
                         experiment_name

positional arguments:
  experiment_name       name of experiment

optional arguments:
  -h, --help            show this help message and exit
  --csv_path CSV_PATH   path to original .csv dataset
  -nr NUM_ROWS, --num_rows NUM_ROWS
                        number of rows to extract (-1 for whole dataset)
  -wsz WINDOW_SIZE, --window_size WINDOW_SIZE
                        (half) window size for skip-gram contexts
  -emb EMBEDDING_DIM, --embedding_dim EMBEDDING_DIM
                        word2vec embedding dimension size
  -e EPOCHS, --epochs EPOCHS
                        number of training epochs
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for Adam optimizer
  -rs RANDOM_SEED, --random_seed RANDOM_SEED
                        random seed for skip-gram generation and training
```

Once run, the relevant files (including the model, embeddings, and tokenizer) will be saved in the `experiments/{experiment_name}` directory. To make inferences using this new experiment's results, simply open `notebooks/Inference.ipynb` and change the `experiment_name` variable.

### Acknowledgements

[TODO](): Write this section.
