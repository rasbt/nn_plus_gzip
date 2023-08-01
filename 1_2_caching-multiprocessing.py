# Parallel processing version of 1_2_nn_plus_gzip_fix-tie-breaking.ipynb
# On a 2020 MacBook Air, it runs about 4 times faster ~1 iter/sec
# than the non-parallel version (~4 iter/sec)

# It should finish in about 2-3 h compared to ~12 h before

from collections import Counter
import gzip
import multiprocessing as mp
import os.path as op

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm

from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, partition_dataset


def process_dataset_subset(df_train_subset, test_text, c_test_text, d):

    distances_to_test = []
    for row_train in df_train_subset.iterrows():
        index = row_train[0]
        train_text = row_train[1]["text"]
        c_train_text = d[index]

        train_plus_test = " ".join([test_text, train_text])
        c_train_plus_test = len(gzip.compress(train_plus_test.encode()))

        ncd = ( (c_train_plus_test - min(c_train_text, c_test_text))
                / max(c_test_text, c_train_text) )

        distances_to_test.append(ncd)

    return distances_to_test


def divide_range_into_chunks(start, end, num_chunks):
    chunk_size = (end - start) // num_chunks
    ranges = [(i, i + chunk_size) for i in range(start, end, chunk_size)]
    ranges[-1] = (ranges[-1][0], end)  # Ensure the last chunk includes the end
    return ranges


if __name__ == '__main__':

    if not op.isfile("train.csv") and not op.isfile("val.csv") and not op.isfile("test.csv"):
        download_dataset()

        df = load_dataset_into_to_dataframe()
        partition_dataset(df)

    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("val.csv")
    df_test = pd.read_csv("test.csv")

    num_processes = mp.cpu_count()
    k = 2
    predicted_classes = []

    start = 0
    end = df_train.shape[0]
    ranges = divide_range_into_chunks(start, end, num_chunks=num_processes)


    # caching compressed training examples
    d = {}
    for i, row_train in enumerate(df_train.iterrows()):
        train_text = row_train[1]["text"]
        train_label = row_train[1]["label"]
        c_train_text = len(gzip.compress(train_text.encode()))
        
        d[i] = c_train_text

    # main loop
    for row_test in tqdm(df_test.iterrows(), total=df_test.shape[0]):

        test_text = row_test[1]["text"]
        test_label = row_test[1]["label"]
        c_test_text = len(gzip.compress(test_text.encode()))
        all_train_distances_to_test = []

        # parallelize iteration over training set into num_processes chunks
        with Parallel(n_jobs=num_processes, backend="loky") as parallel:

            results = parallel(
                delayed(process_dataset_subset)(df_train[range_start:range_end], test_text, c_test_text, d)
                    for range_start, range_end in ranges
                )
            for p in results:
                all_train_distances_to_test.extend(p)

        sorted_idx = np.argsort(np.array(all_train_distances_to_test.extend))
        top_k_class = np.array(df_train["label"])[sorted_idx[:k]]
        predicted_class = Counter(top_k_class).most_common()[0][0]

        predicted_classes.append(predicted_class)

    print("Accuracy:", np.mean(np.array(predicted_classes) == df_test["label"].values))