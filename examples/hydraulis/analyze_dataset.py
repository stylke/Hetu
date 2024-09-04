import os
import signal
import time
import argparse
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from data_utils import LLaMAJsonDataset

counter_file_path = "_counter.pkl"
cdf_file_path = "cdf.png"

def read_counter():
    with open(counter_file_path, 'rb') as file:
        counter = pickle.load(file)
    print(f"Max seq len is {max(counter.keys())}")

def scan_and_dump(dataset, batch_size=1000):
    data = dataset.data
    total_seqs = len(data)
    print(f"Total seqs in dataset: {total_seqs}")
    # Initialize an empty list to store sequence lengths
    seqlen_list = []
    # Process data in batches
    for i in tqdm(range(0, total_seqs, batch_size), desc="Processing batches"):
        batch_data = data[i: min(i + batch_size, total_seqs)]
        batch_array = np.array(batch_data)
        batch_seqlen = np.sum(batch_array != dataset.encoder.pad_id(), axis=1)
        seqlen_list.extend(batch_seqlen)
    # Convert the list to a numpy array for further processing
    # Count the occurrences of each sequence length
    counter = Counter(seqlen_list)
    with open(counter_file_path, 'wb') as file:
        pickle.dump(counter, file)
    x_vals, counts = zip(*sorted(counter.items()))  
    # Calculate the cumulative distribution function (CDF)
    y_vals = np.cumsum(counts) / total_seqs
    # Plot the CDF
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='CDF', color='blue', lw=2)
    plt.fill_between(x_vals, y_vals, color='blue', alpha=0.3)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=16)
    plt.xlabel('Sequence Length', fontsize=14)
    plt.ylabel('Cumulative Probability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(cdf_file_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_seq_len", type=int, default=32768, help="maximum sequence len"
    )
    parser.add_argument(
        "--json_file", type=str, default="data/web/refinedweb0.json", help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, default="content", help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, default="data/vocab.json", help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, default="data/merges.txt", help='gpt merge file path'
    )
    args = parser.parse_args()
    dataset = LLaMAJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.max_seq_len,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file
    )
    scan_and_dump(dataset)
    # read_counter()