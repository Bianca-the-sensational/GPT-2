import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from Dataloader import text

# DATA PREPROCESSING (CONVERTING TEXT TO TOKENS , VOCAB SIZE)
# TOKENISER DEFINITION
def get_stats(ids , counts = None):
    counts = {} if counts is None else counts
    for pair in zip(ids , ids[1 : ]):
        counts[pair] = counts.get(pair , 0) + 1
    return counts

def merge(ids , pair , idx):
    i = 0
    newids = [] # empty list to store the new modified tokens after idx have been replcaed with the pair token

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids

class Tokeniser():

    def __init__(self):
        super().__init__()

    def train(self , text , vocab_size , verbose = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        merges = {}
        # define a dictionary vocab which pairs each token idx to a byte
        vocab = {idx : bytes([idx]) for idx in range (256)}

        for i in range(num_merges):
            # get the dictionary which compares the frequency of all the consecutive pairs in a text
            stats = get_stats(ids)
            # get the pair that appears the MAX number of times
            pair = max (stats , key = stats.get)
            # assign what new idx is to be assigned
            idx = 256 + i
            # add the pair in the merges dictionary
            merges[pair] = idx
            # Replace the tokens pair with the 2 consecutive elements
            ids = merge(ids , pair , idx)
            # Add the new idx with the corresponding byte to the vocabulary
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.vocab = vocab
        self.merges = merges

    # get the text back from the tokens list (ids)
    def decode(self , ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8" , errors = "replace")
        return text

    # convert the given text to a list of tokens
    def encode(self , text):
        text_bytes = text.encode("utf-8")
        # list of tokens indexes on character level
        ids = list(text_bytes)

        while (len(ids) >= 2):
            stats = get_stats(ids)
            pair = min(stats , key = lambda p : self.merges.get(p , float("inf")))

            if pair not in self.merges:
                break # no pair is left to be merged
            idx = self.merges[pair]
            ids = merge(ids , pair , idx)

        return ids

torch.manual_seed(42)
tokeniser = Tokeniser()
vocab_size = 1024

# SPLITTING THE DATA INTO TRAIN , VAL , TEST
n1 = int (0.8 * len(text))
n2 = int (0.9 * len(text))

train_text = text[: n1]
val_text = text[n1 : n2]
test_text = text[n2 :]

tokeniser.train(train_text , vocab_size , verbose = False)

train_data = tokeniser.encode(train_text)
val_data = tokeniser.encode(val_text)
test_data = tokeniser.encode(test_text)
