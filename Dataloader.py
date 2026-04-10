import requests
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# GET THE DATA (THE ENTIRE TEXT)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"

def download_dataset()->None:
    if (os.path.exists(DATA_PATH)):
        print("File already exists")
        return

    text_file = requests.get(url).text
    with open(DATA_PATH , "w") as f:
        f.write(text_file)

def load_dataset(print_text = False)->str:
    with open(DATA_PATH , "r") as f:
        txt = f.read()

    print("Total characters in text : " , len(txt))
    if (print_text):
        print (txt[:500])

    return txt

if __name__ == "__main__":
    download_dataset()

text = load_dataset(print_text= False)
