import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys

START_TAG = "<s>"
STOP_TAG = "</s>"
EMBEDDING_DIM = 100
BATCH_SIZE = 1
all_words = []
word_to_idx = {}
tag_to_idx = {}
transition_probs = None
emission_probs = None
train_data = []
dev_data = []
        
def load_dataset(path, ret_data):
    file = open(path, "r")
    x_s = [START_TAG]
    y_s = [START_TAG]
    for line in file:
        tokens = line.split()
        if len(tokens) > 0:
            x = tokens[0].lower()  
            y = tokens[1] 
            x_s.append(x)
            y_s.append(y)
        else:
            x_s.append(STOP_TAG)
            y_s.append(STOP_TAG)
            pair = (x_s, y_s)
            ret_data.append(pair)
            x_s = [START_TAG]
            y_s = [START_TAG]

def load_data():
    load_dataset("./31210-s19-hw3/en_ewt.train", train_data)
    load_dataset("./31210-s19-hw3/en_ewt.dev", dev_data)

def init_corpus_tags():
    word_to_idx[START_TAG] = 0
    word_to_idx[STOP_TAG] = 1
    tag_to_idx[START_TAG] = 0
    tag_to_idx[STOP_TAG] = 1
    for pair in train_data:
        words = pair[0]
        tags = pair[1]
        seq_len = len(tags)
        for i in range(seq_len):
            word = words[i]
            if not word in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
            tag = tags[i]
            if not tag in tag_to_idx:
                tag_to_idx[tag] = len(tag_to_idx)
    
def init_transition_probs():
    global transition_probs
    tag_size = len(tag_to_idx)
    transition_probs = np.zeros((tag_size, tag_size))
    for pair in train_data:
        tags = pair[1]
        tags_len = len(tags)
        for i in range(1, tags_len):
            idx_1 = tag_to_idx[tags[i-1]]
            idx_2 = tag_to_idx[tags[i]]
            transition_probs[idx_2, idx_1] += 1

def init_emission_probs():
    global emission_probs
    tag_size = len(tag_to_idx)
    corpus_size = len(word_to_idx)
    emission_probs = np.zeros((corpus_size, tag_size))
    for pair in train_data:
        words = pair[0]
        tags = pair[1]
        seq_len = len(tags)
        for i in range(0, seq_len):
            idx_1 = tag_to_idx[tags[i]]
            idx_2 = word_to_idx[words[i]]
            emission_probs[idx_2, idx_1] += 1
    
def main():
    load_data()
    init_corpus_tags()
    init_transition_probs()
    init_emission_probs()
    
    
if __name__ == '__main__':
    main()