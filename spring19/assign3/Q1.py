import random
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
all_tags = []
transition_probs = None
emission_probs = None
lambda_transition = 0.1
lambda_emission = 0.001
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
    global all_words
    global all_tags
    
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

    all_words = [None] * len(word_to_idx)
    all_tags = [None] * len(tag_to_idx)
    for word in word_to_idx:
        all_words[word_to_idx[word]] = word
    for tag in tag_to_idx:
        all_tags[tag_to_idx[tag]] = tag
    
def normalize_probs(probs):
    probs_sum = np.sum(probs, axis=1, keepdims=True)
    norm_probs = probs / probs_sum
    norm_probs_sum = np.sum(norm_probs, axis=1)
    probs_diff = norm_probs_sum - 1
    idxes = np.argmax(norm_probs, axis=1)
    for i in range(len(idxes)):
        norm_probs[i][idxes[i]] += probs_diff[i]
    
    return norm_probs
    
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
            transition_probs[idx_1][idx_2] += 1
            
    transition_probs += lambda_transition
    transition_probs = normalize_probs(transition_probs)
    
    transition_probs[tag_to_idx[START_TAG]][tag_to_idx[STOP_TAG]] = 0
    transition_probs[:, tag_to_idx[START_TAG]] = 0
    transition_probs[tag_to_idx[STOP_TAG], :] = 0

def init_emission_probs():
    global emission_probs
    tag_size = len(tag_to_idx)
    corpus_size = len(word_to_idx)
    emission_probs = np.zeros((tag_size, corpus_size))
    for pair in train_data:
        words = pair[0]
        tags = pair[1]
        seq_len = len(tags)
        for i in range(0, seq_len):
            idx_1 = tag_to_idx[tags[i]]
            idx_2 = word_to_idx[words[i]]
            emission_probs[idx_1][idx_2] += 1
    
    emission_probs += lambda_emission
    emission_probs = normalize_probs(emission_probs)
    
    emission_probs[tag_to_idx[START_TAG], :] = 0
    emission_probs[tag_to_idx[STOP_TAG], :] = 0

def print_top_probs():
    probs = transition_probs[tag_to_idx[START_TAG]]
    K = 5
    top_5_idxes = np.argpartition(probs,-K)[-K:]
    top_5_pairs = []
    for i in range(K):
        idx = top_5_idxes[i]
        pair = (all_tags[idx], probs[idx])
        top_5_pairs.append(pair)
    
    print(top_5_pairs)
    
    probs = emission_probs[tag_to_idx["JJ"]]
    K = 10
    top_10_idxes = np.argpartition(probs,-K)[-K:]
    top_10_pairs = []
    for i in range(K):
        idx = top_10_idxes[i]
        pair = (all_words[idx], probs[idx])
        top_10_pairs.append(pair)
         
    print(top_10_pairs)

def main():
    load_data()
    init_corpus_tags()
    init_transition_probs()
    init_emission_probs()
    print_top_probs()
    
    
if __name__ == '__main__':
    main()