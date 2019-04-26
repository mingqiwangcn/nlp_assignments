import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys


IN_EMBEDDING_DIM = 200
OUT_EMBEDDING_DIM = 200
HIDDEN_DIM = 200

all_words = []
word_to_idx = {}
word_freqs = {}
training_data = []
eval_data = []
test_data = []
show_errors = False
freq_errors = None
best_eval_accu = 0
best_test_accu = 0


class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, word_idxes):
        super(BinaryClassifier, self).__init__()
        corpus_size = len(word_idxes)
        self.word_idxes = torch.tensor(word_idxes, dtype = torch.long)
        self.hidden_dim = hidden_dim
        self.in_word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out_word_embeddings = nn.Embedding(corpus_size, hidden_dim)
        
    def forward(self, word_idxs, start_pos = 0):
        N = len(word_idxs)
        in_embeds = self.in_word_embeddings(word_idxs)
        lstm_out, _ = self.lstm(in_embeds.view(N, 1, -1))
        hidden_state = lstm_out.view(N, -1)
        
        if (not self.training):
            scores = hidden_state.mm(self.label_embs)
            return scores 
                
        return hidden_state
    
    def eval(self):
        ret = nn.Module.eval(self)
        self.label_embs = self.out_word_embeddings(self.word_idxes).t()
        return ret    
         
        

def load_corpus():
    path = "./31210-s19-hw1/bobsue.voc.txt"
    file = open(path, "r")
    idx = 0
    for word in file:
        word = word[:len(word)-1]
        word_to_idx[word] = idx
        all_words.append(word)
        idx += 1

def load_dataset(path, ret_data, is_train = False):
    file = open(path, "r")
    idx = 0
    for line in file:
        tokens = line.split()
        num_words = len(tokens) - 1
        X = []
        y = int(tokens[num_words])
        for i in range(num_words):
            word = tokens[i]
            X.append(word)
            if not word in word_to_idx:
                word_to_idx[word] = idx
                all_words.append(word)
        item = (X, y)
        ret_data.append(item)
        idx += 1

def evaluate_dataset(model, data, is_test = False):
    global freq_errors
    if show_errors and is_test:
        freq_errors = {}
    num_items = 0
    num_correct = 0
    for X, Y, start_pos in data:
        ts_X = torch.tensor(X, dtype = torch.long)
        ts_Y = torch.tensor(Y, dtype = torch.long)
        ts_Y = ts_Y[start_pos:]
        Y_pred = model(ts_X, start_pos).argmax(1)
        num_items += ts_Y.shape[0]
        check_rt = (ts_Y == Y_pred)
        num_correct += check_rt.sum().item()
        if show_errors and is_test:
            for i in range(len(check_rt)):
                if check_rt[i] == 0:
                    e_pair = (ts_Y[i].item(), Y_pred[i].item())
                    e_count = 0
                    if e_pair in freq_errors:
                        e_count = freq_errors[e_pair]
                    e_count += 1
                    freq_errors[e_pair] = e_count
    return num_correct / num_items 

def evaluate(epoc, model):
    global best_eval_accu
    global best_test_accu
    model.eval()
    eval_ratio = evaluate_dataset(model, eval_data, is_test = False)
    if (eval_ratio > best_eval_accu):
        best_eval_accu = eval_ratio
        print("epoc=%d eval accuracy=%.2f" %(epoc, eval_ratio))
        test_ratio = evaluate_dataset(model, test_data, is_test = True)
        if (test_ratio > best_test_accu):
            best_test_accu = test_ratio
            print("epoc=%d test accuracy=%.2f" %(epoc, test_ratio))
    model.train()

def load_data():
    load_dataset("./31210-s19-hw2/senti.train.tsv", training_data, is_train = True)
    load_dataset("./31210-s19-hw2/senti.dev.tsv", eval_data)
    load_dataset("./31210-s19-hw2/senti.test.tsv", test_data)

def eval(model, loss_fn, epocs):
    #torch.manual_seed(1)
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = learing_rate)
    N = len(training_data)
    M = N / 2
    for epoc in range(epocs):
        if epoc > 0:
            random.shuffle(training_data)
        itr = 0
        for X, Y, start_pos in training_data:
            model.zero_grad()
            ts_X = torch.tensor(X, dtype = torch.long)
            ts_Y = torch.tensor(Y, dtype = torch.long)
            ts_Y = ts_Y[start_pos:]
            label_scores = model(ts_X, start_pos)
            loss = loss_fn(label_scores, ts_Y)
            loss.backward()
            optimizer.step()
            itr += 1
            if (itr == M or itr == N):
                print("epoc=%d itr=%d loss=%f" %(epoc, itr, loss.item()))
                evaluate(epoc, model)
    
    print("best_test_accu=%.2f" %(best_test_accu))

def main():
    epocs = 1
    if len(sys.argv) > 1:
        epocs = int(sys.argv[1])
    t1 = time.time()
    load_data()
    '''
    corpus_size = len(lm.word_to_idx)
    labels_size = corpus_size
    model =  lm.LSTMLogLoss(lm.IN_EMBEDDING_DIM, lm.HIDDEN_DIM, corpus_size, labels_size)
    loss_fn = nn.CrossEntropyLoss()
    lm.eval_lm(model, loss_fn, epocs)
    t2 = time.time()
    print("Q1 time=%.3f" %(t2-t1))
    '''
if __name__ == '__main__':
    main()