import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys

EMBEDDING_DIM = 100

UNKNOWN_WORD = "_unk_"
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
    def __init__(self, embedding_dim, corpus_size):
        super(BinaryClassifier, self).__init__()
        np.random.seed(100)
        all_embeds = np.random.uniform(-1.0, 1.0, (corpus_size, embedding_dim))
        ts_all_embeds = torch.tensor(all_embeds)
        nn.Embedding.from_pretrained(ts_all_embeds, freeze = False)
        self.word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.weights = torch.ones(embedding_dim);
        
    def forward(self, word_idxs):
        N = len(word_idxs)
        embeds = self.word_embeddings(word_idxs)
        hidden = embeds.mean(dim = 0)
        prod = self.weights.dot(hidden)
        return prod

class BinaryLogLoss(nn.Module):
    def __init__(self):
        super(BinaryLogLoss, self).__init__()
        self.log_sig = nn.LogSigmoid()
    
    def forward(self, out_prod, y):
        loss = - (y == 1) * self.log_sig(out_prod) - (y == 0) * self.log_sig(-out_prod)
        return loss
        

def load_dataset(path, ret_data, is_train = False):
    file = open(path, "r")
    idx = 0
    unk_id = None
    if not is_train:
        unk_id = word_to_idx[UNKNOWN_WORD]
    for line in file:
        tokens = line.split()
        num_words = len(tokens) - 1
        x = []
        y = int(tokens[num_words])
        for i in range(num_words):
            word = tokens[i]
            w_id = None
            if is_train:
                if not word in word_to_idx:
                    word_to_idx[word] = idx
                    all_words.append(word)
                w_id = word_to_idx[word]
            else:
                if not word in word_to_idx:
                    w_id = unk_id
                else:
                    w_id = word_to_idx[word]
            x.append(w_id)
        ts_x = torch.tensor(x, dtype = torch.float)
        item = (ts_x, y)
        ret_data.append(item)
        idx += 1    
    
    if is_train:
        all_words.append(UNKNOWN_WORD)
        word_to_idx[UNKNOWN_WORD] = len(all_words) - 1
    
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

def eval_model(model, loss_fn, epocs):
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = learing_rate)
    N = len(training_data)
    M = N / 2
    for epoc in range(epocs):
        if epoc > 0:
            random.shuffle(training_data)
        itr = 0
        for x, y in training_data:
            model.zero_grad()
            prod = model(x)
            loss = loss_fn(prod, y)
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
    corpus_size = len(all_words)
    model = BinaryClassifier(EMBEDDING_DIM, corpus_size)
    loss_fn = BinaryLogLoss()
    eval_model(model, loss_fn, epocs)
    t2 = time.time()
    print("Q1 time=%.3f" %(t2-t1))
    
if __name__ == '__main__':
    main()