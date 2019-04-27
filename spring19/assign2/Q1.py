import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys

EMBEDDING_DIM = 100
BATCH_SIZE = 1

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
        ts_all_embeds = torch.tensor(all_embeds, dtype = torch.double)
        self.word_embeddings = nn.Embedding.from_pretrained(ts_all_embeds, freeze = False)
        #self.word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.weights = torch.ones(embedding_dim).double().reshape(embedding_dim, 1);
        
    def forward(self, batch_word_idxs):
        hidden_lst = []
        for word_idxs in batch_word_idxs:
            embeds = self.word_embeddings(word_idxs)
            hidden = embeds.mean(dim = 0)
            hidden_lst.append(hidden)
        hiddens = torch.cat(hidden_lst).reshape(-1, EMBEDDING_DIM)
        prod = hiddens.mm(self.weights)
        return prod

class BinaryLogLoss(nn.Module):
    def __init__(self):
        super(BinaryLogLoss, self).__init__()
        self.log_sig = nn.LogSigmoid()
    
    def forward(self, out_prod, y_s):
        float_y_s = y_s.double()
        loss = - float_y_s * self.log_sig(out_prod) - (1.0 - float_y_s) * self.log_sig(-out_prod)
        return loss.mean()
        

def load_dataset(path, ret_data, is_train = False):
    x_s = []
    y_s = []
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
                    idx += 1
                w_id = word_to_idx[word]
            else:
                if not word in word_to_idx:
                    w_id = unk_id
                else:
                    w_id = word_to_idx[word]
            x.append(w_id)
        ts_x = torch.tensor(x, dtype = torch.long)
        x_s.append(ts_x)
        y_s.append(y)
    ret_data.append(x_s)
    ts_ys = torch.tensor(y_s, dtype = torch.uint8)
    ret_data.append(ts_ys)
    if is_train:
        all_words.append(UNKNOWN_WORD)
        word_to_idx[UNKNOWN_WORD] = len(all_words) - 1
    
def evaluate_dataset(model, data):
    sig = nn.Sigmoid()
    x_s = data[0]
    y_s = data[1]
    num_items = len(x_s)
    prod = model(x_s)
    scores = sig(prod).reshape(num_items)
    y_pred = (scores >= 0.5)
    rt = (y_s == y_pred)
    num_correct = rt.sum().item() 
    return num_correct / num_items 

def evaluate(epoc, model):
    global best_eval_accu
    global best_test_accu
    model.eval()
    eval_ratio = evaluate_dataset(model, eval_data)
    if (eval_ratio > best_eval_accu):
        best_eval_accu = eval_ratio
        print("epoc=%d eval accuracy=%.2f" %(epoc, eval_ratio))
        test_ratio = evaluate_dataset(model, test_data)
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
    N = len(training_data[0])
    num_batches = int(N / BATCH_SIZE)
    if (N % BATCH_SIZE):
        num_batches += 1
    M = int(num_batches / 3)
    test_itrs = [M, M + M, num_batches]
    for epoc in range(epocs):
        if epoc > 0:
            random.shuffle(training_data)
        itr = 0
        pos1 = 0
        pos2 = 0
        for i in range(num_batches):
            model.zero_grad()
            pos2 = pos1 + BATCH_SIZE
            x_s = training_data[0][pos1: pos2]
            y_s = training_data[1][pos1: pos2]
            pos1 = pos2
            prod = model(x_s)
            loss = loss_fn(prod, y_s)
            loss.backward()
            optimizer.step()
            itr += 1
            if (itr % 1000 == 0):
                print("batch size=%d epoc=%d itr=%d loss=%f" %(BATCH_SIZE, epoc, itr, loss.item()))
                evaluate(epoc, model)
    
    print("best_test_accu=%.2f" %(best_test_accu))

def main():
    global BATCH_SIZE
    epocs = 5
    if len(sys.argv) > 2:
        epocs = int(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
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