import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys

EARLY_STOP_ITR = 100
EMBEDDING_DIM = 100
BATCH_SIZE = 1
UNKNOWN_WORD = "_unk_"
all_words = []
word_to_idx = {}
training_data = []
eval_data = []
test_data = []
best_eval_accu = 0
best_test_accu = 0
        
class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim, corpus_size):
        super(BinaryClassifier, self).__init__()
        np.random.seed(100)
        all_embeds = np.random.uniform(-1.0, 1.0, (corpus_size, embedding_dim))
        ts_all_embeds = torch.tensor(all_embeds, dtype = torch.double)
        self.word_embeddings = nn.Embedding.from_pretrained(ts_all_embeds, freeze = False)
        
        ts_weights = torch.ones(EMBEDDING_DIM, 1, dtype = torch.double)
        self.weights = nn.Parameter(ts_weights);
        
        self.M = 50
        self.num_chunk = int(embedding_dim / self.M)
        
        ts_attend_u = torch.ones(EMBEDDING_DIM, dtype = torch.double)
        mult_attend_u = ts_attend_u.reshape(-1, self.M)
        self.mult_attend_u = nn.Parameter(mult_attend_u)
           
        
        self.cosine = nn.CosineSimilarity(dim = 0)
    
    def compute_multi_hidden(self, embeds, attend_u):
        sent_len = embeds.shape[0]
        dist_lst = []
        for i in range(sent_len):
            dist = self.cosine(attend_u, embeds[i]).reshape(1,1)
            dist_lst.append(dist)
        dists = torch.cat(dist_lst, dim = 0)
        alphas = torch.exp(dists)
        norm_alphas = nn.functional.normalize(alphas, p = 1, dim = 0).reshape(sent_len, 1)  
        hidden = (embeds * norm_alphas).sum(dim = 0)
        return hidden
    
    def forward(self, batch_word_idxs):
        hidden_lst = []
        for word_idxs in batch_word_idxs:
            embeds = self.word_embeddings(word_idxs)
            multi_hidden_lst = []
            multi_embeds = torch.split(embeds, self.M, dim = 1)
            for i in range(self.num_chunk):
                hidden = self.compute_multi_hidden(multi_embeds[i], self.mult_attend_u[i])
                multi_hidden_lst.append(hidden)
            multi_hidden = torch.cat(multi_hidden_lst)
            hidden_lst.append(multi_hidden)
            
        hiddens = torch.cat(hidden_lst).reshape(-1, EMBEDDING_DIM)
        prod = hiddens.mm(self.weights)
        return prod
        
class BinaryLogLoss(nn.Module):
    def __init__(self):
        super(BinaryLogLoss, self).__init__()
        self.log_sig = nn.LogSigmoid()
    
    def forward(self, out_prod, y_s):
        d_y_s = y_s.double()
        loss = - d_y_s * self.log_sig(out_prod) - (1.0 - d_y_s) * self.log_sig(-out_prod)
        return loss.mean()
        
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
                    idx += 1
                w_id = word_to_idx[word]
            else:
                if not word in word_to_idx:
                    w_id = unk_id
                else:
                    w_id = word_to_idx[word]
            x.append(w_id)
        ts_x = torch.tensor(x, dtype = torch.long)
        pair = (ts_x, y)
        ret_data.append(pair)

    if is_train:
        all_words.append(UNKNOWN_WORD)
        word_to_idx[UNKNOWN_WORD] = len(all_words) - 1
    
def evaluate_dataset(model, data):
    sig = nn.Sigmoid()
    x_s, y_s = get_data_x_y(data)
    
    num_items = len(x_s)
    prod = model(x_s)
    scores = sig(prod).reshape(num_items)
    y_pred = (scores >= 0.5)
    ts_y_s = torch.tensor(y_s, dtype = torch.uint8) 
    rt = (ts_y_s == y_pred)
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
        print("epoc=%d test accuracy=%.2f" %(epoc, test_ratio))
        if (test_ratio > best_test_accu):
            best_test_accu = test_ratio
            
    model.train()

def load_data():
    load_dataset("./31210-s19-hw2/senti.train.tsv", training_data, is_train = True)
    load_dataset("./31210-s19-hw2/senti.dev.tsv", eval_data)
    load_dataset("./31210-s19-hw2/senti.test.tsv", test_data)

def get_data_x_y(data):
    d = list(zip(*data))
    data_x_s = list(d[0])
    data_y_s = list(d[1])
    return data_x_s, data_y_s

def eval_model(model, loss_fn, epocs):
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = learing_rate)
    N = len(training_data)
    num_batches = int(N / BATCH_SIZE)
    if (N % BATCH_SIZE):
        num_batches += 1
        
    for epoc in range(epocs):
        if epoc > 0:
            random.shuffle(training_data)
        data_x_s, data_y_s = get_data_x_y(training_data)
        pos1 = 0
        pos2 = 0
        for batch in range(num_batches):
            model.zero_grad()
            pos2 = pos1 + BATCH_SIZE
            x_s = data_x_s[pos1: pos2]
            y_s = data_y_s[pos1: pos2]
            ts_ys = torch.tensor(y_s, dtype = torch.uint8)
            pos1 = pos2
            prod = model(x_s)
            loss = loss_fn(prod, ts_ys)
            loss.backward()
            optimizer.step()
            itr = batch + 1
            if itr % EARLY_STOP_ITR == 0:
                print("batch size=%d epoc=%d itr=%d loss=%f" %(BATCH_SIZE, epoc, itr, loss.item()))
                evaluate(epoc, model)
    
    print("best_test_accu=%.2f" %(best_test_accu))
    
def main():
    global BATCH_SIZE
    global EARLY_STOP_ITR
    epocs = 1
    if len(sys.argv) > 3:
        epocs = int(sys.argv[1])
        BATCH_SIZE = int(sys.argv[2])
        EARLY_STOP_ITR = int(sys.argv[3])
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