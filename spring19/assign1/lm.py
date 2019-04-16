import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


IN_EMBEDDING_DIM = 128
OUT_EMBEDDING_DIM = 128
HIDDEN_DIM = 128

all_words = []
word_to_idx = {}
word_freqs = {}
training_data = []
eval_data = []
test_data = []
show_errors = False
freq_errors = {}
best_eval_accu = 0
best_test_accu = 0


class LSTMLogLoss(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, corpus_size, labels_size):
        super(LSTMLogLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_label = nn.Linear(hidden_dim, labels_size)
        
    def forward(self, word_idxs, start_pos = 0):
        N = len(word_idxs)
        embeds = self.word_embeddings(word_idxs)
        lstm_out, _ = self.lstm(embeds.view(N, 1, -1))
        hidden_state = lstm_out.view(N, -1)
        hidden_state = hidden_state[start_pos:,:]    
        scores = self.hidden_to_label(hidden_state)
        return scores

class LSTMBinaryLogLoss(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, word_idxes):
        super(LSTMBinaryLogLoss, self).__init__()
        corpus_size = len(word_idxes)
        self.word_idxes = torch.tensor(word_idxes, dtype = torch.long)
        self.hidden_dim = hidden_dim
        self.in_word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out_word_embeddings = self.in_word_embeddings #nn.Embedding(corpus_size, hidden_dim)
        
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
         
        
class BinaryLogLoss(nn.Module):
    def __init__(self, out_word_embeddings, neg_distr, num_neg_samples):
        super(BinaryLogLoss, self).__init__()
        self.out_word_embeddings = out_word_embeddings
        self.log_sig = nn.LogSigmoid()
        self.neg_distr = neg_distr
        self.num_neg_samples = num_neg_samples
    def forward(self, hidden_state, label_idxex):
        out_embeds = self.out_word_embeddings(label_idxex)
        N = len(label_idxex)
        neg_idxes = self.neg_distr.sampling(self.num_neg_samples * N)
        neg_embeds = self.out_word_embeddings(neg_idxes)
        neg_embeds_3d = neg_embeds.view(N, self.num_neg_samples, -1)
        hidden_state_3d = hidden_state.view(N, 1, -1)
        out_embeds_3d = out_embeds.view(N, 1, -1)
        label_embeds = torch.cat((out_embeds_3d, neg_embeds_3d), dim = 1)
        label_scores = (label_embeds * hidden_state_3d).sum(dim = 2)
        label_scores[:,1:] = -label_scores[:,1:]
        label_sig_scores = self.log_sig(label_scores)
        total_score = (label_sig_scores[:,0] + label_sig_scores[:,1:].mean(dim = 1)).sum()
        loss = -total_score
        return loss

class Distribution:
    def __init__(self):
        return
    def sampling(self, N):
        return

class UniformDistr(Distribution):
    def __init__(self, word_idxes):
        super(UniformDistr, self).__init__()
        self.word_idxes = word_idxes
    
    def sampling(self, N):
        samples = np.random.choice(self.word_idxes, N, replace = False)
        return torch.from_numpy(samples)

class UnigfDistr(Distribution):
    def __init__(self, idxes, word_freqs, f):
        super(UnigfDistr, self).__init__()
        self.idxes = idxes
        num_words = len(idxes)
        weights = [None] * num_words
        for i in range(num_words):
            freq = 0
            if idxes[i] in word_freqs:
                freq = word_freqs[idxes[i]]
            weights[i] = freq**f
        probs = normalize_weights(weights)
        self.probs = probs
    
    def sampling(self, N):
        samples = np.random.choice(self.idxes, N, replace = False, p = self.probs) 
        return torch.from_numpy(samples)

def normalize_weights(weights):
    w_sum = np.sum(weights)
    N = len(weights)
    probs = [None] * N
    prob_sum = 0.0
    for i in range(N - 1):
        probs[i] = weights[i] / w_sum
        prob_sum += probs[i] 
    probs[N - 1] = 1.0 - prob_sum
    return probs


def load_corpus():
    path = "./31210-s19-hw1/bobsue.voc.txt"
    file = open(path, "r")
    idx = 0
    for word in file:
        word = word[:len(word)-1]
        word_to_idx[word] = idx
        all_words.append(word)
        idx += 1

def load_dataset(path, ret_data):
    file = open(path, "r")
    for line in file:
        words = line.split()
        num_words = len(words)
        word_idxs = [None] * num_words
        for i in range(num_words):
            word_idxs[i] = word_to_idx[words[i]]
            freq = 0
            if word_idxs[i] in word_freqs:
                freq = word_freqs[word_idxs[i]]
            word_freqs[word_idxs[i]] = freq + 1
            
        X = word_idxs[:(num_words-1)]
        Y = word_idxs[1:]
        item = (X, Y, 0)
        ret_data.append(item)

def load_prevsent_dataset(path, ret_data):
    file = open(path, "r")
    for line in file:
        words = line.split()
        num_words = len(words)
        word_idxs = [None] * num_words
        for i in range(num_words):
            word_idxs[i] = word_to_idx[words[i]]
        X = word_idxs[:(num_words-1)]
        Y = word_idxs[1:]
        strt_idx = word_to_idx["<s>"]
        start_pos = Y.index(strt_idx)
        item = (X, Y, start_pos)
        ret_data.append(item)

def evaluate_dataset(model, data, is_test = False):
    num_items = 0
    num_correct = 0
    for X, Y, start_pos in data:
        ts_X = torch.tensor(X, dtype = torch.long)
        ts_Y = torch.tensor(Y, dtype = torch.long)
        ts_Y = ts_Y[start_pos:]
        Y_pred = model(ts_X, start_pos).argmax(1)
        num_items += ts_X.shape[0]
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
    load_corpus()
    load_dataset("./31210-s19-hw1/bobsue.lm.train.txt", training_data)
    load_dataset("./31210-s19-hw1/bobsue.lm.dev.txt", eval_data)
    load_dataset("./31210-s19-hw1/bobsue.lm.test.txt", test_data)

def load_prevsent_data():
    load_corpus()
    load_prevsent_dataset("./31210-s19-hw1/bobsue.prevsent.train.tsv", training_data)
    load_prevsent_dataset("./31210-s19-hw1/bobsue.prevsent.dev.tsv", eval_data)
    load_prevsent_dataset("./31210-s19-hw1/bobsue.prevsent.test.tsv", test_data)

def eval_lm(model, loss_fn, epocs):
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

def list_errors(N):
    errors = sorted(freq_errors.items(), key=lambda x: x[1])
    error_count = len(errors)
    i = 0
    j = error_count - 1
    err_str = ""
    print("Top 35 errors")
    while (i < N):
        if (j >= 0):
            y_g = errors[j][0][0]
            y_p = errors[j][0][1]
            freq = errors[j][1]
            word_g = all_words[y_g]
            word_p = all_words[y_p]
            str1 = "(%s %s %d)" %(word_g, word_p, freq)
            err_str = err_str.join(" ")
            err_str = err_str.join(str1)
            if (i % 7) == 0:
                print(err_str)
                err_str = ""
            j -= 1
        else:
            break
        i += 1
