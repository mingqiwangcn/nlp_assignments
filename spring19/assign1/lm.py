import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

IN_EMBEDDING_DIM = 128
OUT_EMBEDDING_DIM = 128
HIDDEN_DIM = 128

EPOCS = 10
word_to_idx = {}
word_freqs = {}
training_data = []
eval_data = []
test_data = []
best_eval_accu = 0
best_test_accu = 0


class LSTMLogLoss(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, corpus_size, labels_size):
        super(LSTMLogLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_label = nn.Linear(hidden_dim, labels_size)
        
    def forward(self, word_idxs):
        N = len(word_idxs)
        embeds = self.word_embeddings(word_idxs)
        lstm_out, _ = self.lstm(embeds.view(N, 1, -1))
        scores = self.hidden_to_label(lstm_out.view(N, -1))
        return scores

class LSTMBinaryLogLoss(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, corpus_size):
        super(LSTMBinaryLogLoss, self).__init__()
        self.hidden_dim = hidden_dim
        self.in_word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        
    def forward(self, word_idxs, ):
        N = len(word_idxs)
        in_embeds = self.in_word_embeddings(word_idxs)
        lstm_out, _ = self.lstm(in_embeds.view(N, 1, -1))
        hidden_state = lstm_out.view(N, -1)
        
        return hidden_state

class BinaryLogLoss(nn.Module):
    def __init__(self, corpus_size, hidden_dim, neg_distr, num_neg_samples):
        super(BinaryLogLoss, self).__init__()
        self.out_word_embeddings = nn.Embedding(corpus_size, hidden_dim)
        self.log_sig = nn.LogSigmoid()
        self.neg_distr = neg_distr
        self.num_neg_samples = num_neg_samples
    def forward(self, hidden_state, label_idxex):
        out_embeds = self.out_word_embeddings(label_idxex)
        scores = (hidden_state * out_embeds).sum(dim=1)
        sig_scores = self.log_sig(scores)
        total_score = sig_scores.sum()
        
        N = len(label_idxex)
        for i in range(N):
            neg_idxes = self.neg_distr.sampling(self.num_neg_samples)
            neg_embeds = self.out_word_embeddings(neg_idxes)
            hidden = hidden_state[i]
            neg_score = -np.dot(neg_embeds, hidden) #note:1-sigmod(x)=sigmod(-x)
            neg_sig_scores = (self.log_sig(neg_score)).mean()
            total_score += neg_sig_scores
        
        return -total_score

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
        return np.random.choice(self.word_idxes, N)

class UnigfDistr(Distribution):
    def __init__(self, idxes, weights, f):
        super(UniformDistr, self).__init__()
        self.idxes = idxes
        num_words = len(idxes)
        for i in range(num_words):
            freq = word_freqs[idxes[i]]
            weights[i] = freq**f
        probs = normalize_weights(weights)
        self.probs = probs
    
    def sampling(self, N):
        return np.random.choice(self.idxes, N, p = self.probs)

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
        idx += 1

def load_data(path, ret_data):
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
        item = (X, Y)
        ret_data.append(item)

def evaluate_dataset(model, data):
    num_items = 0
    num_correct = 0
    for X, Y in data:
        ts_X = torch.tensor(X, dtype = torch.long)
        ts_Y = torch.tensor(Y, dtype = torch.long)
        Y_pred = model(ts_X).argmax(1)
        num_items += ts_X.shape[0]
        num_correct += (ts_Y == Y_pred).sum().item()
    return num_correct / num_items 

def evaluate(epoc, model):
    global best_eval_accu
    global best_test_accu
    model.eval()
    eval_ratio = evaluate_dataset(model, eval_data)
    if (eval_ratio > best_eval_accu):
        best_eval_accu = eval_ratio
        print("[epoc=%d][eval accuracy]=%.2f" %(epoc, eval_ratio))
        test_ratio = evaluate_dataset(model, test_data)
        if (test_ratio > best_test_accu):
            best_test_accu = test_ratio
            print("[epoc=%d][test accuracy]=%.2f" %(epoc, test_ratio))
    model.train()

def train(model, loss_fn):
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = learing_rate)
    N = len(training_data)
    M = 3500
    for epoc in range(EPOCS):
        if epoc > 0:
            random.shuffle(training_data)
        itr = 0
        for X, Y in training_data:
            model.zero_grad()
            ts_X = torch.tensor(X, dtype = torch.long)
            ts_Y = torch.tensor(Y, dtype = torch.long)
            label_scores = model(ts_X)
            loss = loss_fn(label_scores, ts_Y)
            loss.backward()
            optimizer.step()
            itr += 1
            if (itr % M == 0 or itr == N):
                print(epoc, loss.item())
                evaluate(epoc, model)
    
    print("best_test_accu=%.2f" %(best_test_accu))        

def main():
    torch.manual_seed(1)
    load_corpus()
    load_data("./31210-s19-hw1/bobsue.lm.train.txt", training_data)
    load_data("./31210-s19-hw1/bobsue.lm.dev.txt", eval_data)
    load_data("./31210-s19-hw1/bobsue.lm.test.txt", test_data)
    
    corpus_size = len(word_to_idx)
    labels_size = corpus_size
    
    model_1 =  LSTMLogLoss(IN_EMBEDDING_DIM, HIDDEN_DIM, corpus_size, labels_size)
    loss_fn_1 = nn.CrossEntropyLoss()
    train(model_1, loss_fn_1)
    
    word_idxes = list(word_to_idx.keys())
    model_2 =  LSTMBinaryLogLoss(OUT_EMBEDDING_DIM, HIDDEN_DIM, corpus_size)
    neg_distr = UniformDistr(word_idxes)
    loss_fn_2 = BinaryLogLoss(corpus_size, HIDDEN_DIM, neg_distr, 20)
    train(model_2, loss_fn_2)
    

if __name__ == '__main__':
    main()