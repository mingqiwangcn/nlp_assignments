import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EMBEDDING_DIM = 64
HIDDEN_DIM = 128

EPOCS = 10
word_to_idx = {}
training_data = []
eval_data = []
test_data = []
best_eval_accu = 0
best_test_accu = 0


class LSTMPredictor(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, corpus_size, labels_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(corpus_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden_to_label = nn.Linear(hidden_dim, labels_size)
        
    def forward(self, word_idxs):
        N = len(word_idxs)
        embeds = self.word_embeddings(word_idxs)
        lstm_out, _ = self.lstm(embeds.view(N, 1, -1))
        label_space = self.hidden_to_label(lstm_out.view(N, -1))
        label_scores = F.log_softmax(label_space, dim=1)
        return label_scores
    
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

def train():
    corpus_size = len(word_to_idx)
    labels_size = corpus_size
    model = LSTMPredictor(EMBEDDING_DIM, HIDDEN_DIM, corpus_size, labels_size)
    loss_fn = nn.NLLLoss()
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = learing_rate)
    N = len(training_data)
    M = 5000
    for epoc in range(EPOCS):
        if epoc > 0:
            random.shuffle(training_data)
        iter = 0
        for X, Y in training_data:
            model.zero_grad()
            ts_X = torch.tensor(X, dtype = torch.long)
            ts_Y = torch.tensor(Y, dtype = torch.long)
            label_scores = model(ts_X)
            loss = loss_fn(label_scores, ts_Y)
            loss.backward()
            optimizer.step()
            iter += 1
            if (iter % M == 0 or iter == N):
                print(epoc, loss.item())
                evaluate(epoc, model)
    
    print("best_test_accu=%.2f" %(best_test_accu))        

def main():
    torch.manual_seed(1)
    load_corpus()
    load_data("./31210-s19-hw1/bobsue.lm.train.txt", training_data)
    load_data("./31210-s19-hw1/bobsue.lm.dev.txt", eval_data)
    load_data("./31210-s19-hw1/bobsue.lm.test.txt", test_data)
    train()

if __name__ == '__main__':
    main()