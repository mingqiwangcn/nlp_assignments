import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

W_D = 50
D_H = 128
MAX_EPOC = 20
g_w = 1
g_context_size = 2 * g_w + 1
D_in = g_context_size * W_D
g_embedding = dict()
g_labels = dict()
g_train_data = ()
g_eval_data = ()
g_test_data = ()

def load_embedding():
    file = open("./data/embeddings-twitter.txt")
    for line in file:
        data = line.split()
        w = data[0]
        j = 1
        embedding = np.ndarray(50, dtype = float)
        for i in  range(50):
            embedding[i] = float(data[j])
            j += 1 
        g_embedding[w] = embedding
    file.close() 

def load_data(file_path, obj_labels):
    file = open(file_path)
    item_lst = []
    for line in file:
        data = line.split()
        if (len(data) >= 2):
            item = (data[0], data[1])
            item_lst.append(item)    
            
    N = len(item_lst)
    if obj_labels is not None:
        idx = 0
        for item in item_lst:
            if not item[1] in obj_labels:
                obj_labels[item[1]] = idx;
                idx += 1
    
    obj_X = torch.empty((N, D_in), dtype=torch.float)
    obj_Y = torch.empty((N), dtype=torch.int)
    word = ""
    for i in range(N):
        idx = 0
        for j in range(i - g_w, i + g_w + 1):
            if j >= 0 and j < N:
                word = item_lst[j][0]
            else:
                word = "</s>"
            if not word in g_embedding:
                word = "UUUNKKK"
            embedding = g_embedding[word]
            for m in range(W_D):
                obj_X[i, idx + m] = embedding[m]
            idx += W_D
        label = item_lst[i][1]
        obj_Y[i] = g_labels[label]
    return obj_X, obj_Y
    
def train():
    D_out = len(g_labels.values())
    model = nn.Sequential(
        nn.Linear(D_in, D_H),
        nn.Tanh(),
        nn.Linear(D_H, D_out)
        )
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    X = g_train_data[0]
    Y = g_train_data[1]
    N = X.shape[0]
    batch_size = 1
    num_batches = N / batch_size
    if N % batch_size:
        num_batches + 1
        
    for epoc in range(MAX_EPOC):
        if epoc > 0:
            X = X[torch.randperm(X.size()[0])]
        for m in num_batches:
            m1 = m
            m2 = m + batch_size
            if (m == num_batches - 1):
                m2 = num_batches
            X_m = X[m1:m2]
            Y_m = Y[m1:m2] 
            y_pred = model(X_m)
            loss = loss_fn(y_pred, Y_m)
            print(epoc, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
def main():
    load_embedding()
    
    X, Y = load_data("./data/tweet-pos/tweets-d.txt", g_labels)
    g_train_data[0] = X
    g_train_data[1] = Y
    
    X, Y = load_data("./data/tweet-pos/tweets-dev.txt")
    g_eval_data[0] = X
    g_eval_data[1] = Y
    
    X, Y = load_data("./data/tweet-pos/tweets-devtest.txt")
    g_test_data[0] = X
    g_test_data[1] = Y
    
    train()

if __name__ == '__main__':
    main()