import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

W_D = 50
D_H = 128
MAX_EPOC = 3
g_w = 1
g_context_size = 2 * g_w + 1
D_in = g_context_size * W_D
g_embedding = dict()
g_labels = dict()
g_train_data = [None, None]
g_eval_data = [None, None]
g_test_data = [None, None]

g_best_eval_accu = 0
g_best_test_accu = 0

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

def load_data(file_path, obj_labels = None):
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
    
    D_out = len(g_labels.values())
    obj_X = torch.empty((N, D_in), dtype=torch.float)
    obj_Y = torch.empty(N, dtype=torch.long)
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
        str_label = item_lst[i][1]
        label = g_labels[str_label]
        obj_Y[i] = label
        
    return obj_X, obj_Y

def evaluate(model, epoc):
    global g_best_eval_accu
    global g_best_test_accu
    
    e_X = g_eval_data[0]
    e_Y = g_eval_data[1]
    Y_pred = model(e_X).argmax(1)
    num_correct = (e_Y == Y_pred).sum().item()
    N = e_X.shape[0]
    eval_ratio = num_correct / N
    if (eval_ratio > g_best_eval_accu):
        print("[epoc=%d][eval accuracy]=%.2f" %(epoc, eval_ratio))
        g_best_eval_accu = eval_ratio
        t_X = g_test_data[0]
        t_Y = g_test_data[1]
        Y_pred_test = model(t_X).argmax(1)
        num_correct_test = (t_Y == Y_pred_test).sum().item()
        test_ratio = num_correct_test / t_X.shape[0]
        if (test_ratio > g_best_test_accu):
            g_best_test_accu = test_ratio
            print("[epoc=%d][test accuracy]=%.2f" %(epoc, test_ratio))
    
        
def train():
    D_out = len(g_labels.values())
    model = nn.Sequential(
        nn.Linear(D_in, D_H),
        nn.Tanh(),
        nn.Linear(D_H, D_out)
        )
    loss_fn = nn.CrossEntropyLoss()
    learing_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr = learing_rate)
    
    X = g_train_data[0]
    Y = g_train_data[1]
    N = X.shape[0]
    batch_size = 1
    num_batches = int(N / batch_size)
    if N % batch_size:
        num_batches + 1
        
    for epoc in range(MAX_EPOC):
        for m in range(num_batches):
            m1 = m
            m2 = m + batch_size
            if (m == num_batches - 1):
                m2 = num_batches
            X_m = X[m1:m2]
            Y_m = Y[m1:m2] 
            y_pred = model(X_m)
            loss = loss_fn(y_pred, Y_m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (m > 0 and m % 1000 == 0):
                print(epoc, loss.item())
                evaluate(model, epoc)
    
    print("best_test_accu=%.2f" %(g_best_test_accu))
    
def main():
    load_embedding()
    
    X, Y = load_data("./data/tweet-pos/tweets-train.txt", g_labels)
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