import random
import numpy as np

g_features = []
g_feature_words = []
g_feature_labels = []
g_num_features = 0
g_X_features = None
g_weights = None
g_train_data = []
g_labels = []
g_eval_data = []

def compute_x_features():
    global g_X_features
    N = len(g_train_data)
    M = len(g_labels) 
    D = g_num_features 
    g_X_features = np.ndarray(shape = (N, M, D), dtype = int)
    
    g_feature_labels = [None] * D
    g_feature_words = [None] * D
    
    for i in range(D):
        g_feature_labels[i] = g_features[i][0]
        g_feature_words[i] = g_features[i][1]
    
    print("compute_x_features")
    for i in range(N):
        if (i%1000) == 0:
            print("i=" + str(i))
        X = g_train_data[i]
        for y in g_labels:
            label_feature = np.equal(g_feature_labels, y)
            word_feature = np.isin(g_feature_words, X[1])
            g_X_features[i,y] = np.logical_and(label_feature, word_feature)    
        
    

def load_data(file_path, obj_data, obj_labels, obj_features):
    file = open(file_path, "r")
    idx = 0
    for line in file:
        words = line.split()
        num_words_line = len(words)
        label = int(words[num_words_line - 1])
        
        if obj_labels is not None:
            if not label in obj_labels:
                obj_labels[label] = 1
                g_labels.append(label)
            
        X = [label, dict(), idx]
        for i in range(num_words_line - 1):
            if not words[i] in X[1]:
                X[1][words[i]] = 1
            if obj_features is not None:
                feature = (label, words[i])
                if not feature in obj_features:
                    obj_features[feature] = 1
                    g_features.append(feature)
        X[1] = list(X[1].keys())
        obj_data.append(X)
        idx += 1
    file.close()

def score(X, y):
    idx = X[2]
    f = g_X_features[idx, y]
    ret = np.dot(f, g_weights) 
    return ret

def predict(X):
    scores = []
    for label in g_labels:
        scores.append(score(X, label))
    index = scores.index(max(scores))
    return g_labels[index]

def evaluate(epoc):
    N = len(g_eval_data)
    num_correct = 0
    for i in range(N):
        X = g_eval_data[i]
        y = predict(X)
        if y == X[0]:
            num_correct += 1
    correct_ratio = (num_correct / N) * 100
    print("epoc=" + str(epoc) + " correct=" + str(correct_ratio) + "%")
    
def train():
    MAX_EPOC = 20
    N = len(g_train_data)
    eta = 0.01
    for epoc in range(MAX_EPOC):
        if epoc > 0:
            random.shuffle(g_train_data)
        for i in range(N):
            print("i=" + str(i))
            X = g_train_data[i]
            idx = X[2]
            y = X[0]
            j = 0
            for j in range(g_num_features):
                w = g_weights[i]
                f =  g_X_features[idx, y, j]
                label_class = predict(X)
                f_class = g_X_features[idx, label_class, j]
                w = w + eta * f - eta * f_class
                g_weights[i] = w
                
                if (j % 1000) == 0:
                    print("j=" + str(j))
                
            if (i%20000) == 0:
                evaluate(epoc)
        evaluate(epoc)        

def main():
    global g_weights
    global g_num_features
    random.seed(100)
    
    obj_labels = dict()
    obj_features = dict()
    load_data("./data/sst3/sst3.train", g_train_data, obj_labels, obj_features)
    obj_labels = None
    obj_features = None
    g_labels.sort()
    g_num_features = len(g_features)
    g_weights = [0.0] * g_num_features
    compute_x_features()
    
    load_data("./data/sst3/sst3.dev", g_eval_data, None, None)
    train()
    return

if __name__ == '__main__':
    main()