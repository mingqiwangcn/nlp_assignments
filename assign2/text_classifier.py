import random
g_features = dict()
g_train_data = []
g_labels = []
g_eval_data = []
g_test_data= []
g_best_eval_accu = 0
g_best_test_accu = 0

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
            feature = (label, words[i])
            if not feature in X[1]:
                X[1][feature] = 1
                
            if obj_features is not None:
                if not feature in obj_features:
                    obj_features[feature] = 0.0 #feature weight
                    
        obj_data.append(X)
        idx += 1
    file.close()

def score(X, y):
    label = X[0]
    features = X[1].keys()
    ret = 0.0
    for f in features:
        if (label == y):
            if f in g_features:
                ret += g_features[f]
    return ret

def predict(X):
    scores = []
    for label in g_labels:
        scores.append(score(X, label))
    index = scores.index(max(scores))
    return g_labels[index]

def eval_data(epoc, obj_data, dataset_name):
    N = len(obj_data)
    num_correct = 0
    for i in range(N):
        X = obj_data[i]
        y = predict(X)
        if y == X[0]:
            num_correct += 1
    correct_ratio = (num_correct / N) * 100
    print("[epoc]=%2d [%8s accuracy]=%.2f%%" %(epoc, dataset_name, correct_ratio))
    return correct_ratio

def evaluate(epoc):
    global g_best_eval_accu
    global g_best_test_accu
    correct_ratio = eval_data(epoc, g_eval_data, "dev")
    if (correct_ratio > g_best_eval_accu):
        g_best_eval_accu = correct_ratio
        test_ratio = eval_data(epoc, g_eval_data, "devtest")
        if (test_ratio > g_best_test_accu):
            g_best_test_accu = test_ratio
def train():
    global g_weights
    MAX_EPOC = 20
    N = len(g_train_data)
    eta = 0.01
    for epoc in range(MAX_EPOC):
        if epoc > 0:
            random.shuffle(g_train_data)
        for i in range(N):
            X = g_train_data[i]
            y = X[0]
            label_class = predict(X)
            features = X[1].keys()
            all_features = dict()
            for f in features:
                feature = (f[0], f[1])
                if not feature in all_features:
                    all_features[feature] = 1
                feature = (label_class, f[1])
                if (feature in g_features) and (not feature in all_features):
                    all_features[feature] = 1
            
            for f in all_features.keys():
                w = g_features[f]
                f1_val = 1
                if f[0] != y:
                    f1_val = 0
                f2_val = 1
                if (f[0] != label_class):
                    f2_val = 0
                w = w + eta * f1_val - eta * f2_val
                g_features[f] = w
                
            
            if i > 0 and (i%20000 == 0):
                evaluate(epoc)
                
        evaluate(epoc)        

def main():
    random.seed(100)
    
    obj_labels = dict()
    load_data("./data/sst3/sst3.train", g_train_data, obj_labels, g_features)
    obj_labels = None
    g_labels.sort()
    
    load_data("./data/sst3/sst3.dev", g_eval_data, None, None)
    load_data("./data/sst3/sst3.devtest", g_test_data, None, None)
    
    train()
    
    print("[best test accuracy] = %.2f%%" %(g_best_test_accu))
    
    return

if __name__ == '__main__':
    main()