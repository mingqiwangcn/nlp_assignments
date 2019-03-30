import random
g_features = dict()
g_train_data = []
g_labels = None
g_eval_data = []

def load_data(file_path, obj_data, obj_labels, obj_features):
    file = open(file_path, "r")
    for line in file:
        words = line.split()
        num_words_line = len(words)
        label = words[num_words_line - 1]
        
        if obj_labels is not None:
            if not label in obj_labels:
                obj_labels[label] = 1
            
        X = (label, dict())
        for i in range(num_words_line - 1):
            if not words[i] in X[1]:
                X[1][words[i]] = 1
            if obj_features is not None:
                feature = (label, words[i])
                if not feature in obj_features:
                    obj_features[feature] = 0.0 # featue:weight
        
        obj_data.append(X)    
    file.close()

def score(X, y):
    total = 0.0
    for feature in g_features:
        f =  (feature[0] == X[0] and feature[1] in X[1])
        w = g_features[feature];
        total += f * w
    return total

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
            j = 0
            for feature in g_features:
                j += 1
                w = g_features[feature]
                is_word_contained = (feature[1] in X[1])
                f =  (feature[0] == X[0] and is_word_contained)
                label_class = predict(X)
                f_class = (feature[0] == label_class and is_word_contained)
                w = w + eta * f - eta * f_class
                g_features[feature] = w
                
                if (j % 1000) == 0:
                    print("j=" + str(j))
                
            if (i%20000) == 0:
                evaluate(epoc)
        evaluate(epoc)        
def main():
    global g_labels
    random.seed(100)
    g_labels = dict()
    load_data("./data/sst3/sst3.train", g_train_data, g_labels, g_features)
    g_labels = list(g_labels.keys())
    load_data("./data/sst3/sst3.dev", g_eval_data, None, None)
    train()
    return

if __name__ == '__main__':
    main()