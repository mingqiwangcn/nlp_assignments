import time
import numpy as np
import sys

num_seg = 0

def load_characters(path, gamma):
    file = open(path, "r")
    probs = [1.0 - gamma, gamma]
    Xs = []
    for line in file:
        X = line.rstrip()
        N = len(X)
        b = np.random.choice([0, 1], N, p=probs)
        b[N-1] = 1
        Xs.append((X, b))
    return Xs

def load_labels(path):
    file = open(path, "r")
    bs = []
    for line in file:
        label = line.rstrip()
        b = np.array([int(ch) for ch in label])
        bs.append(b)
    return bs

def preprocess(Xs):
    char_dict = {}
    seg_dict = {}
    for X, b in Xs:
        for ch in X:
            if not ch in char_dict:
                char_dict[ch] = 1
            else:
                char_dict[ch] += 1
        N = len(X)
        i = 0
        j = 0
        while (j < N):
            if b[j] == 1:
                y = X[i:j+1]
                seg_dict[y] = (1 if not y in seg_dict else seg_dict[y] + 1) 
                i = j + 1
            j += 1
    return char_dict, seg_dict

def G0(y, beta, char_probs):
    N = len(y)
    log_prob = (N - 1) * np.log(1.0 - beta) + np.log(beta)
    for c in y:
        log_prob += np.log(char_probs[c])
    return np.exp(log_prob)

def compute_range(b, i):
    r1 = i
    while (r1 > 0) and (b[r1 - 1] == 0):
        r1 -= 1
    r2 = i
    N = len(b)
    while (r2 < N - 1) and (b[r2 + 1] == 0):
        r2 += 1
    r2 += 1
    return r1, r2

def choose_0_prob(X, b, i, r1, r2, s, seg_dict, beta, char_probs):
    changed = (b[i] == 1)
    n_full_o = 0
    num_seg_o = 0
    y_full = X[r1:r2+1]
    if y_full in seg_dict:
        n_full_o = seg_dict[y_full]
    if changed:
        num_seg_o = num_seg - 2
    else:
        num_seg_o = num_seg
    prob = (n_full_o + s * G0(y_full, beta, char_probs)) / (num_seg_o + s)
    
    return prob 

def choose_1_prob(X, b, i, r1, r2, s, seg_dict, beta, gamma, char_probs):
    changed = (b[i] == 0)
    y_prev = X[r1:i+1]
    y_next = X[i+1:r2+1]
    num_y_prev_o = 0
    if y_prev in seg_dict:
        seg_dict[y_prev]
    num_y_next_o = 0
    if y_next in seg_dict:
        num_y_next_o = seg_dict[y_next]
        
    num_seg_o = num_seg
    if changed:
        num_seg_o = num_seg - 1
        
    f1 = (num_y_prev_o + s * G0(y_prev, beta, char_probs)) / (num_seg_o + s)
    f2 = 1 - gamma
    f3_1 = num_y_next_o + (1 if y_prev == y_next else 0) + \
                    s * G0(y_next, beta, char_probs)
    f3_2 = (num_seg_o + 1 + s)
    
    log_prob_sum = np.log([f1, f2, f3_1/f3_2]).sum()
    prob = np.exp(log_prob_sum)
    return prob

def sampling(X, b, i, s, seg_dict, beta, gamma, char_probs):
    global num_seg
    r1, r2 = compute_range(b, i)    
    prob_0 = choose_0_prob(X, b, i, r1, r2, s, seg_dict, beta, char_probs)
    prob_1 = choose_1_prob(X, b, i, r1, r2, s, seg_dict, beta, gamma, char_probs)
    p0 = prob_0/ (prob_0 + prob_1)
    probs = [p0, 1.0 - p0]
    sample = np.random.choice([0, 1], 1, p = probs)
    prev_val = b[i]
    b[i] = sample
    if (prev_val != sample):
        y_prev = X[r1:i+1]
        y_next = X[i+1:r2+1]
        y_full = X[r1:r2+1]
        if prev_val == 1 and sample == 0:
            seg_dict[y_prev] -= 1
            seg_dict[y_next] -= 1
            seg_dict[y_full] = 1 if (not y_full in seg_dict) else seg_dict[y_full] + 1 
            num_seg -= 1
        elif(prev_val == 0 and sample == 1):
            seg_dict[y_full] -= 1
            seg_dict[y_prev] = 1 if (not y_prev in seg_dict) else seg_dict[y_prev] + 1
            seg_dict[y_next] = 1 if (not y_next in seg_dict) else seg_dict[y_next] + 1
            num_seg += 1


def evaluate(Xs, labels):
    M = len(Xs)
    total = 0
    correct = 0
    for i in range(M):
        b = Xs[i][1][:-1]
        label = labels[i][:-1]
        total += len(b)
        correct += np.sum(b == label)
    return total, correct
        
def main():
    num_itr = 10
    beta = 0.5
    gamma = 0.5
    s = 1
    if len(sys.argv) > 4:
        num_itr = int(sys.argv[1])
        beta = int(sys.argv[2])
        gamma = int(sys.argv[3])
        s = int(sys.argv[4])
         
    np.random.seed(100)
    Xs = load_characters("./31210-s19-hw5/cbt-characters.txt", gamma)
    labels = load_labels("./31210-s19-hw5/cbt-boundaries.txt")
    char_dict, seg_dict = preprocess(Xs)
    global num_seg
    num_seg = len(seg_dict)
    
    N = len(char_dict)
    char_probs = dict.fromkeys(char_dict.keys(), 1.0 / N)
    
    for itr in range(num_itr):
        for X, b in Xs:
            N = len(Xs)
            for i in range(N - 1):
                sampling(X, b, i, s, seg_dict, beta, gamma, char_probs)
                
        total, correct = evaluate(Xs, labels)
        accuracy = np.round(correct / total, 6)
        print("itr=%d  accuracy=%d/%d=%.6f" %(itr, correct, total, accuracy))
        
if __name__ == '__main__':
    main()