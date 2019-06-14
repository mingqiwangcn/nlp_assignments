import time
import numpy as np
import sys
import enum

num_seg = 0
num_b_changed = 0
char_dist_type = None
char_to_idx = {}
idx_to_char = None
lambda_bigram = 0.01


class CharDistribution(enum.Enum):
    UNIFORM = 0
    UNIGRAM = 1
    BIGRAM  = 2 
    
def load_characters(path, gamma):
    file = open(path, "r")
    probs = [1.0 - gamma, gamma]
    Xs = []
    for line in file:
        X = line.rstrip()
        N = len(X)
        b = np.random.choice([0, 1], N, p=probs)
        b[N-1] = 1
        last_info = [[False, None, None]] * N
        Xs.append((X, b, last_info))
    
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
    for X, b, _ in Xs:
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

def process_bigram_probs(Xs, char_dict):
    global idx_to_char
    N = len(char_dict)
    for ch in char_dict:
        char_to_idx[ch] = len(char_to_idx)
    
    idx_to_char = [None] * N
    for ch in char_to_idx:
        idx_to_char[char_to_idx[ch]] = ch
        
    start_probs = np.zeros(N)
    start_probs += lambda_bigram
    bi_probs = np.zeros((N, N))
    bi_probs += lambda_bigram
    for X, _, _ in Xs:
        idx = char_to_idx[X[0]]
        start_probs[idx] += 1
        M = len(X)
        for i in range(1, M):
            idx1 = char_to_idx[X[i-1]]
            idx2 = char_to_idx[X[i]]
            bi_probs[idx1, idx2] += 1
        
    prob_sum = np.sum(start_probs)
    start_probs = start_probs / prob_sum
    
    prob_sum = np.sum(bi_probs, axis=1, keepdims=True)
    bi_probs = bi_probs / prob_sum
    
    return start_probs, bi_probs

def G0(y, beta, char_probs):
    N = len(y)
    log_prob = (N - 1) * np.log(1.0 - beta) + np.log(beta)
    if (char_dist_type != CharDistribution.BIGRAM):
        for c in y:
            log_prob += np.log(char_probs[c])
    else:
        start_probs = char_probs[0]
        bi_probs = char_probs[1]
        idx = char_to_idx[y[0]]
        log_prob += np.log(start_probs[idx])
        for i in range(1, N):
            idx1 = char_to_idx[y[i-1]]
            idx2 = char_to_idx[y[i]]
            log_prob += np.log(bi_probs[idx1, idx2])
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

def choose_0_prob(X, last_info, b, i, r1, r2, s, seg_dict, beta, char_probs):
    changed = (b[i] == 1)
    if (not last_info[i][0]) and (not last_info[i][1] is None):
        return last_info[i][1]
    
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
    
    last_info[i][1] = prob
    
    return prob 

def choose_1_prob(X, last_info, b, i, r1, r2, s, seg_dict, beta, gamma, char_probs):
    changed = (b[i] == 0)
    
    if (not last_info[i][0]) and (not last_info[i][2] is None):
        return last_info[i][2]
    
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
    
    last_info[i][2] = prob
    
    return prob

def sampling(X, last_info, b, i, s, seg_dict, beta, gamma, char_probs, prob_tao):
    global num_seg
    global num_b_changed
    r1, r2 = compute_range(b, i)    
    prob_0 = choose_0_prob(X, last_info, b, i, r1, r2, s, seg_dict, beta, char_probs)
    prob_1 = choose_1_prob(X, last_info, b, i, r1, r2, s, seg_dict, beta, gamma, char_probs)
    p0 = prob_0 / (prob_0 + prob_1)
    probs = [p0, 1.0 - p0]
    if (prob_tao != 1.0):
        prob_0 = np.power(p0, prob_tao)
        prob_1 = np.power(1.0-p0, prob_tao)
        p0 = prob_0 / (prob_0 + prob_1)
        probs = [p0, 1.0 - p0]
        
    sample = np.random.choice([0, 1], 1, p = probs)
    prev_val = b[i]
    if (prev_val != sample):
        y_prev = X[r1:i+1]
        y_next = X[i+1:r2+1]
        y_full = X[r1:r2+1]
        if prev_val == 1 and sample == 0:
            seg_dict[y_prev] -= 1
            seg_dict[y_next] -= 1
            seg_dict[y_full] = 1 if (not y_full in seg_dict) else seg_dict[y_full] + 1 
            num_seg -= 1
            last_info[i][0] = True
            num_b_changed += 1
        elif(prev_val == 0 and sample == 1):
            seg_dict[y_full] -= 1
            seg_dict[y_prev] = 1 if (not y_prev in seg_dict) else seg_dict[y_prev] + 1
            seg_dict[y_next] = 1 if (not y_next in seg_dict) else seg_dict[y_next] + 1
            num_seg += 1
            last_info[i][0] = True
            num_b_changed += 1
            
    b[i] = sample

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
    global num_b_changed
    global char_dist_type
    
    num_itr = 10
    beta = 0.5
    gamma = 0.5
    s = 1.0
    char_dist_type = CharDistribution.UNIFORM
    start_prob_tao = 1.0
    prob_anneal_flag = 0
    num_arg = len(sys.argv)
    if num_arg > 1:
        num_itr = int(sys.argv[1])
    if num_arg > 2:
        beta = float(sys.argv[2])
    if num_arg > 3:
        gamma = float(sys.argv[3])
    if num_arg > 4:
        s = float(sys.argv[4])
    if num_arg > 5:
        dist_tag = int(sys.argv[5])
        if (dist_tag == 1):
            char_dist_type = CharDistribution.UNIGRAM
        elif(dist_tag == 2):
            char_dist_type = CharDistribution.BIGRAM
    if num_arg > 6:
        start_prob_tao = float(sys.argv[6])
    if num_arg > 7:
        prob_anneal_flag = 1
    
    print("beta=%.1f gamma=%.1f s=%.1f char_dist_type=%s start_prob_tao=%.1f" \
          %(beta, gamma, s, str(char_dist_type), start_prob_tao ))
        
    np.random.seed(100)
    Xs = load_characters("./31210-s19-hw5/cbt-characters.txt", gamma)
    labels = load_labels("./31210-s19-hw5/cbt-boundaries.txt")
    char_dict, seg_dict = preprocess(Xs)
    global num_seg
    num_seg = np.sum(list(seg_dict.values()))
    num_char = len(char_dict)
    char_probs = None
    
    if char_dist_type == CharDistribution.UNIFORM:
        char_probs = dict.fromkeys(char_dict.keys(), 1.0 / num_char)
    elif char_dist_type == CharDistribution.UNIGRAM:
        char_probs = {}
        total_char_count = np.sum(list(char_dict.values()))
        for ch in char_dict:
            char_probs[ch] = char_dict[ch] / total_char_count
    elif char_dist_type == CharDistribution.BIGRAM:
        start_probs, bi_probs = process_bigram_probs(Xs, char_dict)
        char_probs = [start_probs, bi_probs]
    
    prob_tao = start_prob_tao
    step_tao = 0
    if (prob_anneal_flag == 1):
        step_tao = (1.0 - prob_tao) / num_itr
        
    for itr in range(num_itr):
        if (prob_anneal_flag == 1):
            print("prob_tao=%.2f" %(prob_tao))
        num_b_changed = 0
        t1 = time.time()
        for X, b, last_info in Xs:
            N = len(X)
            for i in range(N - 1):
                sampling(X, last_info, b, i, s, seg_dict, beta, gamma, char_probs, prob_tao)
                
        t2 = time.time()
        total, correct = evaluate(Xs, labels)
        accuracy = np.round(correct / total, 6)
        print("itr=%d num_changed=%d num_seg=%d accuracy=%d/%d=%.6f time=%.3f" \
               %(itr, num_b_changed, num_seg, correct, total, accuracy, t2-t1))
        
        if (prob_anneal_flag == 1):
            prob_tao += step_tao
        
if __name__ == '__main__':
    main()