import numpy as np

all_words = []
word_to_idx = {}
tag_to_idx = {}
all_tags = []
start_probs = None
stop_probs = None
transition_probs = None
emission_probs = None
lambda_transition = 0.1
lambda_emission = 0.001
train_data = []
dev_data = []
        
def load_dataset(path, ret_data, is_train = False):
    global all_words
    global all_tags
        
    file = open(path, "r")
    x_s = []
    y_s = []
    for line in file:
        tokens = line.split()
        if len(tokens) > 0:
            word = tokens[0].lower()  
            tag = tokens[1]
            if is_train:
                if not word in word_to_idx:
                    idx = len(word_to_idx)
                    word_to_idx[word] = idx
                if not tag in tag_to_idx:
                    idx = len(tag_to_idx)
                    tag_to_idx[tag] = idx
            x = word_to_idx[word]
            y = tag_to_idx[tag]
            x_s.append(x)
            y_s.append(y)
        else:
            pair = (x_s, y_s)
            ret_data.append(pair)
            x_s = []
            y_s = []

    if is_train:
        all_words = [None] * len(word_to_idx)
        all_tags = [None] * len(tag_to_idx)
        for word in word_to_idx:
            all_words[word_to_idx[word]] = word
        for tag in tag_to_idx:
            all_tags[tag_to_idx[tag]] = tag
            
def load_data():
    load_dataset("./31210-s19-hw3/en_ewt.train", train_data, True)
    load_dataset("./31210-s19-hw3/en_ewt.dev", dev_data, False)

def normalize_probs(probs):
    probs_sum = np.sum(probs, axis=1, keepdims=True)
    norm_probs = probs / probs_sum
    norm_probs_sum = np.sum(norm_probs, axis=1)
    probs_diff = norm_probs_sum - 1
    idxes = np.argmax(norm_probs, axis=1)
    for i in range(len(idxes)):
        norm_probs[i][idxes[i]] += probs_diff[i]
    return norm_probs
    
def init_transition_probs():
    global start_probs
    global stop_probs
    global transition_probs
    tag_size = len(tag_to_idx)
    start_probs = np.zeros(tag_size)
    stop_probs = np.zeros(tag_size)
    transition_probs = np.zeros((tag_size, tag_size))
    for pair in train_data:
        y_s = pair[1]
        tags_len = len(y_s)
        start_probs[y_s[0]] += 1
        
        for i in range(1, tags_len):
            idx_1 = y_s[i-1]
            idx_2 = y_s[i]
            transition_probs[idx_1][idx_2] += 1
            
        stop_probs[y_s[tags_len-1]] += 1
    
    start_probs += lambda_transition
    start_norm_probs = normalize_probs(start_probs[np.newaxis, :])
    start_probs = np.squeeze(start_norm_probs, axis=0) 
    
    stop_probs += lambda_transition
    stop_norm_probs = normalize_probs(stop_probs[np.newaxis, :])
    stop_probs = np.squeeze(stop_norm_probs, axis=0)
    
    transition_probs += lambda_transition
    transition_probs = normalize_probs(transition_probs)
    
def init_emission_probs():
    global emission_probs
    tag_size = len(tag_to_idx)
    corpus_size = len(word_to_idx)
    emission_probs = np.zeros((tag_size, corpus_size))
    for pair in train_data:
        x_s = pair[0]
        y_s = pair[1]
        seq_len = len(y_s)
        for i in range(0, seq_len):
            idx_1 = y_s[i]
            idx_2 = x_s[i]
            emission_probs[idx_1][idx_2] += 1
    
    emission_probs += lambda_emission
    emission_probs = normalize_probs(emission_probs)

def print_top_probs():
    probs = start_probs
    K = 5
    top_5_idxes = np.argpartition(probs,-K)[-K:]
    top_5_pairs = []
    for i in range(K):
        idx = top_5_idxes[i]
        pair = (all_tags[idx], probs[idx])
        top_5_pairs.append(pair)
    
    top_5_pairs = sorted(top_5_pairs, key=lambda x: x[1], reverse = True)
    
    print(top_5_pairs)
    
    probs = emission_probs[tag_to_idx["JJ"]]
    K = 10
    top_10_idxes = np.argpartition(probs,-K)[-K:]
    top_10_pairs = []
    for i in range(K):
        idx = top_10_idxes[i]
        pair = (all_words[idx], probs[idx])
        top_10_pairs.append(pair)
    
    top_10_pairs = sorted(top_10_pairs, key=lambda x: x[1], reverse = True)
    print(top_10_pairs)

def print_dev_log_prob(data):
    sum_log_prob = .0
    for pair in data:
        x_s = pair[0]
        y_s = pair[1]
        seq_len = len(x_s)
        sum_log_prob += np.log(start_probs[y_s[0]])
        
        for i in range(1, seq_len):
            idx_1 = y_s[i-1]
            idx_2 = y_s[i]
            sum_log_prob += np.log(transition_probs[idx_1, idx_2])
            
        sum_log_prob += np.log(stop_probs[y_s[seq_len-1]])
        
        for i in range(0, seq_len):
            idx_1 = y_s[i]
            idx_2 = x_s[i]
            sum_log_prob += np.log(emission_probs[idx_1, idx_2])
            
    print("log probability:", sum_log_prob)

def prepare():
    load_data()
    init_transition_probs()
    init_emission_probs()

def local_predict():
    num_tokens = 0
    num_right = 0
    predict_data = []
    for pair in dev_data:
        x_s = pair[0]
        y_s_m = []
        num_tokens += len(x_s)
        for i in range(len(x_s)):
            y_m = np.argmax(emission_probs[:,x_s[i]])
            y_s_m.append(y_m)
            
        predict_data.append((x_s, y_s_m))
        y_s = pair[1]
        diff = np.sum(np.array(y_s) == np.array(y_s_m))
        num_right += np.sum(diff)
        
    accuracy = round(num_right / num_tokens, 6)
    return (predict_data, accuracy)

def greedy_left_right():
    num_tokens = 0
    num_right = 0
    predict_data = []
    for pair in dev_data:
        x_s = pair[0]
        y_s_m = []
        num_tokens += len(x_s)
        tr_probs = start_probs
        for i in range(len(x_s)-1):
            scores = np.log(emission_probs[:,x_s[i]]) + np.log(tr_probs) 
            y_m = np.argmax(scores)
            y_s_m.append(y_m)
            tr_probs = transition_probs[y_m]
        
        scores = np.log(emission_probs[:,x_s[len(x_s)-1]]) + np.log(tr_probs) + np.log(stop_probs) 
        y_m = np.argmax(scores)
        y_s_m.append(y_m)
        
        predict_data.append((x_s, y_s_m))
        y_s = pair[1]
        diff = np.sum(np.array(y_s) == np.array(y_s_m))
        num_right += np.sum(diff)
        
    accuracy = round(num_right / num_tokens, 6)
    return (predict_data, accuracy)

def greedy_right_left():
    num_tokens = 0
    num_right = 0
    predict_data = []
    for pair in dev_data:
        x_s = pair[0]
        y_s_m = []
        num_tokens += len(x_s)
        tr_probs = stop_probs
        for i in range(len(x_s)-1, 0, -1):
            scores = np.log(emission_probs[:,x_s[i]]) + np.log(tr_probs) 
            y_m = np.argmax(scores)
            y_s_m.append(y_m)
            tr_probs = transition_probs[y_m]
        
        scores = np.log(emission_probs[:,0]) + np.log(tr_probs) + np.log(start_probs) 
        y_m = np.argmax(scores)
        y_s_m.append(y_m)
        y_s_m.reverse()
        
        predict_data.append((x_s, y_s_m))
        y_s = pair[1]
        diff = np.sum(np.array(y_s) == np.array(y_s_m))
        num_right += np.sum(diff)
        
    accuracy = round(num_right / num_tokens, 6)
    return (predict_data, accuracy)

def viterbi_infer():
    num_tokens = 0
    num_right = 0
    predict_data = []
    for pair in dev_data:
        x_s = pair[0]
        y_s_m = []
        num_tokens += len(x_s)
        scores = np.zeros((len(x_s), len(all_tags)))
        for i in range(len(all_tags)):
            scores[0,i] = np.log(emission_probs[i, x_s[0]]) + np.log(start_probs[i])
            
        for i in range(1, len(x_s)):    
            for j in range(len(all_tags)):
                tag_scores = np.log(emission_probs[j,x_s[i]]) + np.log(transition_probs[:,j]) + \
                             scores[i-1,:]
                scores[i,j] = np.max(tag_scores)
        
        tag_scores = np.log(stop_probs) + scores[len(x_s)-1,:]
        y_m = np.argmax(tag_scores)
        y_s_m.append(y_m)
        i = len(x_s) - 1
        while i > 0:
            tag_scores = np.log(emission_probs[y_m,x_s[i]]) + \
                         np.log(transition_probs[:,y_m]) + \
                         scores[i-1,:]
            y_m = np.argmax(tag_scores)
            y_s_m.append(y_m)
            i -= 1
            
        y_s_m.reverse()
        predict_data.append((x_s, y_s_m))
        y_s = pair[1]
        diff = np.sum(np.array(y_s) == np.array(y_s_m))
        num_right += np.sum(diff)
        
    accuracy = round(num_right / num_tokens, 6)
    return (predict_data, accuracy)
