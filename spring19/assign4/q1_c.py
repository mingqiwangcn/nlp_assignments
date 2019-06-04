import hmm
import numpy as np
import time

def norm_vec_probs(prob_weights):
    prob_sum = np.sum(prob_weights)
    probs = prob_weights / prob_sum
    return probs

def gibbs_sampling_tags(Xs, K):
    T = len(Xs)
    M = len(hmm.all_tags)
    y_probs = np.zeros((T, M)) 
    Ys = np.random.choice(M, T)
    for itr in range(K):
        #compute probs of Y_0
        y_probs[0,:] = np.log(hmm.start_probs[:]) + \
                       np.log(hmm.emission_probs[:, Xs[0]]) + \
                       np.log(hmm.transition_probs[:,Ys[1]]) if T > 1 else np.log(hmm.stop_probs[:])
        y_probs[0,:] = np.exp(y_probs[0,:])
        y_probs[0,:] = norm_vec_probs(y_probs[0,:])
        #sampling Y_0
        Ys[0] = np.random.choice(M, 1, replace=True, p=y_probs[0])
        
        for t in range(1, T-1):
            #compute probs of Y_1,...,Y_{T-2}
            y_probs[t,:] = np.log(hmm.transition_probs[Ys[t-1],:]) + \
                           np.log(hmm.emission_probs[:, Xs[t]]) + \
                           np.log(hmm.transition_probs[:,Ys[t+1]])
            y_probs[t,:] = np.exp(y_probs[t,:])
            y_probs[t,:] = norm_vec_probs(y_probs[t,:])
            #sampling Y_1,...,Y_{T-2}
            Ys[t] = np.random.choice(M, 1, replace=True, p=y_probs[t])
        
        #compute probs of Y_{T-1}
        if T > 1:
            y_probs[T-1,:] = np.log(hmm.transition_probs[Ys[T-2],:]) + \
                             np.log(hmm.emission_probs[:, Xs[T-1]]) + \
                             np.log(hmm.stop_probs[:])
            y_probs[T-1,:] = np.exp(y_probs[T-1,:])
            y_probs[T-1,:] = norm_vec_probs(y_probs[T-1,:])
            #sampling Y_{T-1}
            Ys[T-1] = np.random.choice(M, 1, replace=True, p=y_probs[T-1,:])
            
    return Ys 
     

def sampling(K):
    total_T = 0
    total_match = 0
    t1 = time.time()
    predict_data = []
    for Xs, labels in hmm.dev_data:
        T = len(Xs)
        Ys = gibbs_sampling_tags(Xs, K)
        predict_data.append((Xs, Ys))
        num_match = np.sum(np.array(Ys) == np.array(labels))
        total_T += T
        total_match += num_match
    accuracy = total_match / total_T
    t2 = time.time()
    log_prob = hmm.compute_log_prob(predict_data)
    print("K=%d, Accuracy=%.6f, time=%.1fs, log probability=%.3f" \
          %(K, accuracy, t2-t1, log_prob))

def main():
    hmm.prepare()
    for K in (2,5,10,50,100,500,1000):
        sampling(K)
    
if __name__ == '__main__':
    main()
    