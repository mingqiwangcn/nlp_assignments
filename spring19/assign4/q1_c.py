import hmm
import numpy as np
import time

def norm_vec_probs(prob_weights, beta):
    prob_sum = np.sum(prob_weights)
    probs = prob_weights / prob_sum
    if beta != 1:
        power_weights = np.power(probs, beta)
        power_prob_sum = np.sum(power_weights)
        probs = power_weights / power_prob_sum
        
    return probs

def gibbs_sampling_tags(Xs, K, beta, all_samples= None):
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
        y_probs[0,:] = norm_vec_probs(y_probs[0,:], beta)
        #sampling Y_0
        Ys[0] = np.random.choice(M, 1, replace=True, p=y_probs[0])
        
        for t in range(1, T-1):
            #compute probs of Y_1,...,Y_{T-2}
            y_probs[t,:] = np.log(hmm.transition_probs[Ys[t-1],:]) + \
                           np.log(hmm.emission_probs[:, Xs[t]]) + \
                           np.log(hmm.transition_probs[:,Ys[t+1]])
            y_probs[t,:] = np.exp(y_probs[t,:])
            y_probs[t,:] = norm_vec_probs(y_probs[t,:], beta)
            #sampling Y_1,...,Y_{T-2}
            Ys[t] = np.random.choice(M, 1, replace=True, p=y_probs[t])
        
        #compute probs of Y_{T-1}
        if T > 1:
            y_probs[T-1,:] = np.log(hmm.transition_probs[Ys[T-2],:]) + \
                             np.log(hmm.emission_probs[:, Xs[T-1]]) + \
                             np.log(hmm.stop_probs[:])
            y_probs[T-1,:] = np.exp(y_probs[T-1,:])
            y_probs[T-1,:] = norm_vec_probs(y_probs[T-1,:], beta)
            #sampling Y_{T-1}
            Ys[T-1] = np.random.choice(M, 1, replace=True, p=y_probs[T-1,:])
        
        if not all_samples is None:
            all_samples.append(Ys)
            
    return Ys 

def sampling(K, beta):
    predict_data = []
    for Xs, _ in hmm.dev_data:
        Ys = gibbs_sampling_tags(Xs, K, beta)
        predict_data.append((Xs, Ys))
    return predict_data

def main():
    hmm.prepare()
    N = len(hmm.dev_data)
    for K in (2,5,10,50,100,500,1000):
        t1 = time.time()
        predict_data = sampling(K, 1)
        t2 = time.time()
        total_T = 0
        total_match = 0
        for i in range(N):
            Xs, labels = hmm.dev_data[i]
            _, Ys = predict_data[i]
            T = len(Xs)
            num_match = np.sum(np.array(Ys) == np.array(labels))
            total_T += T
            total_match += num_match
            
        accuracy = total_match / total_T
        log_prob = hmm.compute_log_prob(predict_data)
        print("K=%d, Accuracy=%.6f, time=%.1fs, log probability=%.3f" \
              %(K, accuracy, t2-t1, log_prob))
    
if __name__ == '__main__':
    main()
    