import hmm
import numpy as np

def norm_vec_probs(prob_weights):
    prob_sum = np.sum(prob_weights)
    probs = prob_weights / prob_sum
    return probs

def gibbs_sampling_tags(Xs, labels, K):
    T = len(Xs)
    M = len(hmm.all_tags)
    y_probs = np.zeros(T, M) 
    Ys = np.random.choice(M, T)
    for itr in range(K):
        num_y_changed = 0
        #compute probs of Y_0
        y_probs[0,:] = np.log(hmm.start_probs[:]) + \
                       np.log(hmm.emission_probs[:, Xs[0]]) + \
                       np.log(hmm.transition_probs[:,Ys[1]])
        y_probs[0,:] = np.exp(y_probs[0,:])
        y_probs[0,:] = norm_vec_probs(y_probs[0,:])
        #sampling Y_0
        y_pre = Ys[0]
        Ys[0] = np.random.choice(M, 1, replace=True, p=y_probs[0])
        if Ys[0] != y_pre:
            num_y_changed += 1 
        
        for t in range(1, T-1):
            #compute probs of Y_1,...,Y_{T-2}
            y_probs[t,:] = np.log(hmm.transition_probs[Ys[t-1],:]) + \
                           np.log(hmm.emission_probs[:, Xs[t]]) + \
                           np.log(hmm.transition_probs[:,Ys[t+1]])
            y_probs[t,:] = np.exp(y_probs[t,:])
            y_probs[t,:] = norm_vec_probs(y_probs[t,:])
            #sampling Y_1,...,Y_{T-2}
            y_pre = Ys[t] 
            Ys[t] = np.random.choice(M, 1, replace=True, p=y_probs[t])
            if Ys[t] != y_pre:
                num_y_changed += 1
        
        #compute probs of Y_{T-1}
        y_probs[T-1,:] = np.log(hmm.transition_probs[Ys[T-2],:]) + \
                         np.log(hmm.emission_probs[:, Xs[T-1]]) + \
                         np.log(hmm.stop_probs[:])
        y_probs[T-1,:] = np.exp(y_probs[T-1,:])
        y_probs[T-1,:] = norm_vec_probs(y_probs[T-1,:])
        #sampling Y_{T-1}
        y_pre = Ys[T-1]
        Ys[T-1] = np.random.choice(M, 1, replace=True, p=y_probs[T-1,:])
        if Ys[T-1] != y_pre:
            num_y_changed += 1
            
        print("itr=%d, number of changed Ys=%d" %(itr, num_y_changed))
    
    num_match = np.sum(np.array(Ys) == np.array(labels))
    return T, num_match
     

def sampling(K):
    total_T = 0
    total_match = 0
    for Xs, labels in hmm.dev_data:
        T, num_match = gibbs_sampling_tags(Xs, labels, K)
        total_T += T
        total_match += num_match
    
    accuracy = total_match / total_T
    print(accuracy)
    
if __name__ == '__main__':
    