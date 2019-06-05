import hmm
import numpy as np
import q1_c as sampler
import time

def mbr_infer(K, beta):
    predict_data = []
    for Xs, _ in hmm.dev_data:
        all_samples = []
        sampler.gibbs_sampling_tags(Xs, K, beta, all_samples, None)
        sample_data = np.array(all_samples)
        N = len(Xs)
        Ys = []
        for i in range(N):
            (y_values, y_counts) = np.unique(sample_data[:,i], return_counts=True)
            idx = np.argmax(y_counts)
            y = y_values[idx]
            Ys.append(y)
            
        predict_data.append((Xs,Ys))
    return predict_data

def main():
    hmm.prepare()
    N = len(hmm.dev_data)
    for K in (2,5,10,50,100,500,1000):
        for beta in (1, 0.5, 2):
            t1 = time.time()
            predict_data = mbr_infer(K, beta)
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
            print("K=%d, beta=%.1f, accuracy=%.6f, time=%.1fs" %(K, beta, accuracy, t2-t1))
    
if __name__ == '__main__':
    main()
    