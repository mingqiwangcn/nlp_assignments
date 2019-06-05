import hmm
import numpy as np
import q1_c as sampler
import time

def main():
    hmm.prepare()
    N = len(hmm.dev_data)
    for K in (2,5,10,50,100,500,1000):
        for beta, step_beta in ((0.1, 0.1), (0.1, 0.05), (0.1, 0.2)):
            t1 = time.time()
            predict_data = sampler.sampling(K, beta, step_beta)
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
            print("K=%d, beta=%.1f, step_beta=%.2f, accuracy=%.6f, time=%.1fs, log probability=%.3f" \
                  %(K, beta, step_beta, accuracy, t2-t1, log_prob))
    
if __name__ == '__main__':
    main()
    