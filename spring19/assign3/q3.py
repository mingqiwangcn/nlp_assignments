import hmm
import time
def main():
    hmm.prepare()
    t1 = time.time()
    predict_data, accuracy = hmm.greedy_left_right()
    t2 = time.time()
    print("greedy_left_right accuracy:%.6f  time: %.3f" %(accuracy, t2-t1))
    
    hmm.print_dev_log_prob(predict_data)
    
if __name__ == '__main__':
    main()