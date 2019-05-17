import hmm
import time
def main():
    hmm.prepare()
    t1 = time.time()
    predict_data, accuracy = hmm.local_predict()
    t2 = time.time()
    print("accuracy:%.6f  time: %.3f" %(accuracy, t2-t1))
    
    hmm.print_dev_log_prob(predict_data)
    
if __name__ == '__main__':
    main()