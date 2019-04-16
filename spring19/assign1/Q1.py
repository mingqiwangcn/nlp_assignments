import lm as lm
import torch.nn as nn
import time
import sys

def main():
    epocs = 1
    if len(sys.argv) > 1:
        epocs = int(sys.argv[1])
    t1 = time.time()
    lm.show_errors = True
    lm.load_data()
    corpus_size = len(lm.word_to_idx)
    labels_size = corpus_size
    model =  lm.LSTMLogLoss(lm.IN_EMBEDDING_DIM, lm.HIDDEN_DIM, corpus_size, labels_size)
    loss_fn = nn.CrossEntropyLoss()
    lm.eval_lm(model, loss_fn, epocs)
    t2 = time.time()
    print("Q1 time:%.3f" %(t2-t1))
    error_pairs = lm.list_errors(35)
    print("Top 35 errors")
    print(error_pairs)
    
if __name__ == '__main__':
    main()