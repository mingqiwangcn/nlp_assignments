import lm as lm
import torch.nn as nn
import time

def main():
    t1 = time.time()
    lm.load_prevsent_data()
    corpus_size = len(lm.word_to_idx)
    labels_size = corpus_size
    model =  lm.LSTMLogLoss(lm.IN_EMBEDDING_DIM, lm.HIDDEN_DIM, corpus_size, labels_size)
    loss_fn = nn.CrossEntropyLoss()
    lm.eval_lm(model, loss_fn, epocs = 1)
    t2 = time.time()
    print(t2-t1)
    
if __name__ == '__main__':
    main()