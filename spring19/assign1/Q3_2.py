import lm as lm
import time
import sys

def main():
    epocs = 1
    r = 20
    if len(sys.argv) > 2:
        epocs = int(sys.argv[1])
        r = int(sys.argv[2])  
    t1 = time.time()
    lm.load_data()
    word_idxes = list(lm.word_to_idx.values())
    model =  lm.LSTMBinaryLogLoss(lm.OUT_EMBEDDING_DIM, lm.HIDDEN_DIM, word_idxes)
    neg_distr = lm.UniformDistr(word_idxes)
    
    loss_fn = lm.BinaryLogLoss(model.out_word_embeddings, neg_distr, r)
    lm.eval_lm(model, loss_fn, epocs)
    t2 = time.time()
    print("Q3_2 r=%d time=%.3f" %(r, t2-t1))
if __name__ == '__main__':
    main()