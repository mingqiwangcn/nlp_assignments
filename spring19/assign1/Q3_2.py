import lm as lm

def main():
    lm.load_data()
    word_idxes = list(lm.word_to_idx.values())
    model =  lm.LSTMBinaryLogLoss(lm.OUT_EMBEDDING_DIM, lm.HIDDEN_DIM, word_idxes)
    neg_distr = lm.UniformDistr(word_idxes)
    r = 20
    loss_fn = lm.BinaryLogLoss(model.out_word_embeddings, neg_distr, r)
    lm.eval_lm(model, loss_fn, epocs = 20)
    
if __name__ == '__main__':
    main()