import nlp_assignments.spring19.assign1.lm as lm
import torch.nn as nn

def main():
    lm.load_data()
    
    corpus_size = len(lm.word_to_idx)
    labels_size = corpus_size
    
    model =  lm.LSTMLogLoss(lm.IN_EMBEDDING_DIM, lm.HIDDEN_DIM, corpus_size, labels_size)
    loss_fn = nn.CrossEntropyLoss()
    lm.eval_lm(model, loss_fn, epocs = 10)
    
if __name__ == '__main__':
    main()