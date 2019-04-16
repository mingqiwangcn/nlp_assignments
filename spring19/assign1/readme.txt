1) We use python3.7.3 + pytorch stable 1.0

2) Word embeddings and lstm cell/hidden dimensionality: 200

3) All the modle code is in lm.py
   Model class for log loss:
   		LSTMLogLoss: the forward method takes 'start_pos' as input to implement Q4
   Model class for binary log loss: 
   		LSTMBinaryLogLoss
   		BinaryLogLoss
   		Distribution
   		UniformDistr
   		UnigfDistr
   Method for EVALLM:
   		eval_lm(model, loss_fn, epocs)

4) Separate driver code files: Q1.py, Q3_2.py, Q3_3.py and Q4_1.py

5) Dataset
   download the dataset zip file in the directory where *.py exists and unzip it with direcotry structure:
   *.py
   31210-s19-hw1
		*.txt
		*.tsv

6) Command line arguments to run the code
   python Q1.py 	[#epoch]
   		exaples: python Q1.py 10
   python Q3_2.py 	[#epoch] [#negtive samples]
		exaples: python Q3_2.py 20 20
   python Q3_3.py 	[#epoch] [#negtive samples]	[f]
		exaples: python Q3_2.py 20 20 0.1


