import numpy as np
from scipy import stats
import math

dt_r_words = None
dt_c_words = None

def EvalWS(M):
    path = "./data/31190-a1-files/men.txt"
    file = open(path, "r")
    men_similarity = []
    context_similarity = []
    file.readline()
    row = 0;
    for line in file:
        #print(row)
        data = line.split()
        w1 = data[0]
        w2 = data[1]
        score1 = float(data[2])
        men_similarity.append(score1)
        r1 = dt_r_words[w1]
        r2 = dt_r_words[w2]
        dot_r = np.dot(M[r1], M[r2])
        norm_1 = np.linalg.norm(M[r1])
        norm_2 = np.linalg.norm(M[r2])
        norm_product = norm_1 * norm_2
        if norm_product == 0.0:
            print("nan")
        score2 =  dot_r / norm_product
        context_similarity.append(score2)
        row += 1
    
    result = stats.spearmanr(men_similarity, context_similarity)
    print(result)
    
def parse_words(word_file):
    file = open(word_file, "r")
    words = []
    for line in file:
        words.append(line.rstrip())
    file.close()
    return words;

def create_matrix(row_words, col_words, doc_file):
    m = len(row_words);
    n = len(col_words);
    M = np.zeros((m, n), dtype = int)
    global dt_r_words
    global dt_c_words
    dt_r_words = build_word_dict(row_words);
    dt_c_words = build_word_dict(col_words);
    file = open(doc_file, "r");
    w = 3
    index = 0
    low = 0
    high = 0
    num_words = 0
    for line in file:
        sentence = "<s> " + line + " </s>"
        words = sentence.split()
        num_words = len(words)
        r_words_in_line = dict();
        c_words_in_line = dict()
        index = 0
        for word in words:
            if word in dt_r_words:
                r_words_in_line[word] = index
            if word in dt_c_words:
                c_words_in_line[index] = word
            index += 1
        for r_word in r_words_in_line:
            m = r_words_in_line[r_word]
            r = dt_r_words[r_word]
            low = m - w
            if low < 0:
                low = 0
            high = m + w + 1
            if high > num_words:
                high = num_words
            for index in range(low, high):
                if (m != index) and (index in c_words_in_line):
                    c_word = c_words_in_line[index]
                    c = dt_c_words[c_word]
                    M[r, c] += 1
    file.close()
    return M    

def build_word_dict(words):
    dt_words = dict()
    index = 0
    for word in words:
        dt_words[word] = index
        index += 1
    return dt_words

def main():
    row_words = parse_words("./data/31190-a1-files/vocab-wordsim.txt")
    col_words = parse_words("./data/31190-a1-files/vocab-25k.txt")
    doc_file = "./data/wiki-1percent.txt";
    M = create_matrix(row_words, col_words, doc_file)
    EvalWS(M)

if __name__ == '__main__':
    main()
    