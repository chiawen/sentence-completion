import argparse
import time
import csv
import re
import string

from scipy import spatial
import numpy as np
import gensim

def word2vec(tokens, embeddings):
    dim = embeddings['word'].size

    word_vec = []
    for word in tokens:
        if word in embeddings:
            word_vec.append(embeddings[word])
        else:
            word_vec.append(np.random.uniform(-0.25,0.25,dim))
    return word_vec

def total_similarity(vec, ques_vec):
    score = 0
    for v in ques_vec:
        score += (1 - spatial.distance.cosine(vec, v))

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./data/testing_data.csv',help='testing data path')
    parser.add_argument('--embedding_file', type=str, default='./data/GoogleNews-vectors-negative300.bin', help='pretrained word embeddings')
    parser.add_argument('--output', type=str, default='./prediction.csv', help='output file path')
    args = parser.parse_args()

    start = time.time()
    print("Loading pretrained embeddings: {}".format(args.embedding_file))

    # Load pretrained word embeddings
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(args.embedding_file, binary=True)
    


    keys = ['a)', 'b)', 'c)', 'd)', 'e)']
    choices = ['a', 'b', 'c', 'd', 'e']
    prediction = []

    print("Predicting answers")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            question = row['question']
            translator = str.maketrans('','',string.punctuation)
            question = question.translate(translator)
            tokens = question.split()
            
            # get word2vec embedding
            ques_vec = word2vec(tokens, embeddings)

            # calculate total word similarity
            scores =[]
            candidates = [row[x] for x in keys]
            cand_vec = word2vec(candidates, embeddings)
            for word in cand_vec:
               s = total_similarity(word, ques_vec)        
               scores.append(s)
            
            idx = scores.index(max(scores))
            ans = choices[idx]
            prediction.append(ans)
    
    with open(args.output, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id','answer'])
        for i, ans in enumerate(prediction):
            writer.writerow([str(i+1), ans])
    print("Output prediction file: {}".format(args.output))

    print("Total run time: {}s".format(time.time() - start))

