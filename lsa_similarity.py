import argparse
import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import numpy as np
import time
import csv
import random
import glob
import pickle
import re
from scipy import spatial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/Holmes_Training_Data/',
                        help='data directory containing input data')
    parser.add_argument('--output_dir', type=str, default='./lsa_similarity/',
                        help='directory for outputs')
    parser.add_argument('--test_file', type=str, default='./data/testing_data.csv',help='testing data path')
    args = parser.parse_args()
    return args

def clean_str(string):
    
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"[^A-Za-z,!?.;]", " ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

def preprocess(input_files, data_path):
    all_corpus = []
    for input_file in input_files:
        with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
            print("Preprocessing {}".format(input_file))
            
            corpus = f.read()

            corpus = clean_str(corpus)
            sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?;!])\s",corpus)
            clean_sent = []
            for sent in sentences:
                translator = str.maketrans('','',string.punctuation)
                s = sent.translate(translator)
                tokens = s.strip().split()
                important_words = []
                for word in tokens:
                    if word  not in stopwords.words('english'):
                        important_words.append(word)
                if len(important_words) > 5:
                    clean_sent.append(" ".join(important_words)) 
        all_corpus += clean_sent


    with open(data_path, 'wb') as f:
        pickle.dump(all_corpus, f)

    print("Saved clean corpus at {}".format(data_path))

    return all_corpus
    
def load_train(data_dir,save_dir):
    input_files = glob.glob(os.path.join(data_dir, "*.TXT"))
   
    data_file = os.path.join(save_dir, 'clean_data.pkl')
    

    if os.path.exists(data_file): 
        print("Loading preprocessed training data")
        with open(data_file, 'rb') as f:
            clean_corpus = pickle.load(f)
    else: 
        print("Processing training data")
        clean_corpus = preprocess(input_files, data_file)

    return clean_corpus

def build_lsa_feature(corpus, save_dir):
    dict_file = os.path.join(save_dir, 'lsa_word_dict.pkl')
    if os.path.exists(dict_file):
        print("Loading LSA word dictionary")
        with open(dict_file,'rb') as f:
            word_dict = pickle.load(f)
    else:
        print("Building word features with LSA")

        vectorizer = CountVectorizer(analyzer = 'word', tokenizer = None, preprocessor= None, stop_words = None, max_features = 14000)
        bow_features = vectorizer.fit_transform(corpus)

        lsa = TruncatedSVD(500, algorithm = 'arpack')
        word_features = lsa.fit_transform(bow_features.T)
        word_features = Normalizer(copy=False).fit_transform(word_features)
        
        vocabulary = vectorizer.get_feature_names()
        word_dict = {}
        for i, word in enumerate(vocabulary):
            word_dict[word] = word_features[i]

        with open(dict_file, 'wb') as f:
            pickle.dump(word_dict, f)

    return word_dict


def word2feature(tokens, embeddings):
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
    start = time.time()

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    corpus = load_train(args.data_dir, args.output_dir)

    print("Total training sentences: {}".format(len(corpus)))

    word_dict = build_lsa_feature(corpus, args.output_dir)

    

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
            
            ques_vec = word2feature(tokens, word_dict)

            # calculate total word similarity
            scores =[]
            candidates = [row[x] for x in keys]
            cand_vec = word2feature(candidates, word_dict)
            for word in cand_vec:
               s = total_similarity(word, ques_vec)        
               scores.append(s)
            
            idx = scores.index(max(scores))
            ans = choices[idx]
            prediction.append(ans)
   
    output = os.path.join(args.output_dir, "prediction.csv")
    with open(output, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id','answer'])
        for i, ans in enumerate(prediction):
            writer.writerow([str(i+1), ans])
    print("Output prediction file: {}".format(output))
    
    print("Done in {:3f}s".format(time.time() - start))
