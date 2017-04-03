import os
import re
import csv
import glob
import pickle
import string
import collections

import numpy as np
import gensim

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_files = glob.glob(os.path.join(data_dir, "*.TXT"))
        
        input_dir = os.path.dirname(data_dir)
        vocab_file = os.path.join(input_dir, "vocab.pkl")
        data_file = os.path.join(input_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if not (os.path.exists(vocab_file) and os.path.exists(data_file)):
            print("Reading text files")
            self.preprocess(input_files, vocab_file, data_file)
        else:
            print("Loading preprocessed files")
            self.load_preprocessed(vocab_file, data_file)
        self.reset_batch_pointer()

    def clean_str(self, string):
        """
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        words = []
        for s in sentences:
            words += s.split()
        word_counts = collections.Counter(words)
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] > 100]
       
        """
        print("most common words")
        print(vocabulary_inv[:100])
        """
        vocabulary_inv = list(sorted(vocabulary_inv))
        vocabulary_inv = ['<UNK>'] + vocabulary_inv
        
        """ 
        print("sorted words")
        print(vocabulary_inv[:100])
        """
        
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def map_id(self, sentences):
        data = []
        for s in sentences:
            id_s = np.ones(self.seq_length, np.int32) * self.vocab['<END>']
            for i, word in enumerate(s.split()):
                id_s[i] = self.vocab.get(word, 0)
            data.append(id_s)

        data = np.array(data)

        return data

    def preprocess(self, input_files, vocab_file, data_file):
        i = 0
        x_sent = []
        for input_file in input_files:
            with open(input_file, "r", encoding='latin1', errors='ignore') as f:
                data = f.read()
                # text cleaning or make them lower case, etc.
                data = self.clean_str(data) 
                sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?;!])\s",data)
                clean_sent = []
                for sent in sentences:
                    s = re.sub(r"\.", " . ", sent)
                    s = re.sub(r"\s{2,}", " ", s)
                    tokens = s.strip().split()
                    if len(tokens) <= self.seq_length - 2:
                        clean_sent += ['<START> ' + s + ' <END>'] 
                        
                x_sent += clean_sent

        self.vocab, self.words = self.build_vocab(x_sent)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            pickle.dump(self.words, f)
        
        
        """
        print("Vocabulary snippet:")
        print(self.words[:100])
        """

        self.data = self.map_id(x_sent)
        # Save the data to data.npy
        np.save(data_file, self.data)

        """
        print("Text snippet:")
        print(x_sent[:10])
        
        print("Text-to-id snippet:")
        print(self.data[:10])
        """
        self.num_data = len(self.data)
        self.num_batches = self.num_data // self.batch_size

    def load_preprocessed(self, vocab_file, data_file):
        with open(vocab_file, 'rb') as f:
            self.words = pickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.data = np.load(data_file)
        self.num_data = len(self.data)
        self.num_batches = self.num_data // self.batch_size

    def next_batch(self, shuffle=True):
        if self.pointer + self.batch_size > self.num_data:
            self.pointer = 0
        
        if self.pointer == 0 and shuffle:
            perm = np.arange(self.num_data)
            np.random.shuffle(perm)
            self._data = self.data[perm]
            
        x, y = self._data[self.pointer:self.pointer+self.batch_size, 0:-1], self._data[self.pointer:self.pointer+self.batch_size, 1: ]
        self.pointer += self.batch_size
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


def get_vocab_embedding(save_dir, words, embedding_file):
    matrix_file = os.path.join(save_dir, 'embedding.npy')
    
    if os.path.exists(matrix_file):
        print("Loading embedding matrix")
        embedding_matrix = np.load(matrix_file)
    else:
        print("Building embedding matrix")
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
        dim = word_vectors['word'].size
        embedding_matrix = np.zeros(shape=(len(words), dim), dtype='float32')
        
        for i, word in enumerate(words):
            # '<UNK>'
            if i == 0:
                continue
            else:
                if word in word_vectors:
                    embedding_matrix[i] = word_vectors[word]
                else:
                    embedding_matrix[i] = np.random.uniform(-0.25,0.25,dim)

        np.save(matrix_file, embedding_matrix)

    return embedding_matrix


class TestLoader():
    def __init__(self, input_file, vocab_dict, seq_length):
        self.input_file = input_file
        self.vocab_dict = vocab_dict
        self.seq_length = seq_length

        test_dir = os.path.dirname(input_file)
        sent_file = os.path.join(test_dir, "test_sent.pkl")
        test_file = os.path.join(test_dir, "test_data.pkl")

        if not (os.path.exists(sent_file) and os.path.exists(test_file)):
            print("Reading testing file")
            self.preprocess(input_file, sent_file, test_file)
        else:
            print("Loading preprocessed testing file")
            self.load_preprocessed(sent_file, test_file)
        
    
    def clean_str(self, string):
        string = re.sub(r"\[([^\]]+)\]", " ", string)
        string = re.sub(r"\(([^\)]+)\)", " ", string)
        string = re.sub(r"[^A-Za-z0-9,!?.;]", " ", string)
        string = re.sub(r"\.", " . ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r";", " ; ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def map_id(self, test_sent):
        for instance in test_sent:
            candidates = instance['candidates']
            encoded_cand = []
            for sent in candidates:
                # encode a sentence to 40 words
                encoded_sent = np.ones(self.seq_length, np.int32) * self.vocab_dict['<END>'] 
                for i, word in enumerate(sent.split()):
                    if i >= self.seq_length:
                        break
                    encoded_sent[i] = self.vocab_dict.get(word, 0)
                encoded_cand.append(encoded_sent)
            instance['candidates'] = encoded_cand

        return test_sent

    def preprocess(self, input_file, sent_file, test_file):
        test_sent = []
        keys = ['a)', 'b)', 'c)', 'd)', 'e)']
        with open(input_file, "r", encoding='latin1') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                test_instance = {'id':row['id'], 'candidates':[] }
                question = row['question']
                choices = [row[x] for x in keys]
                questions = [question.replace('_____', word) for word in choices]
                test_instance['candidates'] = ['<START> ' + self.clean_str(q) + ' <END>'
                                                for q in questions]
                test_sent.append(test_instance)

        self.sentences = test_sent 
        with open(sent_file, 'wb') as f:
            pickle.dump(test_sent, f)

        self._data = self.map_id(test_sent)
        with open(test_file, 'wb') as f:
            pickle.dump(self._data, f)

    def load_preprocessed(self, sent_file, test_file):
        with open(sent_file, 'rb') as f:
            self.sentences = pickle.load(f)
        with open(test_file, 'rb') as f:
            self._data = pickle.load(f)
    
    def get_data(self):

        return self._data
    
        
