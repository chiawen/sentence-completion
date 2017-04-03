"""
The testing data required for this code can be download from https://goo.gl/9jIYDd

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import pickle
import argparse

import numpy as np
import tensorflow as tf

from utils import TestLoader
from model import BasicLSTM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./data/testing_data.csv',
                        help='testing data path')
    parser.add_argument('--save_dir', type=str, default='./save/',
                        help='directory to store checkpointed models')
    parser.add_argument('--seq_length', type=int, default=40,
                        help='RNN sequence length')
    parser.add_argument('--output', type=str, default='./prediction.csv',
                        help='output path') 
    args = parser.parse_args()
    
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 
    
    infer(args)

def infer(args):
    start = time.time()
    
    # Load testing data
    # ====================================
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
        print('restored args:\n', json.dumps(vars(saved_args), indent=4, separators=(',',':'))) 
    
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        _, vocab = pickle.load(f)    
    data_loader  = TestLoader(args.test_file, vocab, args.seq_length) 
    
    test_data = data_loader.get_data()

    
    # Predict
    # ===================================
    choices = ['a', 'b', 'c', 'd', 'e']
    checkpoint = tf.train.latest_checkpoint(args.save_dir)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = BasicLSTM(saved_args, True)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint)
            
            predictions = []
            print("Predicting answers")
            for question in test_data:
                x_test = np.vstack(question['candidates'])
                feed_dict = {model.x: x_test[ : ,0:-1] , model.y: x_test[ : ,1: ], model.keep_prob: 1.0}
                loss = sess.run(model.seq_loss, feed_dict)
                loss = loss.reshape([5,-1])
                idx_ans = np.argmin(np.sum(loss, axis=1))
                predict = choices[idx_ans]
                predictions.append(predict)

    # Save the predictions to a csv
    out_path = args.output
    with open(out_path, 'w') as out:
        writer = csv.writer(out, delimiter=',')
        writer.writerow(['id','answer'])
        for i, ans in enumerate(predictions):
            writer.writerow([str(i+1), ans])

    print("Saved prediction to {}".format(out_path))
    print("Total run time: {}s".format(time.time() - start))


    


if __name__ == '__main__':
    main()

