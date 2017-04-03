"""
The data required for this code can be download from https://goo.gl/9jIYDd
$tar -xvzf training_data.tgz

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

from utils import TextLoader
from utils import get_vocab_embedding
from model import BasicLSTM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/Holmes_Training_Data/',
                        help='data directory containing input data')
    parser.add_argument('--log_dir', type=str, default='./logs/',
                       help='directory containing tensorboard logs')    
    parser.add_argument('--save_dir', type=str, default='./save/',
                        help='directory to store checkpointed models')
    parser.add_argument('--embedding_file', type=str, 
                        default='./data/GoogleNews-vectors-negative300.bin', 
                        help='pretrained word embeddings')
    parser.add_argument('--rnn_size', type=int, default=400,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help=' 1 - dropout rate')
    parser.add_argument('--num_sampled', type=int, default=5000,
                        help='number of negative examples to sample')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=40,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=4,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency' )
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 
    
    train(args)

def train(args):
    # Data Preparation
    # ====================================

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    print("Number of sentences: {}" .format(data_loader.num_data))
    print("Vocabulary size: {}" .format(args.vocab_size))

    
    # Check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)
        need_be_same=["rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = pickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
    
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.words, data_loader.vocab), f)
    
    """
    embedding_matrix = get_vocab_embedding(args.save_dir, data_loader.words, args.embedding_file)
    print("Embedding matrix shape:",embedding_matrix.shape)
    """
    
    
    # Training
    # ====================================
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = BasicLSTM(args)
          
            # Define training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), args.grad_clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            
            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in zip(grads, tvars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            
            # Summary for loss
            loss_summary = tf.summary.scalar("loss", model.cost)

            # Train summaries
            merged = tf.summary.merge_all()
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)
            train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables())

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Restore model
            if args.init_from is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Start training
            print("Start training")
            for epoch in range(args.num_epochs):
                data_loader.reset_batch_pointer()
                state = sess.run(model.initial_state)
                for i in range(data_loader.num_batches):
                    start = time.time()
                    x_batch, y_batch = data_loader.next_batch()
                    feed_dict = {model.x: x_batch, model.y: y_batch, model.keep_prob: args.keep_prob }
                    _, step, summary, loss, equal = sess.run([train_op, global_step, merged, model.cost, model.equal], feed_dict)
                   
                    print("training step {}, epoch {}, batch {}/{}, loss: {:.4f}, accuracy: {:.4f}, time/batch: {:.3f}"
                        .format(step, epoch, i, data_loader.num_batches, loss, np.mean(equal), time.time()-start))
                    train_writer.add_summary(summary, step)

                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % args.save_every == 0 or (epoch == args.num_epochs-1 
                        and i == data_loader.num_batches-1): #save for the last result
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        path = saver.save(sess, checkpoint_path, global_step = current_step)
                        print("Saved model checkpoint to {}".format(path))

            train_writer.close()

if __name__ == '__main__':
    main()

