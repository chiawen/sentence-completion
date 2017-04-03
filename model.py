import numpy as np
import tensorflow as tf

class BasicLSTM():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 5 
        
        self.x = tf.placeholder(tf.int32, [None, args.seq_length-1])
        self.y = tf.placeholder(tf.int32, [None, args.seq_length-1])
        self.keep_prob = tf.placeholder(tf.float32)

        def lstm_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(args.rnn_size)
            cell_dropout = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            return cell_dropout
        
        self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell()] * args.num_layers)
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

	
        with tf.name_scope('embed'):
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.x)
            inputs = tf.nn.dropout(inputs, self.keep_prob)
        
        outputs = []
        state = self.initial_state
        with tf.variable_scope('LSTM'):
            for time_step in range(args.seq_length-1):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        
        self.final_state = state
        

        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.rnn_size])

        with tf.name_scope('nce_loss'):
            proj_w = tf.get_variable("proj_w", [args.vocab_size, args.rnn_size])
            proj_b = tf.get_variable("proj_b", [args.vocab_size])
            
            labels = tf.reshape(self.y, [-1,1])

            # NCE loss
            loss = tf.nn.nce_loss(proj_w, proj_b, labels, output, args.num_sampled, args.vocab_size)
            # sampled_softmax
            # loss = tf.nn.sampled_softmxax_loss(proj_w, proj_b, output, labels, args.num_sampled, args.vocab_size)
                
            self.cost = tf.reduce_sum(loss) / args.batch_size
            # testing
        with tf.name_scope('softmax'):
            softmax_w = tf.transpose(proj_w)
            self.logits = tf.matmul(output, softmax_w) + proj_b
            self.probs = tf.nn.softmax(self.logits)
            seq_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [self.logits],
                    [tf.reshape(self.y, [-1])],
                    [tf.ones([args.batch_size * (args.seq_length-1)], dtype=tf.float32)])

            self.seq_loss = seq_loss

        with tf.name_scope('accuracy'):
            output_words = tf.argmax(self.probs, axis=1)
            output_words = tf.cast(output_words, tf.int32)
            self.equal = tf.equal(output_words, tf.reshape(self.y, [-1]))

